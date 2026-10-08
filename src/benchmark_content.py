"""Measure complete local recommendation requests, not just text scoring."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.content import ContentModel
from src.data_processing.movielens import movie_metadata_from_records, user_ratings_from_records
from src.data_processing.text_metadata import with_keywords_as_of
from src.evaluate_content import candidate_from_dict, read_csv
from src.evaluation.catalog import development_cases
from src.evaluation.confirmation import sha


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def load_inputs(args):
    cases, _ = development_cases(user_ratings_from_records(read_csv(Path(args.ratings))))
    movies = movie_metadata_from_records(read_csv(Path(args.movies)))
    tags = read_csv(Path(args.tags))
    preset = candidate_from_dict(json.loads(Path(args.preset).read_text())["candidate"])
    return cases, movies, tags, preset


def build_model(case, movies, tags, preset):
    # Benchmark the actual serving catalog, including the whole current snapshot.
    catalog = with_keywords_as_of(movies, tags, case.cutoff, case.user_id)
    return ContentModel(case.history, catalog, preset.scoring, preset.profile, preset.text)


def request(model, user_id):
    started = time.perf_counter()
    recommendations = model.recommend(user_id, 10)
    response = json.dumps([r.to_dict() for r in recommendations])
    return time.perf_counter()-started, response


def distribution(values):
    return {"requests": len(values), "p50_ms": float(np.quantile(values, .5)*1000),
            "p95_ms": float(np.quantile(values, .95)*1000), "max_ms": float(max(values)*1000)}


def run(args):
    cases, movies, tags, preset = load_inputs(args)
    if args.worker:
        case = next(c for c in cases if c.user_id == args.worker)
        model = build_model(case, movies, tags, preset)
        _, response = request(model, case.user_id)
        print(json.dumps({"peak_rss_mb": peak_rss_mb(), "response": json.loads(response)}))
        return
    ordered = sorted(cases, key=lambda c: (len(c.history), c.user_id))
    chosen = {"short": ordered[0], "median": ordered[len(ordered)//2], "long": ordered[-1]}
    samples, cold_memory, models, expected = [], [], {}, {}
    for label, case in chosen.items():
        command = [sys.executable, "-m", "src.benchmark_content", "--worker", str(case.user_id),
                   "--movies", args.movies, "--ratings", args.ratings, "--tags", args.tags, "--preset", args.preset]
        for repetition in range(args.cold_repeats):
            started = time.perf_counter()
            child = subprocess.run(command, capture_output=True, text=True, check=True)
            seconds = time.perf_counter()-started
            payload = json.loads(child.stdout)
            cold_memory.append(payload["peak_rss_mb"])
            samples.append({"cohort": label, "mode": "cold_process", "seconds": seconds})
            cold_response = json.dumps(payload["response"], sort_keys=True)
            if label in expected and cold_response != expected[label]:
                raise AssertionError("Cold responses are not deterministic")
            expected[label] = cold_response
        models[label] = build_model(case, movies, tags, preset)
        _, response = request(models[label], case.user_id)  # Warm up before timed trials.
        if json.dumps(json.loads(response), sort_keys=True) != expected[label]:
            raise AssertionError("Warm and cold serving responses differ")
        for repetition in range(args.warm_repeats):
            seconds, _ = request(models[label], case.user_id)
            samples.append({"cohort": label, "mode": "warm_request", "seconds": seconds})
        print(f"Benchmarked {label} history ({len(case.history)} ratings)", flush=True)
    concurrency_started = time.perf_counter()

    def queued_request(label, submitted):
        duration, response = request(models[label], chosen[label].user_id)
        if json.dumps(json.loads(response), sort_keys=True) != expected[label]:
            raise AssertionError("Concurrent serving response differs from serial response")
        return {"cohort": label, "mode": "concurrent_queued_request", "seconds": time.perf_counter()-submitted,
                "service_seconds": duration}

    # Already-warmed models are shared. Include queue time in latency, and report
    # this finite burst explicitly; it is not a steady-state production load test.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for _ in range(args.concurrent_rounds):
            for label in chosen:
                futures.append(executor.submit(queued_request, label, time.perf_counter()))
        concurrent = [f.result() for f in futures]
    concurrent_seconds = time.perf_counter()-concurrency_started
    samples.extend(concurrent)
    by_mode = {mode: distribution([s["seconds"] for s in samples if s["mode"] == mode])
               for mode in ("cold_process", "warm_request", "concurrent_queued_request")}
    report = {"protocol": {
        "scope": "public ContentModel.recommend over full current catalog; top 10 and JSON serialization",
        "cold": "fresh subprocess per request; includes interpreter startup, CSV/config loading, profile fit, scoring, sorting, serialization, IPC",
        "warm": "in-process request after model/profile warm-up; includes scoring, sorting and serialization",
        "concurrency": "finite burst, shared warmed models; latency includes thread-pool queue time",
        "limits": "local microbenchmark; small cold sample, no HTTP stack, no production SLO claim",
        "cold_repeats_per_history": args.cold_repeats, "warm_repeats_per_history": args.warm_repeats,
        "concurrent_rounds": args.concurrent_rounds, "workers": args.workers,
        "sample_history_counts": {k: len(c.history) for k, c in chosen.items()},
        "catalog_size": len(movies), "response_parity": True},
        "latency": by_mode,
        "latency_by_history": {label: {mode: distribution([s["seconds"] for s in samples if s["cohort"] == label and s["mode"] == mode])
                                       for mode in by_mode} for label in chosen},
        "concurrent_requests_per_second": len(concurrent)/concurrent_seconds,
        "cold_process_peak_rss_mb": {"max": max(cold_memory), "median": float(np.median(cold_memory))},
        "warm_process_peak_rss_mb": peak_rss_mb(),
        "memory_note": "OS process lifetime high-water RSS, not allocation delta; concurrent cold workers are not measured",
        "input_sha256": {p: sha(p) for p in (args.movies, args.ratings, args.tags, args.preset)},
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "blas_threads": 1}}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "request_samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cohort", "mode", "seconds", "service_seconds"])
        writer.writeheader()
        writer.writerows(samples)
    print(f"Saved {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", default="data/movies_enriched.csv")
    parser.add_argument("--ratings", default="data/ratings.csv")
    parser.add_argument("--tags", default="data/tags.csv")
    parser.add_argument("--preset", default="reports/tmdb-full/selected_config.json")
    parser.add_argument("--output", default="reports/evaluation/latency")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=10)
    parser.add_argument("--concurrent-rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.cold_repeats, args.warm_repeats, args.concurrent_rounds, args.workers) < 1:
        parser.error("repeat counts and workers must be positive")
    with threadpool_limits(limits=1):
        run(args)
