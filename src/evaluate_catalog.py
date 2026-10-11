"""Full-catalog development benchmark and preregistered future confirmation.

python -m src.evaluate_catalog develop
python -m src.evaluate_catalog register --plan reports/evaluation/future-plan.json
python -m src.evaluate_catalog confirm --plan ... --outcomes data/future_ratings.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.content import ContentModel, TextProfile
from src.data_processing.movielens import movie_metadata_from_records, user_ratings_from_records
from src.data_processing.text_metadata import with_keywords_as_of
from src.evaluate_content import blend, candidate_from_dict, read_csv
from src.evaluation.catalog import (aggregate, candidate_ids, development_cases, first_observed,
                                    full_catalog_metrics, future_cases, paired_inference)
from src.evaluation.confirmation import claim_once, register_plan, sha, source_hashes, verify_plan


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def case_catalog(case, movies, availability, tags):
    available = [m for m in movies if availability.get(m.movie_id, float("inf")) <= case.cutoff]
    return with_keywords_as_of(available, tags, case.cutoff, case.user_id)


def score_models(case, catalog, ids, preset):
    """Uses history/candidate metadata only; never reads case.outcomes."""
    model = ContentModel(case.history, catalog, preset.scoring, preset.profile)
    predicted = model.predict((case.user_id,), ids, include_debug=True)
    raw = np.array([r.debug.unclamped_score for r in predicted])
    mapping = {m.movie_id: m for m in catalog}
    if preset.text:
        text = TextProfile(case.history, mapping, preset.text, preset.profile)
        evidence = text.evidence(mapping[m] for m in ids)
        selected = blend([raw], [evidence], preset.text)[0]
    else:
        selected = np.clip(raw, 0, 5)
    return {"structured_core": np.clip(raw, 0, 5), "selected": selected}


def run_benchmark(cases, movies, tags, availability, preset, output, protocol, minimum_effect=.01,
                  export_context=None, input_hashes=None):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    movie_ids = {m.movie_id for m in movies}
    if len(movie_ids) != len(movies):
        raise ValueError("Duplicate movie IDs")
    if any(r.movie_id not in movie_ids for c in cases for r in (*c.history, *c.outcomes)):
        raise ValueError("An interaction has no movie metadata record")
    rows = {"structured_core": {}, "selected": {}}
    recommended = {name: {10: set(), 20: set()} for name in rows}
    all_candidates = set()
    context = None
    if export_context:
        Path(export_context).parent.mkdir(parents=True, exist_ok=True)
        context = gzip.open(export_context, "wt", encoding="utf-8")
    try:
        for i, case in enumerate(cases):
            ids = candidate_ids(case, availability, movie_ids)
            all_candidates.update(ids)
            catalog = case_catalog(case, movies, availability, tags)
            if context:
                # Target outcomes deliberately never exported to model developers.
                context.write(json.dumps({"user_id": case.user_id, "cutoff": case.cutoff,
                                          "history": [asdict(r) for r in case.history],
                                          "candidate_ids": ids}) + "\n")
            predictions = score_models(case, catalog, ids, preset)
            for name, scores in predictions.items():
                metrics, top = full_catalog_metrics(ids, scores, case.outcomes)
                rows[name][case.user_id] = {"history_count": len(case.history), **metrics}
                for k in (10, 20):
                    recommended[name][k].update(top[:k])
            if (i+1) % 25 == 0 or i+1 == len(cases):
                print(f"Evaluated {i+1}/{len(cases)} users; {time.perf_counter()-started:.1f}s", flush=True)
    finally:
        if context:
            context.close()
    results = {name: aggregate(list(values.values())) for name, values in rows.items()}
    for name in results:
        results[name]["recommended_catalog_coverage"] = {
            str(k): len(recommended[name][k])/len(all_candidates) if all_candidates else None for k in (10, 20)}
    stats = {metric: paired_inference(rows["structured_core"], rows["selected"], metric,
                                     minimum_effect=minimum_effect)
             for metric in ("ndcg_at_10", "recall_at_10", "ndcg_at_20", "recall_at_20")}
    slices = {}
    for label, lower, upper in (("history_5_to_19", 5, 19), ("history_20_to_99", 20, 99),
                                ("history_100_plus", 100, float("inf"))):
        slices[label] = {name: aggregate([r for r in values.values() if lower <= r["history_count"] <= upper])
                         for name, values in rows.items()}
    report = {"protocol": {"primary_metric": "ndcg_at_10", "ks": [10, 20],
                           "positive_rating_threshold": 4., "selection": "fixed preset; no tuning on this run",
                           "relevance": "known positives only; unobserved items are unjudged, not confirmed negatives",
                           "denominator": "eligible held-out positives; all-observed-positive recall also reported",
                           "ties": "score descending, movie ID ascending for every model",
                           "secondary_metrics": "exploratory; primary inference is NDCG@10 only",
                           **protocol}, "input_sha256": input_hashes or {}, "source_sha256": source_hashes(),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(), "blas_threads": 1},
              "preset": preset.to_dict(), "metrics": results, "selected_vs_core": stats,
              "history_slices": slices, "unique_eligible_movies": len(all_candidates),
              "elapsed_seconds": time.perf_counter()-started}
    write_json(output / "results.json", report)
    with (output / "test_users.csv").open("w", newline="") as f:
        first = next(iter(rows["selected"].values()), None)
        if first:
            writer = csv.DictWriter(f, fieldnames=["user_id", "model", *first])
            writer.writeheader()
            for name, values in rows.items():
                for user_id, metrics in values.items():
                    writer.writerow({"user_id": user_id, "model": name, **metrics})
    print(f"Saved {output / 'results.json'}", flush=True)
    return report


def develop(args):
    movies = movie_metadata_from_records(read_csv(Path(args.movies)))
    ratings = user_ratings_from_records(read_csv(Path(args.ratings)))
    tags = read_csv(Path(args.tags))
    cases, skipped = development_cases(ratings)
    if args.limit:
        cases = cases[:args.limit]
    preset = candidate_from_dict(json.loads(Path(args.preset).read_text())["candidate"])
    return run_benchmark(cases, movies, tags, first_observed(ratings), preset, args.output,
                         {"mode": "development", "test_status": "previously inspected dataset; not confirmation",
                          "split": "per-user first 60% history, next 20% development outcomes; timestamp groups intact",
                          "catalog_availability": "first observed rating <= each prediction cutoff; conservative proxy",
                          "metadata_temporality": "retrospective snapshot; not a historical metadata backtest",
                          "skipped_users": skipped, "smoke_limit": args.limit},
                         args.minimum_effect, args.export_context,
                         {p: sha(p) for p in (args.movies, args.ratings, args.tags, args.preset)})


def confirm(args):
    plan = json.loads(Path(args.plan).read_text())
    verify_plan(plan)  # Fail before opening outcome data, including on early attempts.
    claim = claim_once(plan, args.state_dir)
    try:
        files = {k: v["path"] for k, v in plan["files"].items()}
        history = user_ratings_from_records(read_csv(Path(files["history"])))
        outcomes = user_ratings_from_records(read_csv(Path(args.outcomes)))
        if not outcomes:
            raise ValueError("No future outcomes supplied")
        counts = Counter(r.user_id for r in history)
        users = sorted(u for u, n in counts.items() if n >= plan["minimum_history"])
        cases, unknown = future_cases(history, outcomes, plan["history_cutoff"], plan["outcome_end"], users)
        movies = movie_metadata_from_records(read_csv(Path(files["movies"])))
        availability = {m.movie_id: plan["history_cutoff"] for m in movies}
        report = run_benchmark(cases, movies, read_csv(Path(files["tags"])), availability,
                               candidate_from_dict(json.loads(Path(files["preset"]).read_text())["candidate"]),
                               args.output,
                               {"mode": "future_confirmation", "plan_id": plan["plan_id"],
                                "split": "single frozen global cutoff; future outcome window",
                                "metadata_temporality": "frozen at registration, before outcome window",
                                "catalog_availability": plan["catalog_availability"],
                                "history_cutoff": plan["history_cutoff"], "outcome_end": plan["outcome_end"],
                                "out_of_cohort_users": unknown}, plan["minimum_worthwhile_effect"],
                               input_hashes={**{v["path"]: v["sha256"] for v in plan["files"].values()},
                                             args.outcomes: sha(args.outcomes)})
        write_json(claim, {"plan_id": plan["plan_id"], "status": "complete", "report": str(args.output),
                           "outcomes_sha256": sha(args.outcomes)})
        return report
    except Exception:
        write_json(claim, {"plan_id": plan["plan_id"], "status": "failed after opening; audit before any rerun"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("develop", "register"):
        sub = commands.add_parser(name)
        sub.add_argument("--movies", default="data/movies_enriched.csv")
        sub.add_argument("--ratings", default="data/ratings.csv")
        sub.add_argument("--tags", default="data/tags.csv")
        sub.add_argument("--preset", default="reports/tmdb-full/selected_config.json")
        sub.add_argument("--minimum-effect", type=float, default=.01)
        if name == "develop":
            sub.add_argument("--output", default="reports/evaluation/development")
            sub.add_argument("--export-context", default=".cache/evaluation/development-contexts.jsonl.gz")
            sub.add_argument("--limit", type=int, help="Smoke run only; never a full benchmark")
        else:
            sub.add_argument("--plan", default="reports/evaluation/future-plan.json")
            sub.add_argument("--days", type=int, default=30)
    sub = commands.add_parser("confirm")
    sub.add_argument("--plan", required=True)
    sub.add_argument("--outcomes", required=True)
    sub.add_argument("--output", default="reports/evaluation/confirmation")
    sub.add_argument("--state-dir", default=".cache/evaluation/confirmations")
    args = parser.parse_args()
    if args.command == "develop":
        if args.limit is not None and args.limit <= 0:
            parser.error("--limit must be positive")
        develop(args)
    elif args.command == "register":
        plan = register_plan(args.movies, args.ratings, args.tags, args.preset, args.plan,
                             args.days, args.minimum_effect)
        print(f"Registered {plan['plan_id']}; future window ends at Unix {plan['outcome_end']}")
    else:
        confirm(args)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
