"""Controlled representation/scorer ablations on the frozen enriched snapshot.

Run src.encode_content first, then python -m src.experiment_content.
All grids are declared before test evaluation; historical test reuse is explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.content.experimental import (FieldTextIndex, TextIndex, combine, history_targets, personal_matrices,
                                     ridge_signal, separate_fields, shared_index, similarity_signal)
from src.content.text import movie_text
from src.data_processing.movielens import movie_metadata_from_records, user_ratings_from_records
from src.encode_content import document_hash, experiment_documents
from src.evaluate_content import (candidate_from_dict, core_scores, evaluate_scores, prepare_cases,
                                  read_csv, select_best, summarize)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bootstrap(old, new, repeats=10000, comparisons=5):
    result = {"resamples": repeats, "seed": 42, "unit": "user",
              "familywise_comparisons": comparisons}
    for metric in ("pairwise_accuracy", "ndcg_at_10", "mse", "mae"):
        pairs = [(a[metric], b[metric]) for a, b in zip(old, new, strict=True)
                 if a[metric] is not None and b[metric] is not None]
        a, b = np.array(pairs).T
        rng = np.random.default_rng(42)
        draws = []
        for start in range(0, repeats, 100):
            indices = rng.integers(0, len(a), size=(min(100, repeats - start), len(a)))
            draws.extend(np.sqrt(b[indices].mean(axis=1)) - np.sqrt(a[indices].mean(axis=1))
                         if metric == "mse" else (b[indices] - a[indices]).mean(axis=1))
        delta = np.sqrt(b.mean()) - np.sqrt(a.mean()) if metric == "mse" else (b - a).mean()
        label = "rmse" if metric == "mse" else metric
        result[label] = {"delta": float(delta), "users": len(a),
                         "ci95": np.quantile(draws, [.025, .975]).tolist(),
                         "ci99_bonferroni_five": np.quantile(draws, [.005, .995]).tolist()}
    return result


def write_summary(output, report):
    counts = report["test"]["baseline"]["metrics"]
    lines = ["# Controlled content-model experiment", "",
             f"Same enriched movie snapshot, {counts['users']} users and {counts['ratings']:,} test ratings. "
             "Choices were frozen on validation before this run's test evaluation. "
             "The historical holdout had already been inspected, so results are exploratory.", "",
             "| Family | Validation pairwise | Test pairwise | NDCG@10 | RMSE | MAE | Text scoring seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    validation_by_name = {r["candidate"]["name"]: r for r in report["validation"]}
    for family, row in report["test"].items():
        m = row["metrics"]
        val = validation_by_name[row["candidate"]["name"]]["metrics"]["pairwise_accuracy"]
        lines.append(f"| {family} | {100*val:.3f}% | {100*m['pairwise_accuracy']:.3f}% | "
                     f"{m['ndcg_at_10']:.6f} | {m['rmse']:.6f} | {m['mae']:.6f} | "
                     f"{row['text_scoring_seconds']:.3f} |")
    lines.extend(["", "## Validation choices", "",
                  f"Overall choice: `{report['selection']['overall_selected']['name']}`.", "",
                  f"Best representation: `{report['selection']['representation_selected_on_validation']}`.", ""])
    for family, config in report["selection"]["family_selections"].items():
        lines.append(f"- {family}: `{config['name']}`")
    lines.extend(["", "## Paired differences against current selected baseline", "",
                  "Percentage-point changes in pairwise accuracy; 10,000 user bootstrap resamples, seed 42. "
                  "The 99% intervals apply a Bonferroni correction for the five primary family comparisons.", "",
                  "| Family | Change (pp) | 95% interval (pp) | 99% interval (pp) |",
                  "| --- | ---: | --- | --- |"])
    for family, stats in report["vs_baseline"].items():
        d = stats["pairwise_accuracy"]
        lo, hi = [100*x for x in d["ci95"]]
        low, high = [100*x for x in d["ci99_bonferroni_five"]]
        lines.append(f"| {family} | {100*d['delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{low:+.3f}, {high:+.3f}] |")
    lines.extend(["", "## Timing and interpretation", "",
                  "Text scoring timings include per-user fitting, matrix selection and scoring for all held-out "
                  "test candidates; they exclude the common structured core, metric computation, shared-index "
                  "construction and one-off embedding inference. They are single-run batch timings, not "
                  "end-to-end request latency or full-catalog serving benchmarks. `results.json` records "
                  "index build and encoder times separately.", "",
                  "Core settings, text weight/prior/cap and recency are fixed. Ridge replaces only the text "
                  "adjustment and uses the same rating residual target; it is not a complete replacement of "
                  "the structured model. Semantic cosine is clipped at zero as in the existing LSA scorer.", "",
                  "Shared IDF uses the full static catalog and is transductive. Descriptions are retrospective "
                  "TMDB metadata; timestamp-filtered other-user tags are identical across methods. No held-out "
                  "ratings enter representation fitting, user fitting or selection. Candidate texts pass "
                  "through frozen transforms only. Evaluation ranks held-out rated movies, not all movies.", "",
                  "NDCG, RMSE and MAE intervals are secondary exploratory analyses; the primary selection "
                  "objective remains pairwise accuracy. All intervals and history-size slices are available "
                  "in `results.json`. The serving preset has not been changed.", ""])
    (output / "summary.md").write_text("\n".join(lines))


class Runner:
    def __init__(self, cases, baseline, indexes):
        self.cases, self.baseline, self.indexes = cases, baseline, indexes
        self.core_cache = {}

    def core(self, phase):
        if phase not in self.core_cache:
            self.core_cache[phase] = core_scores(self.cases, replace(self.baseline, text=None), phase)
        return self.core_cache[phase]

    def matrices(self, case, representation, phase):
        catalog = {m.movie_id: m for m in case.movies}
        candidates = [catalog[r.movie_id] for r in getattr(case, phase)]
        if representation == "personal_tfidf":
            return personal_matrices(case.history, catalog, candidates, self.baseline.text)
        index = self.indexes[representation]
        return (index.for_movies([catalog[r.movie_id] for r in case.history]),
                index.for_movies(candidates))

    def score(self, config, phase):
        cores = self.core(phase)
        started = time.perf_counter()
        predictions, evidence_fractions, adjustments = [], [], []
        for case, core in zip(self.cases, cores, strict=True):
            h, c = self.matrices(case, config["representation"], phase)
            residuals, weights = history_targets(case.history, self.baseline.profile)
            if config["scorer"] == "ridge":
                signal = ridge_signal(h, c, residuals, weights, config["alpha"])
            else:
                signal, mass = similarity_signal(h, c, residuals, weights,
                                                self.baseline.text.regularization_strength,
                                                config.get("top_k"))
                evidence_fractions.extend(mass > 0)
            predictions.append(combine(core, signal, self.baseline.text))
            adjustments.extend(np.clip(signal * self.baseline.text.weight,
                                       -self.baseline.text.max_adjustment,
                                       self.baseline.text.max_adjustment))
        seconds = time.perf_counter() - started
        metrics, per_user = evaluate_scores(self.cases, predictions, phase)
        return ({"candidate": config, "metrics": metrics, "text_scoring_seconds": seconds,
                 "mean_abs_text_adjustment": float(np.mean(np.abs(adjustments))),
                 "text_adjustment_std": float(np.std(adjustments)),
                 "nonzero_similarity_fraction": float(np.mean(evidence_fractions)) if evidence_fractions else None},
                per_user, predictions)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline = candidate_from_dict(json.loads(Path(args.preset).read_text())["candidate"])
    if baseline.text.representation != "tfidf":
        raise ValueError("This controlled experiment requires the selected TF-IDF baseline")
    movies = movie_metadata_from_records(read_csv(Path(args.movies_file)))
    cases, skipped = prepare_cases(user_ratings_from_records(read_csv(Path(args.data_dir) / "ratings.csv")),
                                   movies, read_csv(Path(args.data_dir) / "tags.csv"))
    print(f"Prepared {len(cases)} users; {len(movies)} movies", flush=True)
    documents = experiment_documents(movies, cases)
    indexes, build_seconds, encoders = {}, {}, {}
    t0 = time.perf_counter()
    indexes["shared_tfidf"] = shared_index(movies, documents, baseline.text)
    build_seconds["shared_tfidf"] = time.perf_counter() - t0

    # Each field has its own IDF/vocabulary; both are fitted on catalog metadata only.
    all_movies = list(movies) + [m for c in cases for m in c.movies]
    t0 = time.perf_counter()
    fields = {}
    for source in ("plot", "keywords"):
        field_docs = sorted({movie_text(m, source) for m in all_movies})
        fields[source] = shared_index(movies, field_docs, baseline.text, source)
    # Field identity must survive even if different fields concatenate to the
    # same combined text (e.g. keyword-only "space" versus plot-only "space").
    representatives = {(movie_text(m, "plot"), movie_text(m, "keywords")): m for m in all_movies}
    field_keys = sorted(representatives)
    ordered = [representatives[d] for d in field_keys]
    plots, keywords = (fields[s].for_movies(ordered) for s in ("plot", "keywords"))
    for weight in (0., .25, .5, .75, 1.):
        name = f"fields-plot{weight:g}"
        indexes[name] = FieldTextIndex(separate_fields(plots, keywords, weight),
                                      {d: i for i, d in enumerate(field_keys)})
    build_seconds["separate_fields_all_five"] = time.perf_counter() - t0
    for name in args.encoders:
        prefix = Path(args.embedding_cache) / name
        metadata = json.loads(prefix.with_suffix(".json").read_text())
        if metadata["documents_sha256"] != document_hash(documents):
            raise ValueError(f"{name}: embedding cache has different documents")
        if metadata["matrix_sha256"] != sha(prefix.with_suffix(".npy")):
            raise ValueError(f"{name}: embedding matrix hash mismatch")
        matrix = np.load(prefix.with_suffix(".npy"), allow_pickle=False)
        if len(matrix) != len(documents) or not np.isfinite(matrix).all():
            raise ValueError("Invalid frozen embeddings")
        indexes[name] = TextIndex(matrix, {d: i for i, d in enumerate(documents)})
        encoders[name] = metadata

    runner = Runner(cases, baseline, indexes)
    validation, selections = [], {}

    def validate(config):
        row, _, _ = runner.score(config, "validation")
        validation.append(row)
        write_json(output / "validation.json", validation)
        print(f"{config['name']}: validation pairwise={row['metrics']['pairwise_accuracy']:.6f} "
              f"NDCG={row['metrics']['ndcg_at_10']:.6f}", flush=True)
        return row

    def similarity_config(name):
        return {"name": name, "representation": name, "scorer": "similarity"}

    selections["baseline"] = validate(similarity_config("personal_tfidf"))
    selections["shared_catalog"] = validate(similarity_config("shared_tfidf"))
    selections["separate_fields"] = select_best([validate(similarity_config(f"fields-plot{w:g}"))
                                                for w in (0., .25, .5, .75, 1.)])
    semantic = [validate(similarity_config(name)) for name in args.encoders]
    if semantic:
        selections["semantic"] = select_best(semantic)
    best_representation = select_best(list(selections.values()))["candidate"]["representation"]
    print(f"Best representation from validation: {best_representation}", flush=True)
    selections["ridge"] = select_best([
        validate({"name": f"{best_representation}+ridge-a{alpha:g}",
                  "representation": best_representation, "scorer": "ridge", "alpha": alpha})
        for alpha in (.1, 1., 10., 100.)])
    selections["top_k"] = select_best([
        validate({"name": f"{best_representation}+top{k}", "representation": best_representation,
                  "scorer": "similarity", "top_k": k}) for k in (5, 10, 20, 50)])
    selected = select_best(list(selections.values()))
    frozen = {"baseline_preset": json.loads(Path(args.preset).read_text()),
              "representation_selected_on_validation": best_representation,
              "family_selections": {k: v["candidate"] for k, v in selections.items()},
              "overall_selected": selected["candidate"],
              "objective": "macro validation pairwise accuracy; RMSE then name tie-break"}
    # Must be persisted before any test labels are evaluated.
    write_json(output / "selection.json", frozen)
    print("Validation choices frozen; evaluating test now.", flush=True)
    tests, user_rows, prediction_arrays = {}, {}, {}
    for name, selection in selections.items():
        row, per_user, predictions = runner.score(selection["candidate"], "test")
        tests[name], user_rows[name] = row, per_user
        prediction_arrays[name] = np.concatenate(predictions)
        print(f"{name}: TEST pairwise={row['metrics']['pairwise_accuracy']:.6f}", flush=True)
    # Baseline reconstruction must agree with the already recorded experiment.
    previous = json.loads(Path(args.reference_report).read_text())
    if previous["movie_metadata"]["sha256"] != sha(args.movies_file):
        raise ValueError("Snapshot differs from reference report")
    for metric in ("pairwise_accuracy", "ndcg_at_10", "rmse", "mae"):
        if abs(tests["baseline"]["metrics"][metric] - previous["test"]["selected"]["metrics"][metric]) > 1e-10:
            raise AssertionError(f"Baseline parity failed: {metric}")
    comparisons = {name: bootstrap(user_rows["baseline"], rows)
                   for name, rows in user_rows.items() if name != "baseline"}
    representation_family = next(k for k, v in selections.items()
                                 if v["candidate"]["name"] == best_representation)
    scorer_comparisons = {name: bootstrap(user_rows[representation_family], user_rows[name])
                          for name in ("ridge", "top_k")}
    slices = {}
    for label, low, high in (("history_5_to_19", 5, 19), ("history_20_to_99", 20, 99),
                              ("history_100_plus", 100, float("inf"))):
        indices = [i for i, c in enumerate(cases) if low <= len(c.history) <= high]
        slices[label] = {name: summarize([rows[i] for i in indices]) for name, rows in user_rows.items()}
    with (output / "test_users.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "history_count", "model", *user_rows["baseline"][0]])
        writer.writeheader()
        for name, rows in user_rows.items():
            for case, metrics in zip(cases, rows, strict=True):
                writer.writerow({"user_id": case.user_id, "history_count": len(case.history), "model": name, **metrics})
    np.savez_compressed(output / "predictions.npz", **prediction_arrays)
    report = {"protocol": {
        "split": previous["protocol"]["split"], "profile": previous["protocol"]["profile"],
        "selection": frozen["objective"], "test_status": "previously inspected holdout; exploratory comparisons",
        "metadata": "fixed retrospective TMDB catalog; full-catalog vocabulary is transductive",
        "text": "same documents for every method, including permitted historical other-user tags; shared IDF fitted only on static TMDB metadata",
        "scoring": "core, recency, target rating residual, text prior/weight/cap fixed to enriched preset",
        "ridge": "weighted per-user ridge on same text vectors/residuals, no intercept; replaces bounded text component only",
        "grid": {"plot_weights": [0, .25, .5, .75, 1], "ridge_alpha": [.1, 1, 10, 100], "top_k": [5, 10, 20, 50]},
        "candidate_set": "same held-out rated movies; not full-catalog retrieval",
        "inference": "paired user bootstrap; unadjusted 95% and approximate Bonferroni 99% intervals for five primary pairwise comparisons",
        "skipped_users": skipped, "baseline_parity": True},
        "input_sha256": {str(p): sha(p) for p in (args.movies_file, Path(args.data_dir) / "ratings.csv",
                                                  Path(args.data_dir) / "tags.csv", args.preset)},
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
                         for p in ("numpy", "scipy", "scikit-learn", "torch", "sentence-transformers", "transformers")}},
        "command": ["python", "-m", "src.experiment_content", *sys.argv[1:]],
        "source_sha256": {p: sha(p) for p in ("src/experiment_content.py", "src/encode_content.py",
                                              "src/content/experimental.py", "src/evaluate_content.py")},
        "build_seconds": build_seconds, "encoders": encoders,
        "representation_shapes": {name: {"documents": index.matrix.shape[0],
                                            "dimensions": index.matrix.shape[1],
                                            "dtype": str(index.matrix.dtype)}
                                  for name, index in indexes.items()},
        "selection": frozen, "validation": validation, "test": tests,
        "vs_baseline": comparisons, "scorers_vs_best_representation": scorer_comparisons,
        "test_slices": slices, "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "results.json", report)
    write_summary(output, report)
    print(f"Saved {output / 'results.json'} in {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--movies-file", default="data/movies_enriched.csv")
    parser.add_argument("--preset", default="reports/tmdb-full/selected_config.json")
    parser.add_argument("--reference-report", default="reports/tmdb-full/results.json")
    parser.add_argument("--embedding-cache", default=".cache/content-experiments")
    parser.add_argument("--encoders", nargs="*", choices=("minilm", "qwen"), default=["minilm", "qwen"])
    parser.add_argument("--output", default="reports/content-controlled")
    with threadpool_limits(limits=1):
        run(parser.parse_args())
