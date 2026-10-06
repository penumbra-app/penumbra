"""Reproducible Week 6 validation search and untouched final holdout report.

Run: python -m src.evaluate_content --output-dir reports/week6
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import sklearn
import scipy
from sklearn.metrics import ndcg_score
from threadpoolctl import threadpool_limits

from src.content import ContentModel, ProfileConfig, ScoringConfig, TextConfig, TextProfile
from src.content.schemas import MovieMetadata, UserRating
from src.data_processing.movielens import movie_metadata_from_records, user_ratings_from_records
from src.data_processing.text_metadata import with_keywords_as_of
from src.enrich_movies import metadata_summary


@dataclass(frozen=True)
class Candidate:
    name: str
    profile: ProfileConfig = ProfileConfig()
    scoring: ScoringConfig = ScoringConfig()
    text: TextConfig | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Case:
    user_id: int
    history: tuple[UserRating, ...]
    validation: tuple[UserRating, ...]
    test: tuple[UserRating, ...]
    movies: tuple[MovieMetadata, ...]


def temporal_split(ratings):
    """Approximate 60/20/20 by count; never divide a timestamp group."""
    ratings = tuple(ratings)
    if not ratings or any(r.timestamp is None for r in ratings):
        raise ValueError("Temporal evaluation requires timestamped ratings")
    ordered = tuple(sorted(ratings, key=lambda r: (r.timestamp, r.movie_id)))
    if len({r.movie_id for r in ordered}) != len(ordered):
        raise ValueError("Duplicate movie ratings are not allowed")
    times = [r.timestamp for r in ordered]
    train_end = bisect_right(times, times[max(0, int(len(times) * .6) - 1)])
    validation_end = bisect_right(times, times[max(0, int(len(times) * .8) - 1)])
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as source:
        return tuple(csv.DictReader(source))


def prepare_cases(ratings, movies, tags):
    grouped = {}
    for rating in ratings:
        grouped.setdefault(rating.user_id, []).append(rating)
    cases = []
    skipped = 0
    for user_id, history in sorted(grouped.items()):
        train, validation, test = temporal_split(history)
        if len(train) < 5 or len(validation) < 2 or len(test) < 2:
            skipped += 1
            continue
        ids = {r.movie_id for r in history}
        catalog = with_keywords_as_of(
            (m for m in movies if m.movie_id in ids), tags,
            max(r.timestamp for r in train), exclude_user_id=user_id,
        )
        if not ids <= {m.movie_id for m in catalog}:
            raise ValueError("Every evaluated rating must have a movie record")
        cases.append(Case(user_id, train, validation, test, catalog))
    return tuple(cases), skipped


def user_metrics(actual, scores):
    actual, scores = np.asarray(actual, dtype=float), np.asarray(scores, dtype=float)
    if actual.shape != scores.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("Ratings and scores must be matching non-empty vectors")
    if not np.isfinite(actual).all() or not np.isfinite(scores).all():
        raise ValueError("Ratings and scores must be finite")
    i, j = np.triu_indices(len(actual), 1)
    mask = actual[i] != actual[j]
    actual_delta, predicted_delta = (actual[i] - actual[j])[mask], (scores[i] - scores[j])[mask]
    ties = np.abs(predicted_delta) <= 1e-12
    credit = np.where(ties, .5, (actual_delta * predicted_delta > 0).astype(float))
    return {
        "pairwise_accuracy": float(credit.mean()) if len(credit) else None,
        "pairs": len(credit), "ratings": len(actual),
        "mse": float(np.mean((scores - actual) ** 2)),
        "mae": float(np.mean(np.abs(scores - actual))),
        "ndcg_at_10": float(ndcg_score([2 ** actual - 1], [scores], k=10)) if len(actual) > 1 else 1.0,
    }


def summarize(per_user):
    valid = [m for m in per_user if m["pairwise_accuracy"] is not None]
    if not valid:
        raise ValueError("No unequal-rating pairs are available for evaluation")
    return {
        "pairwise_accuracy": float(np.mean([m["pairwise_accuracy"] for m in valid])),
        "ndcg_at_10": float(np.mean([m["ndcg_at_10"] for m in per_user])),
        "rmse": float(np.sqrt(np.mean([m["mse"] for m in per_user]))),
        "mae": float(np.mean([m["mae"] for m in per_user])),
        "users": len(per_user), "ranking_users": len(valid),
        "ratings": sum(m["ratings"] for m in per_user),
        "pairs": sum(m["pairs"] for m in per_user),
    }


def evaluate_scores(cases, score_arrays, phase):
    per_user = [user_metrics([r.rating for r in getattr(case, phase)], scores)
                for case, scores in zip(cases, score_arrays, strict=True)]
    return summarize(per_user), per_user


def select_best(rows):
    """Primary objective: macro pairwise accuracy; RMSE then name break ties."""
    return min(rows, key=lambda row: (-row["metrics"]["pairwise_accuracy"],
                                     row["metrics"]["rmse"], row["candidate"]["name"]))


def paired_bootstrap(baseline, selected, repeats=2000):
    deltas = np.array([b["pairwise_accuracy"] - a["pairwise_accuracy"]
                       for a, b in zip(baseline, selected, strict=True)
                       if a["pairwise_accuracy"] is not None and b["pairwise_accuracy"] is not None])
    rng = np.random.default_rng(42)
    means = np.array([rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(repeats)])
    return {"mean_delta": float(deltas.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist(), "resamples": repeats}


def core_grid():
    candidates = [Candidate("baseline")]
    for regularization in (1., 5., 20.):
        for genre in (.5, 1., 2.):
            for era in (0., .25):
                candidate = Candidate(
                    f"core-r{regularization:g}-g{genre:g}-e{era:g}",
                    ProfileConfig(regularization_strength=regularization),
                    ScoringConfig(genre_weight=genre, release_era_weight=era),
                )
                if candidate.profile == candidates[0].profile and candidate.scoring == candidates[0].scoring:
                    continue
                candidates.append(candidate)
    return candidates


def core_scores(cases, candidate, phase):
    arrays = []
    for case in cases:
        model = ContentModel(case.history, case.movies, candidate.scoring, candidate.profile)
        results = model.predict((case.user_id,), tuple(r.movie_id for r in getattr(case, phase)), True)
        # Blend text with the unclamped core score; clamp exactly once at the end.
        arrays.append(np.array([r.debug.unclamped_score for r in results]))
    return arrays


def text_evidence(cases, candidate, phase):
    evidence = []
    for case in cases:
        catalog = {m.movie_id: m for m in case.movies}
        profile = TextProfile(case.history, catalog, candidate.text, candidate.profile)
        evidence.append(profile.evidence(catalog[r.movie_id] for r in getattr(case, phase)))
    return evidence


def blend(arrays, evidence, config):
    results = []
    for scores, (numerator, mass) in zip(arrays, evidence, strict=True):
        raw = np.divide(numerator, mass + config.regularization_strength,
                        out=np.zeros_like(mass), where=mass > 0)
        results.append(np.clip(scores + np.clip(raw * config.weight,
                                               -config.max_adjustment, config.max_adjustment), 0, 5))
    return results


def row_for(candidate, cases, scores, seconds):
    metrics, _ = evaluate_scores(cases, scores, "validation")
    return {"candidate": candidate.to_dict(), "metrics": metrics, "seconds": seconds}


def candidate_from_dict(data):
    return Candidate(data["name"], ProfileConfig(**data["profile"]),
                     ScoringConfig(**data["scoring"]),
                     TextConfig(**data["text"]) if data["text"] else None)


def run(data_directory, output_directory, movies_file=None):
    started = time.perf_counter()
    directory, output = Path(data_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    ratings = user_ratings_from_records(read_csv(directory / "ratings.csv"))
    movie_path = Path(movies_file) if movies_file is not None else directory / "movies.csv"
    movie_rows = read_csv(movie_path)
    movies = movie_metadata_from_records(movie_rows)
    metadata = metadata_summary(movie_rows)
    if metadata["enriched_movies"]:
        print("Using retrospective TMDB enrichment; results are not a strictly historical metadata backtest.", flush=True)
    tags = read_csv(directory / "tags.csv")
    cases, skipped = prepare_cases(ratings, movies, tags)
    print(f"Prepared {len(cases)} users; skipped {skipped} with insufficient distinct-time holdouts.", flush=True)
    rows, core_cache = [], {}
    for candidate in core_grid():
        t0 = time.perf_counter()
        scores = core_scores(cases, candidate, "validation")
        core_cache[candidate.name] = scores
        row = row_for(candidate, cases, [np.clip(a, 0, 5) for a in scores], time.perf_counter() - t0)
        rows.append(row)
        print(f"{candidate.name}: validation pairwise={row['metrics']['pairwise_accuracy']:.5f}", flush=True)
    top_core = sorted(rows, key=lambda r: (-r["metrics"]["pairwise_accuracy"], r["metrics"]["rmse"]))[:3]
    # Recency is fixed across core configurations. Text residuals use the same
    # personal mean, so representations can be fitted once across all blends.
    best_core = candidate_from_dict(top_core[0]["candidate"])
    tfidf_rows = []

    def search_text(variants, family_rows):
        for text_config in variants:
            t0 = time.perf_counter()
            evidence = text_evidence(cases, replace(best_core, text=text_config), "validation")
            fit_seconds = time.perf_counter() - t0
            for core_row in top_core:
                core = candidate_from_dict(core_row["candidate"])
                for prior in (1., 5.):
                    for weight in (.25, .5, 1., 2.):
                        config = replace(text_config, regularization_strength=prior, weight=weight)
                        name = (f"{core.name}+{config.representation}-n{config.ngram_max}"
                                f"-df{config.min_df}-d{config.latent_dimensions}-r{prior:g}-w{weight:g}")
                        candidate = replace(core, name=name, text=config)
                        scores = blend(core_cache[core.name], evidence, config)
                        row = row_for(candidate, cases, scores, fit_seconds)
                        rows.append(row)
                        family_rows.append(row)
            leader = select_best(family_rows)
            print(f"{text_config.representation} n={text_config.ngram_max} df={text_config.min_df} "
                  f"dim={text_config.latent_dimensions}: best validation="
                  f"{leader['metrics']['pairwise_accuracy']:.5f}", flush=True)

    search_text([TextConfig(ngram_max=n, min_df=df) for n in (1, 2) for df in (1, 2)], tfidf_rows)
    tfidf_best = select_best(tfidf_rows)
    lsa_rows = []
    if tfidf_best["metrics"]["pairwise_accuracy"] > top_core[0]["metrics"]["pairwise_accuracy"]:
        best_text = candidate_from_dict(tfidf_best["candidate"]).text
        search_text([replace(best_text, representation="lsa", latent_dimensions=d)
                     for d in (8, 32)], lsa_rows)

    winner = select_best(rows)
    selected = candidate_from_dict(winner["candidate"])
    selection = {
        "schema_version": 1, "objective": "macro validation pairwise accuracy; RMSE tie-break",
        "candidate": selected.to_dict(), "validation_metrics": winner["metrics"],
    }
    # The choice is persisted BEFORE any final test labels are scored.
    (output / "selected_config.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Locked selection: {selected.name}. Evaluating final test holdout.", flush=True)
    leaders = {"baseline": Candidate("baseline"), "tuned_core": best_core,
               "best_tfidf": candidate_from_dict(tfidf_best["candidate"]), "selected": selected}
    if lsa_rows:
        leaders["best_lsa"] = candidate_from_dict(select_best(lsa_rows)["candidate"])
    test, test_users = {}, {}
    for name, candidate in leaders.items():
        t0 = time.perf_counter()
        scores = core_scores(cases, candidate, "test")
        if candidate.text:
            scores = blend(scores, text_evidence(cases, candidate, "test"), candidate.text)
        else:
            scores = [np.clip(a, 0, 5) for a in scores]
        metrics, per_user = evaluate_scores(cases, scores, "test")
        test[name] = {"candidate": candidate.to_dict(), "metrics": metrics,
                      "seconds": time.perf_counter() - t0}
        test_users[name] = per_user
        print(f"{name}: test pairwise={metrics['pairwise_accuracy']:.5f}, "
              f"NDCG@10={metrics['ndcg_at_10']:.5f}, RMSE={metrics['rmse']:.5f}", flush=True)

    # Verify the vectorized search actually matches the public serving API.
    parity_max_error = 0.
    for case in cases:
        model = ContentModel(case.history, case.movies, selected.scoring, selected.profile, selected.text)
        predictions = model.predict((case.user_id,), tuple(r.movie_id for r in case.test))
        expected = core_scores((case,), selected, "test")
        expected = (blend(expected, text_evidence((case,), selected, "test"), selected.text)
                    if selected.text else [np.clip(a, 0, 5) for a in expected])
        parity_max_error = max(parity_max_error,
                               float(np.max(np.abs([r.predicted_score for r in predictions] - expected[0]))))
    if parity_max_error > 1e-10:
        raise AssertionError(f"Evaluation/serving mismatch: {parity_max_error}")

    slices = {}
    for name, predicate in (("history_5_to_19", lambda c: len(c.history) < 20),
                            ("history_20_to_99", lambda c: 20 <= len(c.history) < 100),
                            ("history_100_plus", lambda c: len(c.history) >= 100)):
        indices = [i for i, c in enumerate(cases) if predicate(c)]
        if indices:
            slices[name] = {label: summarize([test_users[label][i] for i in indices])
                            for label in ("baseline", "selected")}

    coverage = {}
    for phase in ("history", "validation", "test"):
        text_count, count = 0, 0
        for case in cases:
            catalog = {m.movie_id: m for m in case.movies}
            phase_ratings = getattr(case, phase)
            count += len(phase_ratings)
            text_count += sum(bool(catalog[r.movie_id].keywords or catalog[r.movie_id].plot) for r in phase_ratings)
        coverage[phase] = {"with_text": text_count, "ratings": count, "fraction": text_count / count}

    runtime = benchmark_serving(cases, movies, tags, selected)
    report = {
        "protocol": {
            "split": "per-user chronological 60/20/20; timestamp groups kept together",
            "profile": "first 60% frozen for both validation and test; no holdout refit",
            "text": "only tags at/before training cutoff; evaluation user's tags excluded; vocabulary/IDF/SVD fit on history movies only",
            "selection": selection["objective"], "candidate_count": len(rows),
            "embeddings": "LSA fit on training TF-IDF; not a pretrained semantic model" if lsa_rows else "skipped: TF-IDF did not improve validation over tuned core",
            "metrics": "user-macro pairwise and NDCG@10 (exponential gains, averaged score ties); RMSE=sqrt(mean per-user MSE); MAE=user-macro",
            "candidate_set": "rated held-out movies only; not full-catalog retrieval evaluation",
            "skipped_users": skipped, "seed": 42,
            "metadata_temporality": metadata["temporality"],
        },
        "data_sha256": {f: hashlib.sha256((directory / f).read_bytes()).hexdigest()
                        for f in ("movies.csv", "ratings.csv", "tags.csv")},
        "movie_metadata": {"file": str(movie_path), "sha256": hashlib.sha256(movie_path.read_bytes()).hexdigest(),
                           **metadata},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                        "platform": platform.platform(), "blas_threads": 1},
        "coverage": coverage, "validation": rows, "test": test,
        "selected_vs_baseline": paired_bootstrap(test_users["baseline"], test_users["selected"]),
        "test_slices": slices, "serving": runtime, "serving_parity_max_error": parity_max_error,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "validation.csv").open("w", newline="") as destination:
        fields = ["name", "pairwise_accuracy", "ndcg_at_10", "rmse", "mae", "seconds"]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: -r["metrics"]["pairwise_accuracy"]):
            writer.writerow({"name": row["candidate"]["name"], "seconds": row["seconds"],
                             **{key: row["metrics"][key] for key in fields[1:-1]}})
    with (output / "test_users.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["user_id", "history_count", "model",
                                                       *test_users["baseline"][0].keys()])
        writer.writeheader()
        for label, users in test_users.items():
            for case, metrics in zip(cases, users, strict=True):
                writer.writerow({"user_id": case.user_id, "history_count": len(case.history),
                                 "model": label, **metrics})
    print(f"Saved report to {output}; elapsed {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def benchmark_serving(cases, movies, tags, selected):
    measurements = []
    # Deterministic short, median, and long histories; rank the entire catalog.
    ordered = sorted(cases, key=lambda c: (len(c.history), c.user_id))
    for case in (ordered[0], ordered[len(ordered) // 2], ordered[-1]):
        catalog = with_keywords_as_of(movies, tags, max(r.timestamp for r in case.history), case.user_id)
        for label, config in (("baseline", Candidate("baseline")), ("selected", selected)):
            t0 = time.perf_counter()
            model = ContentModel(case.history, catalog, config.scoring, config.profile, config.text)
            model.recommend(case.user_id)
            cold = time.perf_counter() - t0
            warm = []
            for _ in range(3):
                t0 = time.perf_counter()
                model.recommend(case.user_id)
                warm.append(time.perf_counter() - t0)
            measurements.append({"user_id": case.user_id, "history_count": len(case.history),
                                 "catalog_size": len(catalog), "model": label,
                                 "cold_ms": cold * 1000, "warm_median_ms": float(np.median(warm)) * 1000})
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", help="Default: reports/week6, or reports/tmdb-experiment with --movies-file")
    parser.add_argument("--movies-file", help="Optional enriched CSV; metadata dates and hash are recorded")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir, args.output_dir or ("reports/tmdb-experiment" if args.movies_file else "reports/week6"),
            args.movies_file)


if __name__ == "__main__":
    main()
