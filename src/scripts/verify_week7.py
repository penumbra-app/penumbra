"""Validation-only balance comparison and real serving smoke checks.

No final-test evaluation, no model promotion, no overwriting existing reports.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.hybrid.ml_reranker import load_ranker
from src.hybrid.recommender import recommend_for_user
from src.hybrid.week7 import FEATURES, build_feature_rows, load_bundle, validate_ranker
from src.hybrid.content_adapter import build_content_model_from_frames
from src.load_data import load_movielens
from src.scripts.evaluate_week7_tfidf import evaluate
from src.scripts.train_week7_tfidf import DEFAULT_MOVIES, DEFAULT_OUTPUT, DEFAULT_PRESET, DEFAULT_REPORT
from src.scripts.week6_ablations import load_content_configuration


def paired_bootstrap(first, second, *, seed=42, repetitions=2000):
    if [r["user_id"] for r in first] != [r["user_id"] for r in second]:
        raise ValueError("Paired bootstrap needs the same ordered users")
    a = np.array([[r["pairwise_accuracy"], r["ndcg_at_10"], r["credit"], r["pairs"]] for r in first])
    b = np.array([[r["pairwise_accuracy"], r["ndcg_at_10"], r["credit"], r["pairs"]] for r in second])
    if not len(a):
        raise ValueError("No users to bootstrap")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(a), len(a))
        aa, bb = a[indices], b[indices]
        deltas.append([(bb[:, 0] - aa[:, 0]).mean(),
                       bb[:, 2].sum() / bb[:, 3].sum() - aa[:, 2].sum() / aa[:, 3].sum(),
                       (bb[:, 1] - aa[:, 1]).mean()])
    return {"seed": seed, "resamples": repetitions, "unit": "user", "percentile_95_intervals": {
        key: np.quantile(np.array(deltas)[:, i], [.025, .975]).tolist()
        for i, key in enumerate(("pairwise_macro", "pairwise_weighted", "ndcg_at_10"))}}


def smoke():
    bundle = load_bundle()
    ratings, movies = load_movielens("data", movies_path=DEFAULT_MOVIES)
    counts = ratings.groupby("userId").size()
    existing = int(counts.idxmax())
    sparse = int(counts.idxmin())
    cold = int(counts.index.max()) + 1
    checks = []
    for label, uid, frame in (
        ("existing", existing, ratings),
        ("sparse_one_rating", sparse, ratings.loc[(ratings.userId != sparse) | (ratings.index == ratings.loc[ratings.userId == sparse].index[0])]),
        ("cold", cold, ratings),
    ):
        start = perf_counter()
        result = recommend_for_user(uid, frame, movies, 10, week7_bundle=bundle)
        elapsed = perf_counter() - start
        repeat = recommend_for_user(uid, frame, movies, 10, week7_bundle=bundle)
        watched = set(frame.loc[frame.userId == uid, "movieId"])
        ids = [r.movie_id for r in result]
        assert len(ids) == 10 and len(set(ids)) == 10 and not watched.intersection(ids)
        assert result == repeat and all(np.isfinite(r.score) for r in result)
        checks.append({"case": label, "top10": ids, "seconds": elapsed, "deterministic": True})
    history = ratings.loc[ratings.userId == existing]
    content = build_content_model_from_frames(history, bundle["movies"], use_selected_content=True,
                                             selected_config_path=DEFAULT_PRESET)
    ids, X = build_feature_rows(existing, history, bundle["reference_ratings"], bundle["movies"],
                               [int(movies.movieId.iloc[-1])], content, None)
    assert X.shape == (1, len(FEATURES)) and X[0, 0] == 2.5
    checks.append({"case": "missing_mf", "neutral_score": float(X[0, 0]), "finite_features": True})
    text_profile = content.build_text_profile(existing)
    assert content.text_config.representation == "tfidf"
    assert text_profile.vectorizer is not None
    checks.append({"case": "tfidf_preset", "representation": content.text_config.representation,
                   "vocabulary_terms": len(text_profile.vectorizer.vocabulary_),
                   "config_sha256": bundle["selected_config_sha256"]})
    return checks


def inspect_artifacts(balanced_model):
    bundle = load_bundle()
    loaded = {}
    for name, path in (
        ("baseline", DEFAULT_OUTPUT / "baseline_plus_personal.joblib"),
        ("tfidf", DEFAULT_OUTPUT / "tfidf_plus_personal.joblib"),
        ("balanced", balanced_model),
    ):
        ranker = load_ranker(str(path))
        validate_ranker(ranker, FEATURES)
        loaded[name] = {"features": int(ranker.scaler.n_features_in_), "finite_parameters": True}
    legacy = load_ranker()
    assert legacy.scaler.n_features_in_ == legacy.model.n_features_in_ == 3
    loaded["legacy"] = {"features": 3}
    uid = int(bundle["reference_ratings"].groupby("userId").size().idxmax())
    history = bundle["reference_ratings"].loc[bundle["reference_ratings"].userId == uid]
    content = build_content_model_from_frames(history, bundle["movies"], use_selected_content=True,
                                             selected_config_path=DEFAULT_PRESET)
    profile = content.build_text_profile(uid)
    assert content.text_config.representation == "tfidf" and profile.vectorizer is not None
    return {"rankers": loaded, "tfidf_vocabulary_terms": len(profile.vectorizer.vocabulary_),
            "profile_rows": len(bundle["reference_ratings"]), "mf_fitted": True,
            "preset_sha256": bundle["selected_config_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-model", type=Path,
                        default=Path("src/models/week7_balanced_validation/tfidf_plus_personal.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/week7/finalization_validation.json"))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--artifacts-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    report = {"partition": "validation", "final_test_rerun": False}
    if args.artifacts_only:
        report["artifacts"] = inspect_artifacts(args.balanced_model)
    else:
        report["smoke"] = smoke()
    if not args.smoke_only and not args.artifacts_only:
        ratings, movies = load_movielens("data", movies_path=DEFAULT_MOVIES)
        scoring, profile, _ = load_content_configuration(DEFAULT_REPORT)
        rankers = {"unbalanced": load_ranker(str(DEFAULT_OUTPUT / "tfidf_plus_personal.joblib")),
                   "balanced": load_ranker(str(args.balanced_model))}
        start = perf_counter()
        results = evaluate(ratings, movies, rankers, scoring_config=scoring, profile_config=profile,
                           selected_config_path=DEFAULT_PRESET, partition="validation",
                           content_variants={name: "tfidf" for name in rankers}, include_user_metrics=True)
        report["evaluation_seconds"] = perf_counter() - start
        report["paired_user_bootstrap"] = paired_bootstrap(
            results["unbalanced"]["user_metrics"], results["balanced"]["user_metrics"])
        report["results"] = results
        report["balanced_training"] = json.loads(args.balanced_model.with_suffix(".json").read_text(encoding="utf-8"))
        report["deltas_balanced_minus_unbalanced"] = {key: results["balanced"][key] - results["unbalanced"][key]
            for key in ("pairwise_macro", "pairwise_weighted", "ndcg_at_10")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    if "results" in report:
        print(json.dumps({name: {k: v for k, v in r.items() if k != "user_metrics"}
                          for name, r in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
