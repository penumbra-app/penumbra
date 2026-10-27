"""Shared-split rating/ranking benchmark and frozen-weight ablations.

Run: python -m src.collaborative.evaluation.benchmark
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform

import joblib
import sklearn
import numpy as np
import pandas as pd

from src.collaborative import BiasedMatrixFactorization, MovieAverageBaseline
from src.collaborative.matrix_factorization import checked_ids
from src.evaluation.metrics import ranking_metrics
from src.evaluation.splits import build_user_evaluation_split

ROOT = Path(__file__).resolve().parents[1]


def shared_split(ratings):
    required = ["userId", "movieId", "rating", "timestamp"]
    if not set(required) <= set(ratings) or ratings.empty:
        raise ValueError("Nonempty ratings with userId, movieId, rating, timestamp required")
    if ratings[required].isna().any().any():
        raise ValueError("Complete ratings and timestamps required")
    checked_ids(ratings.userId, "userId")
    checked_ids(ratings.movieId, "movieId")
    for column in ("rating", "timestamp"):
        if not pd.api.types.is_numeric_dtype(ratings[column]) or not np.isfinite(ratings[column]).all():
            raise ValueError(f"{column} must be finite numeric values")
    if not ratings.rating.between(.5, 5).all():
        raise ValueError("Ratings must be in [0.5, 5]")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Resolve duplicate user/movie ratings before splitting")
    parts = []
    for _, history in ratings.groupby("userId", sort=True):
        # Exactly the content evaluator's helper and eligibility policy, including
        # small histories and the existing timestamp tie behavior.
        for label, frame in zip(("train", "validation", "test"), build_user_evaluation_split(history)):
            parts.append(frame.assign(split=label))
    assignments = pd.concat(parts, ignore_index=True)[required + ["split"]]
    if any(not (assignments.split == label).any() for label in ("train", "validation", "test")):
        raise ValueError("Nonempty train, validation and test partitions required")
    return assignments


def ablation_models(model):
    """Remove one inference contribution at a time; do not retrain or mutate full."""
    models = {"full": model}
    for name in ("without_shrinkage", "without_latent", "without_user_bias", "without_movie_bias"):
        variant = copy.deepcopy(model)
        if name == "without_shrinkage":
            variant.shrink_latent = False
        elif name == "without_latent":
            variant.weights_.user_factors.fill(0)
        elif name == "without_user_bias":
            variant.weights_.user_biases.fill(0)
        else:
            variant.weights_.movie_biases.fill(0)
        models[name] = variant
    return models


def summarize_predictions(rows, scores):
    errors = np.asarray(scores) - rows.rating.to_numpy(dtype=float)
    details = []
    frame = rows.assign(predicted_score=np.asarray(scores))
    for uid, group in frame.groupby("userId", sort=True):
        details.append({"user_id": int(uid), "ratings": len(group),
                        **ranking_metrics(group.rating, group.predicted_score)})
    pairwise = [r["pairwise_accuracy"] for r in details if r["pairwise_accuracy"] is not None]
    ndcg = [r["ndcg_at_10"] for r in details if r["ndcg_at_10"] is not None]
    pairs = sum(r["pairs"] for r in details)
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt((errors ** 2).mean())),
            "pairwise_accuracy_macro": float(np.mean(pairwise)) if pairwise else None,
            "pairwise_accuracy_micro": sum(r["pairwise_credit"] for r in details) / pairs if pairs else None,
            "ndcg_at_10": float(np.mean(ndcg)) if ndcg else None,
            "ratings": len(rows), "users": len(details), "pairs": pairs,
            "pairwise_users": len(pairwise), "ndcg_users": len(ndcg)}, details


def evaluate_benchmark(ratings, settings=None):
    assignments = shared_split(ratings)
    train = assignments.loc[assignments.split == "train"]
    # Prespecified Week 3 selected settings. Never load the 80%-trained artifact:
    # that would expose this benchmark's validation labels to the model.
    settings = dict(n_epochs=40) if settings is None else dict(settings)
    model = BiasedMatrixFactorization(**settings).fit(train)
    models = {"baseline": MovieAverageBaseline().fit(train), **ablation_models(model)}
    results, user_results = {}, []
    selected = None
    for partition in ("validation", "test"):
        rows = assignments.loc[assignments.split == partition]
        results[partition] = {}
        for name, variant in models.items():
            scores = variant.predict(rows.userId.tolist(), rows.movieId.tolist()).predicted_score
            summary, details = summarize_predictions(rows, scores)
            results[partition][name] = summary
            user_results.extend(dict(split=partition, model=name, **r) for r in details)
        if partition == "validation":
            selected = min(results[partition], key=lambda name: (results[partition][name]["rmse"], name))
    report = {
        "protocol": "shared per-user chronological 60/20/20; fit 60%; no test refit",
        "ranking_candidates": "held-out rated movies only; liked means rating >= 4",
        "ranking_policy": "binary NDCG@10, expected gain for tied scores; macro average over users with likes; pairwise excludes equal ratings and gives score ties half credit",
        "ablation_policy": "one contribution removed at inference using identical fitted weights; not separately retrained models",
        "selection_metric": "validation RMSE, alphabetical exact tiebreak; diagnostic selection only, no artifact promotion",
        "selected_variant": selected,
        "settings": model.get_params(),
        "split_counts": {k: int(v) for k, v in assignments.split.value_counts().items()},
        "split_sha256": hashlib.sha256(assignments.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "baseline_settings": {"prior_strength": 10.0},
        "caveat": "Development benchmark: holdout and 40-epoch configuration were examined in prior work. No statistical significance or full-catalog retrieval claim.",
        "results": results,
    }
    return report, assignments, pd.DataFrame(user_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=ROOT.parents[1] / "data/ratings.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmark")
    args = parser.parse_args()
    print("Fitting 40-epoch MF on the shared 60% training partition...", flush=True)
    report, splits, users = evaluate_benchmark(pd.read_csv(args.ratings))
    report["input_sha256"] = hashlib.sha256(args.ratings.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    splits.to_csv(args.output / "splits.csv", index=False)
    users.to_csv(args.output / "per_user_metrics.csv", index=False)
    for partition, results in report["results"].items():
        print(partition, flush=True)
        print(pd.DataFrame(results).T[["rmse", "mae", "pairwise_accuracy_macro", "ndcg_at_10"]].to_string())
    print("Validation-selected diagnostic variant:", report["selected_variant"])


if __name__ == "__main__":
    main()
