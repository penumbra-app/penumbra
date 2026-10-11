"""Held-out content evaluation using the team's existing 60/20/20 split."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from src.content import ContentModel, ProfileConfig, ScoringConfig, predict_batch
from src.content.features import FEATURE_TYPES, feature_values
from src.content.schemas import MovieMetadata
from src.data_processing.movielens import movie_metadata_from_records, user_ratings_from_records
from src.evaluation.splits import build_user_evaluation_split
from src.evaluation.metrics import ranking_metrics

KINDS = ("genre",) + FEATURE_TYPES


def ablation_configs() -> dict[str, ScoringConfig]:
    """Prespecified candidates; change one feature at a time from full."""
    full = ScoringConfig(cast_weight=0.25)
    configs = {
        "user_mean": replace(full, **{f"{kind}_weight": 0 for kind in KINDS}),
        "genre_only": replace(full, **{f"{kind}_weight": 0 for kind in FEATURE_TYPES}),
        "full": full,
    }
    configs.update({f"without_{kind}": replace(full, **{f"{kind}_weight": 0})
                    for kind in KINDS})
    return configs




def evaluate_content(ratings: pd.DataFrame, movie_records: pd.DataFrame,
                     profile_config: ProfileConfig | None = None) -> tuple[dict, pd.DataFrame]:
    """Fit only earliest 60%; choose by validation RMSE before reading test scores.

    Returns a JSON-ready report and reusable row-level split assignments.
    Missing catalog rows receive empty metadata, preserving evaluation coverage.
    """
    required = {"userId", "movieId", "rating", "timestamp"}
    if not required <= set(ratings.columns) or ratings.empty:
        raise ValueError("nonempty ratings with userId, movieId, rating, timestamp are required")
    if ratings[list(required)].isna().any().any():
        raise ValueError("evaluation requires complete ratings and timestamps")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("resolve duplicate user/movie ratings before splitting")
    user_ratings_from_records(ratings.to_dict("records"))  # Validate before slicing.
    movies = movie_metadata_from_records(movie_records.to_dict("records"))
    movies_by_id = {m.movie_id: m for m in movies}
    if len(movies_by_id) != len(movies):
        raise ValueError("duplicate movie IDs")
    for movie_id in ratings.movieId.unique():
        movies_by_id.setdefault(int(movie_id), MovieMetadata(int(movie_id), f"Movie {movie_id}"))
    active_profile = profile_config or ProfileConfig()
    configs = ablation_configs()
    assignments, training = [], []
    held_out = {"validation": {}, "test": {}}
    for user_id, rows in ratings.groupby("userId", sort=True):
        train, validation, test = build_user_evaluation_split(rows)
        training.extend(user_ratings_from_records(train.to_dict("records")))
        for label, frame in (("train", train), ("validation", validation), ("test", test)):
            assignments.append(frame.assign(split=label))
            if label != "train" and not frame.empty:
                held_out[label][int(user_id)] = frame
    splits = pd.concat(assignments, ignore_index=True)[
        ["userId", "movieId", "rating", "timestamp", "split"]]
    if not held_out["validation"] or not held_out["test"]:
        raise ValueError("not enough data for nonempty validation and test partitions")
    model = ContentModel(training, movies_by_id.values(), profile_config=active_profile)
    coverage = {kind: sum(bool(m.genres) if kind == "genre" else bool(
        feature_values(m, kind, active_profile.max_cast_members)) for m in movies)
        for kind in KINDS}
    # A feature with no training evidence cannot be promoted by an identical score.
    learned_kinds = set()
    for user_id in held_out["validation"]:
        profile = model.build_profile(user_id)
        if profile.genre_preferences:
            learned_kinds.add("genre")
        learned_kinds.update(p.feature_type for p in profile.feature_preferences)
    summaries = {}
    selected = None
    for partition in ("validation", "test"):
        totals = {name: dict(errors=[], squared=[], pairs=0, credit=0.0,
                            pairwise=[], ndcg=[], users=0, cold_users=0)
                  for name in configs}
        for user_id, frame in held_out[partition].items():
            profile = model.build_profile(user_id)
            candidates = tuple(movies_by_id[int(mid)] for mid in frame.movieId)
            actual = frame.rating.to_numpy(dtype=float)
            for name, config in configs.items():
                predicted = np.array([p.predicted_score for p in
                    predict_batch(profile, candidates, config)])
                metrics = ranking_metrics(actual, predicted)
                total = totals[name]
                total["errors"].extend(np.abs(predicted - actual).tolist())
                total["squared"].extend(((predicted - actual) ** 2).tolist())
                total["pairs"] += metrics["pairs"]
                total["credit"] += metrics["pairwise_credit"]
                total["users"] += 1
                total["cold_users"] += profile.rating_count == 0
                if metrics["pairwise_accuracy"] is not None:
                    total["pairwise"].append(metrics["pairwise_accuracy"])
                if metrics["ndcg_at_10"] is not None:
                    total["ndcg"].append(metrics["ndcg_at_10"])
        summaries[partition] = {name: {
            "mae": float(np.mean(t["errors"])),
            "rmse": float(np.sqrt(np.mean(t["squared"]))),
            "pairwise_accuracy_macro": float(np.mean(t["pairwise"])) if t["pairwise"] else None,
            "pairwise_accuracy_micro": t["credit"] / t["pairs"] if t["pairs"] else None,
            "ndcg_at_10": float(np.mean(t["ndcg"])) if t["ndcg"] else None,
            "ratings": len(t["errors"]), "users": t["users"], "pairs": t["pairs"],
            "ndcg_users": len(t["ndcg"]), "pairwise_users": len(t["pairwise"]),
            "cold_users": t["cold_users"],
        } for name, t in totals.items()}
        if partition == "validation":
            selected = min(configs, key=lambda name: (
                summaries[partition][name]["rmse"],
                sum(getattr(configs[name], f"{kind}_weight") > 0 for kind in KINDS), name))
    selected_config = replace(configs[selected], **{
        f"{kind}_weight": 0 for kind in KINDS if kind not in learned_kinds})
    report = {
        "protocol": "shared per-user chronological 60/20/20; fit 60%; no test refit",
        "selection_metric": "validation RMSE; exact ties prefer fewer enabled features",
        "ranking_candidates": "held-out rated movies only; liked means rating >= 4",
        "split_sha256": hashlib.sha256(splits.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "split_counts": {k: int(v) for k, v in splits.split.value_counts().items()},
        "catalog_movies": len(movies), "metadata_coverage": coverage,
        "unvalidated_features": sorted(set(KINDS) - learned_kinds),
        "profile_config": asdict(active_profile),
        "configs": {name: asdict(config) for name, config in configs.items()},
        "selected_variant": selected, "selected_config": asdict(selected_config),
        "results": summaries,
    }
    return report, splits
