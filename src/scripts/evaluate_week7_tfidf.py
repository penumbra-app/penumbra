
"""Matched Week 7 Hybrid comparison: original Content vs selected TF-IDF.

Split: 60% profile, 10% reranker training, 10% validation, 20% final test.
The final test has already been accessed; do not use it for further selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.hybrid.ml_reranker import chronological_split

from src.collaborative.matrix_factorization import (
    BiasedMatrixFactorization,
)
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.splits import (
    build_user_evaluation_split,
)
from src.hybrid.content_adapter import (
    build_content_model_from_frames,
)
from src.hybrid.genre_recommender import (
    score_movies_by_genre,
)
from src.hybrid.ml_reranker import (
    AblationCandidate,
    calculate_movie_popularity,
    load_ranker,
)
from src.load_data import load_movielens
from src.scripts.week6_ablations import (
    load_content_configuration,
)
from src.scripts.train_week7_tfidf import (
    FEATURES,
    DEFAULT_MOVIES,
    DEFAULT_PRESET,
    DEFAULT_REPORT,
    DEFAULT_OUTPUT,
)


def evaluate(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    rankers: dict,
    *,
    scoring_config,
    profile_config,
    selected_config_path: Path,
    partition: str,
    content_variants: dict[str, str] | None = None,
    include_user_metrics: bool = False,
) -> dict:
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be validation or test")
    variants = content_variants or {name: name for name in rankers}
    if set(variants) != set(rankers) or not set(variants.values()) <= {"baseline", "tfidf"}:
        raise ValueError("Each ranker needs a baseline or tfidf content variant")
    profiles = []
    heldout = {}

    for raw_id in ratings["userId"].unique():
        uid = int(raw_id)

        rows = ratings.loc[
            ratings["userId"] == uid
        ].copy()

        if len(rows) < 10:
            continue

        profile, train, remaining = chronological_split(
            rows,
            profile_fraction=0.60,
            train_fraction=0.10,
        )

        validation_end = int(len(rows) * 0.80) - int(len(rows) * 0.70)

        validation = remaining.iloc[:validation_end].copy()
        test = remaining.iloc[validation_end:].copy()

        profiles.append(profile)

        candidate_slice = (
            validation
            if partition == "validation"
            else test
        )

        if len(candidate_slice) >= 2:
            heldout[uid] = (
                profile,
                candidate_slice,
            )

    if not profiles:
        raise ValueError(
            "No eligible profiles"
        )

    global_profile = pd.concat(
        profiles,
        ignore_index=True,
    )

    mf = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=40,
        prior_strength=5.0,
        random_state=42,
        shrink_latent=True,
    )
    mf.fit(global_profile)

    agg = {
        name: {
            "macro": [],
            "ndcg": [],
            "credit": 0.0,
            "pairs": 0,
            "users": [],
            "user_metrics": [],
        }
        for name in rankers
    }

    for uid, (
        profile,
        candidate_slice,
    ) in heldout.items():

        movie_ids = (
            candidate_slice["movieId"]
            .astype(int)
            .tolist()
        )

        mf_scores = mf.predict(
            user_ids=[uid] * len(movie_ids),
            movie_ids=movie_ids,
        ).set_index("movie_id")[
            "predicted_score"
        ].to_dict()

        references = global_profile.loc[
            global_profile["userId"] != uid
        ].copy()

        pop = calculate_movie_popularity(
            references
        )

        scored = score_movies_by_genre(
            user_id=uid,
            user_history=profile,
            reference_ratings=references,
            movies=movies,
            movie_ids=movie_ids,
        )

        personal = {
            s.movie_id: s.personal_score
            for s in scored
        }
        quality = {
            s.movie_id: s.quality_score
            for s in scored
        }

        models = {
            "baseline": (
                build_content_model_from_frames(
                    profile_ratings=profile,
                    movies=movies,
                    scoring_config=scoring_config,
                    profile_config=profile_config,
                )
            ),
            "tfidf": (
                build_content_model_from_frames(
                    profile_ratings=profile,
                    movies=movies,
                    use_selected_content=True,
                    selected_config_path=(
                        selected_config_path
                    ),
                )
            ),
        }

        actual_map = dict(
            zip(
                candidate_slice[
                    "movieId"
                ].astype(int),
                candidate_slice[
                    "rating"
                ].astype(float),
            )
        )

        eligible = [
            mid
            for mid in movie_ids
            if mid in mf_scores
            and mid in pop
            and mid in personal
            and mid in quality
            and mid in actual_map
        ]

        if len(eligible) < 2:
            continue

        actual = np.asarray(
            [
                actual_map[mid]
                for mid in eligible
            ],
            dtype=float,
        )

        direction = np.sign(
            actual[:, None]
            - actual[None, :]
        )
        mask = np.triu(
            direction != 0,
            k=1,
        )
        pairs = int(mask.sum())

        if pairs == 0:
            continue

        for name, ranker in rankers.items():
            content = models[variants[name]]
            feature_rows = []

            for mid in eligible:
                cp = content.predict_one(
                    user_id=uid,
                    movie_id=mid,
                )

                candidate = AblationCandidate(
                    movie_id=mid,
                    collaborative_score=float(
                        mf_scores[mid]
                    ),
                    personal_score=float(
                        personal[mid]
                    ),
                    content_score=float(
                        cp.predicted_score
                    ),
                    quality_score=float(
                        quality[mid]
                    ),
                    popularity=float(
                        pop[mid]
                    ),
                )

                feature_rows.append(
                    candidate.as_array(
                        FEATURES
                    )
                )

            X = np.vstack(
                feature_rows
            )

            predicted = np.asarray(
                ranker.model.decision_function(
                    ranker.scaler.transform(X)
                ),
                dtype=float,
            )

            predicted_direction = np.sign(
                predicted[:, None]
                - predicted[None, :]
            )

            credit = float(
                np.sum(
                    np.where(
                        predicted_direction[mask]
                        == 0,
                        0.5,
                        (
                            predicted_direction[mask]
                            == direction[mask]
                        ).astype(float),
                    )
                )
            )

            a = agg[name]

            a["macro"].append(
                credit / pairs
            )
            a["credit"] += credit
            a["pairs"] += pairs

            a["ndcg"].append(
                float(
                    ndcg_at_k(
                        actual.tolist(),
                        predicted.tolist(),
                        k=10,
                    )
                )
            )

            a["users"].append(uid)
            a["user_metrics"].append({
                "user_id": uid, "pairwise_accuracy": credit / pairs,
                "credit": credit, "pairs": pairs, "ndcg_at_10": a["ndcg"][-1],
            })

    result = {}

    for name, values in agg.items():
        if not values["pairs"]:
            raise ValueError(
                "No eligible evaluation pairs "
                f"for {name}"
            )

        result[name] = {
            "pairwise_macro": float(
                np.mean(
                    values["macro"]
                )
            ),
            "pairwise_weighted": float(
                values["credit"]
                / values["pairs"]
            ),
            "ndcg_at_10": float(
                np.mean(
                    values["ndcg"]
                )
            ),
            "pairs": int(
                values["pairs"]
            ),
            "users": len(
                values["macro"]
            ),
            "user_ids": values["users"],
        }
        if include_user_metrics:
            result[name]["user_metrics"] = values["user_metrics"]

    if any(entry["user_ids"] != next(iter(result.values()))["user_ids"] for entry in result.values()):
        raise AssertionError(
            "Evaluation users differ"
        )

    for entry in result.values():
        del entry["user_ids"]

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--movies",
        type=Path,
        default=DEFAULT_MOVIES,
    )
    parser.add_argument(
        "--selected-content-config",
        type=Path,
        default=DEFAULT_PRESET,
    )
    parser.add_argument(
        "--content-report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--partition",
        choices=[
            "validation",
            "test",
        ],
        default="validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/week7/tfidf_comparison.json"
        ),
    )

    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing evaluation report: {args.output}")

    if args.partition == "validation":
        print("Evaluating on held-out 70–80% validation ratings.")
    else:
        print("FINAL TEST: Run only after model selection is frozen.")

    ratings, movies = load_movielens(
        "data",
        movies_path=args.movies,
    )

    if not {"plot", "keywords"}.issubset(
        movies.columns
    ):
        parser.error(
            "The provided movie catalog "
            "lacks plot/keywords columns"
        )

    scoring, profile, _ = (
        load_content_configuration(
            args.content_report
        )
    )

    rankers = {}

    for name in (
        "baseline",
        "tfidf",
    ):
        file = (
            args.models_dir
            / f"{name}_plus_personal.joblib"
        )

        if not file.is_file():
            parser.error(
                f"Missing model {file}; "
                "run train_week7_tfidf first"
            )

        ranker = load_ranker(
            path=str(file)
        )

        if (
            int(
                ranker.scaler.n_features_in_
            )
            != len(FEATURES)
        ):
            raise ValueError(
                f"{name}: wrong feature count"
            )

        rankers[name] = ranker

    results = evaluate(
        ratings,
        movies,
        rankers,
        scoring_config=scoring,
        profile_config=profile,
        selected_config_path=(
            args.selected_content_config
        ),
        partition=args.partition,
    )

    report = {
        "partition": args.partition,
        "movies": str(args.movies),
        "models_dir": str(
            args.models_dir
        ),
        "comparability": (
            "same catalog, ratings, "
            "chronological split, MF, "
            "candidate set"
        ),
        "selection_warning": (
            "The 70–80% partition is held-out validation. "
            "The final 80–100% partition must remain untouched "
            "until model selection is frozen."
        ),
        "results": results,
        "deltas_tfidf_minus_baseline": {
            k: (
                results["tfidf"][k]
                - results["baseline"][k]
            )
            for k in (
                "pairwise_macro",
                "pairwise_weighted",
                "ndcg_at_10",
            )
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "results": results,
                "deltas": report[
                    "deltas_tfidf_minus_baseline"
                ],
            },
            indent=2,
        )
    )

    print(
        f"Saved {args.output}"
    )


if __name__ == "__main__":
    main()
