import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.collaborative.matrix_factorization import (
    BiasedMatrixFactorization,
)
from src.content.profiles import ProfileConfig
from src.content.scoring import ScoringConfig
from src.evaluation.metrics import (
    comparison_credit,
    ndcg_at_k,
)
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
from src.scripts.week6_ablations import EXPERIMENTS


OLD_ML_MACRO = 0.63876
OLD_ML_WEIGHTED = 0.68516
OLD_ML_NDCG = 0.9051

IMDB_BASELINE_MACRO = 0.63498
IMDB_BASELINE_WEIGHTED = 0.68267
IMDB_BASELINE_NDCG = 0.9069


def load_content_configuration(
    report_path: Path,
) -> tuple[ScoringConfig, ProfileConfig, str]:
    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    return (
        ScoringConfig(**report["selected_config"]),
        ProfileConfig(**report["profile_config"]),
        str(report["selected_variant"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--movies",
        type=Path,
        default=Path("data/movies-enriched.csv"),
    )

    parser.add_argument(
        "--content-report",
        type=Path,
        default=Path("docs/week6-content-repro.json"),
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("src/models/week6_ablations"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/week6-ablation-results.json"),
    )

    args = parser.parse_args()

    ratings, movies = load_movielens(
        "data",
        movies_path=args.movies,
    )

    (
        scoring_config,
        profile_config,
        selected_variant,
    ) = load_content_configuration(
        args.content_report
    )

    print("Week 6 Ablation Evaluation")
    print("=" * 84)

    print(f"Ratings:        {len(ratings):,}")
    print(f"Users:          {ratings['userId'].nunique():,}")
    print(f"Movies:         {len(movies):,}")
    print(f"Content config: {selected_variant}")

    # ---------------------------------------------------------
    # Build the exact shared 60 / 20 / 20 split.
    # Only the first 60% is used to construct upstream
    # Content/MF/personal/quality/popularity evidence.
    # ---------------------------------------------------------

    profile_parts: list[pd.DataFrame] = []

    user_test_data: dict[
        int,
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}

    for raw_user_id in ratings["userId"].unique():
        user_id = int(raw_user_id)

        user_ratings = ratings.loc[
            ratings["userId"] == user_id
        ].copy()

        if len(user_ratings) < 10:
            continue

        profile, _, test = (
            build_user_evaluation_split(
                user_ratings
            )
        )

        profile_parts.append(profile)

        if len(test) >= 2:
            user_test_data[user_id] = (
                profile,
                test,
            )

    if not profile_parts:
        raise ValueError(
            "No profile ratings were generated"
        )

    global_profile_ratings = pd.concat(
        profile_parts,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Must match Week 6 ablation training MF exactly.
    # ---------------------------------------------------------

    print("\nTraining evaluation MF...")

    collaborative_model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=40,
        prior_strength=5.0,
        random_state=42,
        shrink_latent=True,
    )

    collaborative_model.fit(
        global_profile_ratings
    )

    # ---------------------------------------------------------
    # Load all trained ablation rankers.
    # ---------------------------------------------------------

    rankers = {}

    for experiment_name, features in EXPERIMENTS.items():
        model_path = (
            args.models_dir
            / f"{experiment_name}.joblib"
        )

        ranker = load_ranker(
            path=str(model_path)
        )

        expected = len(features)
        actual = int(
            ranker.scaler.n_features_in_
        )

        if actual != expected:
            raise ValueError(
                f"{experiment_name}: model expects "
                f"{actual} features but experiment "
                f"defines {expected}"
            )

        rankers[experiment_name] = ranker

    user_accuracies = {
        name: []
        for name in EXPERIMENTS
    }

    user_ndcgs = {
        name: []
        for name in EXPERIMENTS
    }

    weighted_correct = {
        name: 0.0
        for name in EXPERIMENTS
    }

    total_pairs = 0
    users_evaluated = 0

    print("Evaluating final 20%...\n")

    # ---------------------------------------------------------
    # Evaluate all experiments over identical user/movie sets.
    # ---------------------------------------------------------

    for user_id, (
        profile,
        test,
    ) in user_test_data.items():

        test_movie_ids = (
            test["movieId"]
            .astype(int)
            .tolist()
        )

        mf_predictions = (
            collaborative_model.predict(
                user_ids=(
                    [user_id]
                    * len(test_movie_ids)
                ),
                movie_ids=test_movie_ids,
            )
            .set_index("movie_id")[
                "predicted_score"
            ]
            .to_dict()
        )

        content_model = (
            build_content_model_from_frames(
                profile_ratings=profile,
                movies=movies,
                scoring_config=scoring_config,
                profile_config=profile_config,
            )
        )

        reference_ratings = (
            global_profile_ratings.loc[
                global_profile_ratings[
                    "userId"
                ] != user_id
            ].copy()
        )

        popularity = (
            calculate_movie_popularity(
                reference_ratings
            )
        )

        scored_movies = score_movies_by_genre(
            user_id=user_id,
            user_history=profile,
            reference_ratings=reference_ratings,
            movies=movies,
            movie_ids=test_movie_ids,
        )

        personal_by_movie = {
            movie.movie_id:
            movie.personal_score
            for movie in scored_movies
        }

        quality_by_movie = {
            movie.movie_id:
            movie.quality_score
            for movie in scored_movies
        }

        movie_data = {}

        for movie_id in test_movie_ids:
            if movie_id not in mf_predictions:
                continue

            if movie_id not in popularity:
                continue

            if movie_id not in personal_by_movie:
                continue

            if movie_id not in quality_by_movie:
                continue

            actual_rows = test.loc[
                test["movieId"] == movie_id,
                "rating",
            ]

            if actual_rows.empty:
                continue

            content_prediction = (
                content_model.predict_one(
                    user_id=user_id,
                    movie_id=movie_id,
                )
            )

            candidate = AblationCandidate(
                movie_id=movie_id,
                collaborative_score=float(
                    mf_predictions[movie_id]
                ),
                personal_score=float(
                    personal_by_movie[movie_id]
                ),
                content_score=float(
                    content_prediction.predicted_score
                ),
                quality_score=float(
                    quality_by_movie[movie_id]
                ),
                popularity=float(
                    popularity[movie_id]
                ),
            )

            scores = {}

            for (
                experiment_name,
                features,
            ) in EXPERIMENTS.items():

                ranker = rankers[
                    experiment_name
                ]

                raw = (
                    candidate
                    .as_array(features)
                    .reshape(1, -1)
                )

                scaled = (
                    ranker.scaler.transform(raw)
                )

                score = float(
                    ranker.model.decision_function(
                        scaled
                    )[0]
                )

                scores[
                    experiment_name
                ] = score

            movie_data[movie_id] = {
                "actual": float(
                    actual_rows.iloc[0]
                ),
                "scores": scores,
            }

        movie_ids = list(
            movie_data.keys()
        )

        if len(movie_ids) < 2:
            continue

        correct = {
            name: 0.0
            for name in EXPERIMENTS
        }

        user_pairs = 0

        for i in range(len(movie_ids)):
            for j in range(
                i + 1,
                len(movie_ids),
            ):
                first = movie_data[
                    movie_ids[i]
                ]

                second = movie_data[
                    movie_ids[j]
                ]

                if (
                    first["actual"]
                    == second["actual"]
                ):
                    continue

                for name in EXPERIMENTS:
                    correct[name] += (
                        comparison_credit(
                            first_score=(
                                first["scores"][
                                    name
                                ]
                            ),
                            second_score=(
                                second["scores"][
                                    name
                                ]
                            ),
                            first_rating=(
                                first["actual"]
                            ),
                            second_rating=(
                                second["actual"]
                            ),
                        )
                    )

                user_pairs += 1

        if user_pairs == 0:
            continue

        users_evaluated += 1
        total_pairs += user_pairs

        actual_relevance = [
            movie_data[movie_id]["actual"]
            for movie_id in movie_ids
        ]

        for name in EXPERIMENTS:
            accuracy = (
                correct[name]
                / user_pairs
            )

            predicted_scores = [
                movie_data[movie_id][
                    "scores"
                ][name]
                for movie_id in movie_ids
            ]

            ndcg = ndcg_at_k(
                actual_relevance=actual_relevance,
                predicted_scores=predicted_scores,
                k=10,
            )

            user_accuracies[
                name
            ].append(accuracy)

            user_ndcgs[
                name
            ].append(ndcg)

            weighted_correct[
                name
            ] += correct[name]

    if total_pairs == 0:
        raise ValueError(
            "No valid test pairs were generated"
        )

    results = []

    for name, features in EXPERIMENTS.items():
        macro = float(
            np.mean(
                user_accuracies[name]
            )
        )

        weighted = float(
            weighted_correct[name]
            / total_pairs
        )

        ndcg = float(
            np.mean(
                user_ndcgs[name]
            )
        )

        results.append({
            "name": name,
            "features": features,
            "pairwise_macro": macro,
            "pairwise_weighted": weighted,
            "ndcg_at_10": ndcg,
        })

    results.sort(
        key=lambda item: (
            item["pairwise_macro"],
            item["pairwise_weighted"],
            item["ndcg_at_10"],
        ),
        reverse=True,
    )

    print("=" * 84)
    print(
        f"Users evaluated: {users_evaluated:,}"
    )
    print(
        f"Test pairs:      {total_pairs:,}"
    )

    print("\nResults")
    print("=" * 84)

    print(
        f"{'Experiment':<28}"
        f"{'Pairwise':>14}"
        f"{'Weighted':>14}"
        f"{'NDCG@10':>14}"
    )

    print("-" * 84)

    for result in results:
        print(
            f"{result['name']:<28}"
            f"{result['pairwise_macro']:>13.3%}"
            f"{result['pairwise_weighted']:>13.3%}"
            f"{result['ndcg_at_10']:>14.4f}"
        )

    print("-" * 84)

    print(
        f"{'Old ML benchmark':<28}"
        f"{OLD_ML_MACRO:>13.3%}"
        f"{OLD_ML_WEIGHTED:>13.3%}"
        f"{OLD_ML_NDCG:>14.4f}"
    )

    print(
        f"{'IMDb Hybrid baseline':<28}"
        f"{IMDB_BASELINE_MACRO:>13.3%}"
        f"{IMDB_BASELINE_WEIGHTED:>13.3%}"
        f"{IMDB_BASELINE_NDCG:>14.4f}"
    )

    best = results[0]

    print("\nBest Week 6 Ablation")
    print("=" * 84)

    print(f"Model:    {best['name']}")
    print(
        f"Features: "
        f"{', '.join(best['features'])}"
    )
    print(
        f"Macro:    "
        f"{best['pairwise_macro']:.3%}"
    )
    print(
        f"Weighted: "
        f"{best['pairwise_weighted']:.3%}"
    )
    print(
        f"NDCG@10:  "
        f"{best['ndcg_at_10']:.4f}"
    )

    print("\nVs Old ML:")
    print(
        "Macro:    "
        f"{(
            best['pairwise_macro']
            - OLD_ML_MACRO
        ) * 100:+.3f} pp"
    )
    print(
        "Weighted: "
        f"{(
            best['pairwise_weighted']
            - OLD_ML_WEIGHTED
        ) * 100:+.3f} pp"
    )
    print(
        "NDCG@10:  "
        f"{(
            best['ndcg_at_10']
            - OLD_ML_NDCG
        ):+.4f}"
    )

    output = {
        "movies": str(args.movies),
        "content_report": str(
            args.content_report
        ),
        "content_variant": (
            selected_variant
        ),
        "users_evaluated": (
            users_evaluated
        ),
        "test_pairs": total_pairs,
        "benchmarks": {
            "old_ml": {
                "pairwise_macro": (
                    OLD_ML_MACRO
                ),
                "pairwise_weighted": (
                    OLD_ML_WEIGHTED
                ),
                "ndcg_at_10": (
                    OLD_ML_NDCG
                ),
            },
            "imdb_hybrid": {
                "pairwise_macro": (
                    IMDB_BASELINE_MACRO
                ),
                "pairwise_weighted": (
                    IMDB_BASELINE_WEIGHTED
                ),
                "ndcg_at_10": (
                    IMDB_BASELINE_NDCG
                ),
            },
        },
        "results": results,
        "best_by_macro": best["name"],
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved results: {args.output}"
    )


if __name__ == "__main__":
    main()