import numpy as np
import pandas as pd

from src.collaborative.matrix_factorization import (
    BiasedMatrixFactorization,
)
from src.evaluation.metrics import comparison_credit, ndcg_at_k
from src.evaluation.splits import build_user_evaluation_split
from src.hybrid.content_adapter import (
    build_content_model_from_frames,
)
from src.hybrid.genre_recommender import score_movies_by_genre
from src.hybrid.ml_reranker import (
    AblationCandidate,
    calculate_movie_popularity,
    load_ranker,
)
from src.load_data import load_movielens


EXPERIMENTS = {
    "MF + Old Personal + Quality + Popularity": {
        "features": [
            "collaborative",
            "personal",
            "quality",
            "popularity",
        ],
        "path": (
            "src/models/ablations/"
            "mf_oldpersonal_quality_popularity.joblib"
        ),
    },
    "MF + Quality + Popularity": {
        "features": [
            "collaborative",
            "quality",
            "popularity",
        ],
        "path": (
            "src/models/ablations/"
            "mf_quality_popularity.joblib"
        ),
    },
    "MF + Content": {
        "features": [
            "collaborative",
            "content",
        ],
        "path": (
            "src/models/ablations/"
            "mf_content.joblib"
        ),
    },
    "MF + Old Personal": {
        "features": [
            "collaborative",
            "personal",
        ],
        "path": (
            "src/models/ablations/"
            "mf_oldpersonal.joblib"
        ),
    },
    "MF + Old Personal + Content": {
        "features": [
            "collaborative",
            "personal",
            "content",
        ],
        "path": (
            "src/models/ablations/"
            "mf_oldpersonal_content.joblib"
        ),
    },
    "All 5": {
        "features": [
            "collaborative",
            "personal",
            "content",
            "quality",
            "popularity",
        ],
        "path": (
            "src/models/ablations/"
            "all_5.joblib"
        ),
    },
}


def main() -> None:
    ratings, movies = load_movielens("data")

    print("Week 3 Ablation Evaluation")
    print("=" * 76)

    print(f"Ratings: {len(ratings):,}")
    print(f"Users:   {ratings['userId'].nunique():,}")

    # ---------------------------------------------------------
    # Build the exact same 60/20/20 user splits.
    # Hybrid ablations only learn from the first 60%.
    # Final 20% remains untouched for this evaluation.
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

        profile, _, test = build_user_evaluation_split(
            user_ratings
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
    # Same MF used during ablation training:
    # one model trained on everybody's first 60%.
    # ---------------------------------------------------------

    print("\nTraining evaluation MF...")

    collaborative_model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=20,
        prior_strength=5.0,
        random_state=42,
    )

    collaborative_model.fit(
        global_profile_ratings
    )

    # Load all six already-trained logistic rerankers.

    rankers = {
        name: load_ranker(
            path=config["path"]
        )
        for name, config in EXPERIMENTS.items()
    }

    # Each experiment stores one accuracy per user.
    user_accuracies = {
        name: []
        for name in EXPERIMENTS
    }

    user_ndcgs = {
        name: []
        for name in EXPERIMENTS
    }

    # For pair-weighted accuracy.
    weighted_correct = {
        name: 0.0
        for name in EXPERIMENTS
    }

    total_pairs = 0
    users_evaluated = 0

    print("Evaluating final 20%...\n")

    # ---------------------------------------------------------
    # Evaluate every user.
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
                user_ids=[
                    user_id
                ] * len(test_movie_ids),
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
            )
        )

        reference_ratings = (
            global_profile_ratings.loc[
                global_profile_ratings[
                    "userId"
                ] != user_id
            ].copy()
        )

        popularity = calculate_movie_popularity(
            reference_ratings
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
                config,
            ) in EXPERIMENTS.items():

                features = config["features"]
                ranker = rankers[
                    experiment_name
                ]

                raw = (
                    candidate
                    .as_array(features)
                    .reshape(1, -1)
                )

                scaled = (
                    ranker.scaler.transform(
                        raw
                    )
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

        for i in range(
            len(movie_ids)
        ):
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
            accuracy = correct[name] / user_pairs

            predicted_scores = [
                movie_data[movie_id]["scores"][name]
                for movie_id in movie_ids
            ]

            ndcg = ndcg_at_k(
                actual_relevance=actual_relevance,
                predicted_scores=predicted_scores,
                k=10,
            )

            user_accuracies[name].append(accuracy)
            user_ndcgs[name].append(ndcg)

            weighted_correct[name] += correct[name]

    if total_pairs == 0:
        raise ValueError(
            "No valid test pairs were generated"
        )

    # ---------------------------------------------------------
    # Final results.
    # ---------------------------------------------------------

    print("=" * 76)
    print(
        f"Users evaluated: "
        f"{users_evaluated:,}"
    )
    print(
        f"Test pairs:      "
        f"{total_pairs:,}"
    )

    print("\nResults")
    print("=" * 76)

    print(
        f"{'Experiment':<44}"
        f"{'Pairwise':>14}"
        f"{'Weighted':>14}"
        f"{'NDCG@10':>14}"
    )

    print("-" * 76)

    results = []

    for name in EXPERIMENTS:

        macro = float(np.mean(user_accuracies[name]))

        weighted = (
            weighted_correct[name]
            / total_pairs
        )

        ndcg = float(np.mean(user_ndcgs[name]))

        results.append(
            (
                name,
                macro,
                weighted,
                ndcg,
            )
        )

    # Highest macro first.
    results.sort(
        key=lambda result: result[1],
        reverse=True,
    )

    for (
        name,
        macro,
        weighted,
        ndcg,
    ) in results:

        print(
            f"{name:<44}"
            f"{macro:>13.3%}"
            f"{weighted:>13.3%}"
            f"{ndcg:>14.4f}"
        )

    print("-" * 76)

    # Your frozen benchmarks from the immediately
    # preceding evaluation.
    old_ml_macro = 0.63876
    old_ml_weighted = 0.68516

    hybrid_v1_macro = 0.63253
    hybrid_v1_weighted = 0.67798

    print(
        f"{'Old ML benchmark':<44}"
        f"{old_ml_macro:>13.3%}"
        f"{old_ml_weighted:>13.3%}"
    )

    print(
        f"{'Hybrid V1 benchmark':<44}"
        f"{hybrid_v1_macro:>13.3%}"
        f"{hybrid_v1_weighted:>13.3%}"
    )

    print("\nBest Ablation")
    print("=" * 76)

    best_name, best_macro, best_weighted, best_ndcg = (
        results[0]
    )

    print(
        f"Model:    {best_name}"
    )
    print(
        f"Macro:    {best_macro:.3%}"
    )
    print(
        f"Weighted: {best_weighted:.3%}"
    )
    print(
        f"NDCG@10:  {best_ndcg:.4f}"
    )

    print(
        "\nVs Old ML:"
    )

    print(
        f"Macro:    "
        f"{(best_macro - old_ml_macro) * 100:+.3f} pp"
    )

    print(
        f"Weighted: "
        f"{(best_weighted - old_ml_weighted) * 100:+.3f} pp"
    )


if __name__ == "__main__":
    main()