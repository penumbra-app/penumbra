from __future__ import annotations

import numpy as np
import pandas as pd

from src.collaborative.matrix_factorization import BiasedMatrixFactorization
from src.evaluation.metrics import comparison_credit, ndcg_at_k
from src.evaluation.splits import build_user_evaluation_split
from src.hybrid.content_adapter import build_content_model_from_frames
from src.hybrid.genre_recommender import score_movies_by_genre
from src.hybrid.ml_reranker import calculate_movie_popularity
from src.load_data import load_movielens


SIGNALS = (
    "personal",
    "quality",
    "popularity",
    "content",
    "collaborative",
)

DISPLAY_NAMES = {
    "personal": "Personal",
    "quality": "Quality",
    "popularity": "Popularity",
    "content": "Content",
    "collaborative": "Collaborative (MF)",
}


def build_evaluation_data(
    ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, tuple[pd.DataFrame, pd.DataFrame]]]:
    profile_parts: list[pd.DataFrame] = []
    user_test_data: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for user_id in ratings["userId"].unique():
        user_ratings = ratings.loc[
            ratings["userId"] == user_id
        ].copy()

        if len(user_ratings) < 10:
            continue

        profile, _, test = build_user_evaluation_split(user_ratings)

        profile_parts.append(profile)

        if len(test) >= 2:
            user_test_data[int(user_id)] = (profile, test)

    if not profile_parts:
        raise ValueError("No profile ratings were generated")

    global_profile_ratings = pd.concat(
        profile_parts,
        ignore_index=True,
    )

    return global_profile_ratings, user_test_data


def evaluate_signal_usefulness(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    global_profile_ratings, user_test_data = build_evaluation_data(
        ratings
    )

    # This MF is intentionally fit on the same profile-only data used by
    # Hybrid V1's collaborative signal. That keeps this diagnostic aligned
    # with the signal the hybrid reranker actually receives.
    collaborative_model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=20,
        prior_strength=5.0,
        random_state=42,
    )

    print("Training profile-only collaborative model...")
    collaborative_model.fit(global_profile_ratings)

    user_pairwise: dict[str, list[float]] = {
        signal: []
        for signal in SIGNALS
    }

    user_ndcg: dict[str, list[float]] = {
        signal: []
        for signal in SIGNALS
    }

    weighted_correct = {
        signal: 0.0
        for signal in SIGNALS
    }

    total_pairs = 0
    users_evaluated = 0

    print("Evaluating raw signals on final 20%...\n")

    for user_id, (profile, test) in user_test_data.items():
        test_movie_ids = (
            test["movieId"]
            .astype(int)
            .tolist()
        )

        # Exclude the target user from population statistics so hidden
        # information from that user cannot leak into quality/popularity.
        reference_ratings = global_profile_ratings.loc[
            global_profile_ratings["userId"] != user_id
        ].copy()

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

        if len(scored_movies) < 2:
            continue

        collaborative_predictions = (
            collaborative_model.predict(
                user_ids=[user_id] * len(test_movie_ids),
                movie_ids=test_movie_ids,
            )
            .set_index("movie_id")["predicted_score"]
            .to_dict()
        )

        content_model = build_content_model_from_frames(
            profile_ratings=profile,
            movies=movies,
        )

        movie_data: dict[int, dict[str, object]] = {}

        for movie in scored_movies:
            movie_id = int(movie.movie_id)

            if movie_id not in popularity:
                continue

            if movie_id not in collaborative_predictions:
                continue

            actual_rows = test.loc[
                test["movieId"] == movie_id,
                "rating",
            ]

            if actual_rows.empty:
                continue

            content_prediction = content_model.predict_one(
                user_id=user_id,
                movie_id=movie_id,
            )

            movie_data[movie_id] = {
                "actual": float(actual_rows.iloc[0]),
                "scores": {
                    "personal": float(movie.personal_score),
                    "quality": float(movie.quality_score),
                    "popularity": float(popularity[movie_id]),
                    "content": float(
                        content_prediction.predicted_score
                    ),
                    "collaborative": float(
                        collaborative_predictions[movie_id]
                    ),
                },
            }

        movie_ids = list(movie_data)

        if len(movie_ids) < 2:
            continue

        correct = {
            signal: 0.0
            for signal in SIGNALS
        }

        user_pairs = 0

        for i in range(len(movie_ids)):
            for j in range(i + 1, len(movie_ids)):
                first = movie_data[movie_ids[i]]
                second = movie_data[movie_ids[j]]

                first_rating = float(first["actual"])
                second_rating = float(second["actual"])

                if first_rating == second_rating:
                    continue

                first_scores = first["scores"]
                second_scores = second["scores"]

                for signal in SIGNALS:
                    correct[signal] += comparison_credit(
                        first_score=float(first_scores[signal]),
                        second_score=float(second_scores[signal]),
                        first_rating=first_rating,
                        second_rating=second_rating,
                    )

                user_pairs += 1

        if user_pairs == 0:
            continue

        actual_relevance = [
            float(movie_data[movie_id]["actual"])
            for movie_id in movie_ids
        ]

        for signal in SIGNALS:
            predicted_scores = [
                float(movie_data[movie_id]["scores"][signal])
                for movie_id in movie_ids
            ]

            pairwise_accuracy = (
                correct[signal] / user_pairs
            )

            ndcg = ndcg_at_k(
                actual_relevance=actual_relevance,
                predicted_scores=predicted_scores,
                k=10,
            )

            user_pairwise[signal].append(
                pairwise_accuracy
            )

            user_ndcg[signal].append(
                ndcg
            )

            weighted_correct[signal] += (
                correct[signal]
            )

        total_pairs += user_pairs
        users_evaluated += 1

    if users_evaluated == 0 or total_pairs == 0:
        raise ValueError(
            "No valid users or test pairs were generated"
        )

    print("=" * 84)
    print("Week 5 Signal Usefulness")
    print("=" * 84)

    print(f"Users evaluated: {users_evaluated:,}")
    print(f"Test pairs:      {total_pairs:,}")
    print()

    print(
        f"{'Signal':<24}"
        f"{'Pairwise':>14}"
        f"{'Weighted':>14}"
        f"{'NDCG@10':>14}"
    )

    print("-" * 66)

    results = []

    for signal in SIGNALS:
        macro_pairwise = float(
            np.mean(user_pairwise[signal])
        )

        pair_weighted = (
            weighted_correct[signal]
            / total_pairs
        )

        macro_ndcg = float(
            np.mean(user_ndcg[signal])
        )

        results.append(
            (
                signal,
                macro_pairwise,
                pair_weighted,
                macro_ndcg,
            )
        )

    # Sorting is only for readability. It does not select or tune a model.
    results.sort(
        key=lambda result: result[1],
        reverse=True,
    )

    for (
        signal,
        macro_pairwise,
        pair_weighted,
        macro_ndcg,
    ) in results:
        print(
            f"{DISPLAY_NAMES[signal]:<24}"
            f"{macro_pairwise:>13.3%}"
            f"{pair_weighted:>13.3%}"
            f"{macro_ndcg:>14.4f}"
        )

    print("-" * 66)

    print(
        "\nPairwise = mean per-user pairwise accuracy."
    )
    print(
        "Weighted = pairwise accuracy weighted by each user's number of pairs."
    )
    print(
        "NDCG@10 = mean per-user top-10 ranking quality."
    )


def main() -> None:
    ratings, movies = load_movielens("data")

    print("Week 5 Raw Signal Diagnostic")
    print("=" * 84)

    print(f"Ratings:      {len(ratings):,}")
    print(f"Users:        {ratings['userId'].nunique():,}")
    print(f"Rated movies: {ratings['movieId'].nunique():,}")
    print(f"Movie rows:   {len(movies):,}")
    print()

    evaluate_signal_usefulness(
        ratings=ratings,
        movies=movies,
    )


if __name__ == "__main__":
    main()