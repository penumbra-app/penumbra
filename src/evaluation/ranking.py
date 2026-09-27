# src/evaluation/ranking.py
from dataclasses import dataclass

import numpy as np
import pandas as pd


from src.collaborative.baseline import MovieAverageBaseline
from src.collaborative.matrix_factorization import BiasedMatrixFactorization
from src.hybrid.genre_recommender import score_movies_by_genre
from src.evaluation.splits import build_user_evaluation_split

from src.evaluation.metrics import (
    comparison_credit,
    mean_absolute_error,
    ndcg_at_k,
    root_mean_squared_error,
)

from src.hybrid.content_adapter import build_content_model_from_frames
from src.hybrid.ml_reranker import (
    HybridCandidate,
    MovieFeatures,
    calculate_movie_popularity,
    load_ranker,
)

@dataclass(frozen=True)
class UserPairwiseResult:
    user_id: int
    rating_count: int
    pairs_evaluated: int
    baseline_accuracy: float
    heuristic_accuracy: float
    ml_accuracy: float
    matrix_factorization_accuracy: float
    content_accuracy: float = 0.0
    hybrid_accuracy: float = 0.0

    baseline_mae: float = 0.0
    baseline_rmse: float = 0.0
    baseline_ndcg: float = 0.0

    heuristic_mae: float = 0.0
    heuristic_rmse: float = 0.0
    heuristic_ndcg: float = 0.0

    matrix_factorization_mae: float = 0.0
    matrix_factorization_rmse: float = 0.0
    matrix_factorization_ndcg: float = 0.0

    content_mae: float = 0.0
    content_rmse: float = 0.0
    content_ndcg: float = 0.0

    ml_ndcg: float = 0.0
    hybrid_ndcg: float = 0.0

@dataclass(frozen=True)
class PairwiseEvaluation:
    baseline_accuracy: float
    heuristic_accuracy: float
    ml_accuracy: float
    matrix_factorization_accuracy: float
    users_evaluated: int
    pairs_evaluated: int
    user_results: list[UserPairwiseResult]
    content_accuracy: float = 0.0
    hybrid_accuracy: float = 0.0



def evaluate_pairwise_accuracy(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> PairwiseEvaluation:
    train_ratings_list: list[pd.DataFrame] = []
    profile_ratings_list: list[pd.DataFrame] = []
    user_test_data: dict[
        int,
        tuple[pd.DataFrame, pd.DataFrame, int],
    ] = {}

    for user_id in ratings["userId"].unique():
        user_ratings = ratings.loc[ratings["userId"] == user_id].copy()

        if len(user_ratings) < 10:
            train_ratings_list.append(user_ratings)
            continue

        profile, train, test = build_user_evaluation_split(user_ratings)
        profile_ratings_list.append(profile)

        train_ratings_list.append(profile)
        train_ratings_list.append(train)

        if len(test) >= 2:
            user_test_data[int(user_id)] = (
                profile,
                test,
                len(user_ratings),
            )

    global_profile_ratings = pd.concat(
        profile_ratings_list,
        ignore_index=True,
    )

    if not train_ratings_list:
        raise ValueError("No training ratings were generated")

    all_train_ratings = pd.concat(
        train_ratings_list,
        ignore_index=True,
    )

    if all_train_ratings.empty:
        raise ValueError("Training ratings are empty")

    baseline_model = MovieAverageBaseline(
        prior_strength=10.0
    )

    baseline_model.fit(all_train_ratings)

    matrix_factorization_model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=20,
        prior_strength=5.0,
        random_state=42,
    )

    matrix_factorization_model.fit(all_train_ratings)

    hybrid_matrix_factorization_model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=20,
        prior_strength=5.0,
        random_state=42,
    )

    hybrid_matrix_factorization_model.fit(
        global_profile_ratings
    )

    ranker = load_ranker()
    hybrid_ranker = load_ranker(
        path="src/models/hybrid_v1_reranker.joblib"
    )
    content_user_accuracies: list[float] = []
    hybrid_user_accuracies: list[float] = []

    baseline_user_accuracies: list[float] = []
    heuristic_user_accuracies: list[float] = []
    ml_user_accuracies: list[float] = []
    matrix_factorization_user_accuracies: list[float] = []

    user_results: list[UserPairwiseResult] = []

    total_pairs = 0

    for user_id, (profile, test, rating_count) in user_test_data.items():

        test_movie_ids = (
            test["movieId"]
            .astype(int)
            .tolist()
        )

        baseline_preds = (
            baseline_model.predict(
                user_ids=[user_id] * len(test_movie_ids),
                movie_ids=test_movie_ids,
            )
            .set_index("movie_id")["predicted_score"]
            .to_dict()
        )

        matrix_factorization_preds = (
            matrix_factorization_model.predict(
                user_ids=[user_id] * len(test_movie_ids),
                movie_ids=test_movie_ids,
            )
            .set_index("movie_id")["predicted_score"]
            .to_dict()
        )

        hybrid_mf_preds = (
            hybrid_matrix_factorization_model.predict(
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

        hybrid_reference_ratings = global_profile_ratings.loc[
            global_profile_ratings["userId"] != user_id
        ].copy()

        hybrid_popularity = calculate_movie_popularity(
            hybrid_reference_ratings
        )

        hybrid_scored_movies = score_movies_by_genre(
            user_id=user_id,
            user_history=profile,
            reference_ratings=hybrid_reference_ratings,
            movies=movies,
            movie_ids=test_movie_ids,
        )

        hybrid_quality = {
            movie.movie_id: movie.quality_score
            for movie in hybrid_scored_movies
        }

        reference_ratings = all_train_ratings.loc[
            all_train_ratings["userId"] != user_id
        ]

        popularity = calculate_movie_popularity(
            reference_ratings
        )

        scored_movies = score_movies_by_genre(
            user_id=user_id,
            user_history=profile,
            reference_ratings=all_train_ratings,
            movies=movies,
            movie_ids=test_movie_ids,
        )

        if len(scored_movies) < 2:
            continue

        movie_data = {}

        for movie in scored_movies:
            movie_id = movie.movie_id
            if movie_id not in hybrid_mf_preds:
                continue

            if movie_id not in hybrid_quality:
                continue

            if movie_id not in hybrid_popularity:
                continue

            if movie_id not in popularity:
                continue

            if movie_id not in baseline_preds:
                continue

            if movie_id not in matrix_factorization_preds:
                continue

            actual_rows = test.loc[
                test["movieId"] == movie_id,
                "rating",
            ]

            if actual_rows.empty:
                continue

            actual_rating = float(
                actual_rows.iloc[0]
            )

            features = MovieFeatures(
                personal_score=movie.personal_score,
                quality_score=movie.quality_score,
                popularity=popularity[movie_id],
            )
            raw_features = features.as_array().reshape(1, -1)
            scaled_features = ranker.scaler.transform(raw_features)
            ml_score = float(
                ranker.model.decision_function(scaled_features)[0])

            content_prediction = content_model.predict_one(
                user_id=user_id,
                movie_id=movie_id,
            )

            content_score = float(
                content_prediction.predicted_score
            )

            hybrid_candidate = HybridCandidate(
                movie_id=movie_id,
                content_score=content_score,
                collaborative_score=float(
                    hybrid_mf_preds[movie_id]
                ),
                quality_score=float(
                    hybrid_quality[movie_id]
                ),
                popularity=float(
                    hybrid_popularity[movie_id]
                ),
            )

            hybrid_raw = hybrid_candidate.as_array().reshape(1, -1)
            hybrid_scaled = hybrid_ranker.scaler.transform(
                hybrid_raw
            )

            hybrid_score = float(
                hybrid_ranker.model.decision_function(
                    hybrid_scaled
                )[0]
            )

            movie_data[movie_id] = {
                "actual": actual_rating,
                "baseline": float(
                    baseline_preds[movie_id]
                ),
                "heuristic": float(
                    movie.predicted_rating
                ),
                "ml": ml_score,
                "matrix_factorization": float(
                    matrix_factorization_preds[movie_id]
                ),
                "content": content_score,
                "hybrid": hybrid_score,
            }


        movie_ids = list(movie_data.keys())
        if len(movie_ids) < 2:
            continue

        baseline_correct = 0.0
        heuristic_correct = 0.0
        ml_correct = 0.0
        matrix_factorization_correct = 0.0
        content_correct = 0.0
        hybrid_correct = 0.0

        user_pairs = 0

        for i in range(len(movie_ids)):
            for j in range(i + 1, len(movie_ids)):

                first = movie_data[movie_ids[i]]
                second = movie_data[movie_ids[j]]

                if first["actual"] == second["actual"]:
                    continue

                baseline_correct += comparison_credit(
                    first_score=first["baseline"],
                    second_score=second["baseline"],
                    first_rating=first["actual"],
                    second_rating=second["actual"],
                )

                heuristic_correct += comparison_credit(
                    first_score=first["heuristic"],
                    second_score=second["heuristic"],
                    first_rating=first["actual"],
                    second_rating=second["actual"],
                )

                ml_correct += comparison_credit(
                    first_score=first["ml"],
                    second_score=second["ml"],
                    first_rating=first["actual"],
                    second_rating=second["actual"],
                )

                matrix_factorization_correct += comparison_credit(
                    first_score=first["matrix_factorization"],
                    second_score=second["matrix_factorization"],
                    first_rating=first["actual"],
                    second_rating=second["actual"],
                )

                content_correct += comparison_credit(
                    first_score=first["content"],
                    second_score=second["content"],
                    first_rating=first["actual"],
                    second_rating=second["actual"],
                )

                hybrid_correct += comparison_credit(
                    first_score=first["hybrid"],
                    second_score=second["hybrid"],
                    first_rating=first["actual"],
                    second_rating=second["actual"],
                )

                user_pairs += 1

        if user_pairs == 0:
            continue

        user_baseline_accuracy = baseline_correct / user_pairs
        user_heuristic_accuracy = heuristic_correct / user_pairs
        user_ml_accuracy = ml_correct / user_pairs
        user_matrix_factorization_accuracy = (
            matrix_factorization_correct / user_pairs
        )

        baseline_user_accuracies.append(
            user_baseline_accuracy
        )

        heuristic_user_accuracies.append(
            user_heuristic_accuracy
        )

        ml_user_accuracies.append(
            user_ml_accuracy
        )

        matrix_factorization_user_accuracies.append(
            user_matrix_factorization_accuracy
        )

        user_content_accuracy = content_correct / user_pairs
        user_hybrid_accuracy = hybrid_correct / user_pairs

        content_user_accuracies.append(
            user_content_accuracy
        )

        hybrid_user_accuracies.append(
            user_hybrid_accuracy
        )

        actual_ratings = [
        movie_data[movie_id]["actual"]
            for movie_id in movie_ids
        ]

        baseline_scores = [
            movie_data[movie_id]["baseline"]
            for movie_id in movie_ids
        ]

        heuristic_scores = [
            movie_data[movie_id]["heuristic"]
            for movie_id in movie_ids
        ]

        ml_scores = [
            movie_data[movie_id]["ml"]
            for movie_id in movie_ids
        ]

        mf_scores = [
            movie_data[movie_id]["matrix_factorization"]
            for movie_id in movie_ids
        ]

        content_scores = [
            movie_data[movie_id]["content"]
            for movie_id in movie_ids
        ]

        hybrid_scores = [
            movie_data[movie_id]["hybrid"]
            for movie_id in movie_ids
        ]

        user_baseline_mae = mean_absolute_error(actual_ratings, baseline_scores)
        user_baseline_rmse = root_mean_squared_error(actual_ratings, baseline_scores)
        user_baseline_ndcg = ndcg_at_k(actual_ratings, baseline_scores, k=10)

        user_heuristic_mae = mean_absolute_error(actual_ratings, heuristic_scores)
        user_heuristic_rmse = root_mean_squared_error(actual_ratings, heuristic_scores)
        user_heuristic_ndcg = ndcg_at_k(actual_ratings, heuristic_scores, k=10)

        user_mf_mae = mean_absolute_error(actual_ratings, mf_scores)
        user_mf_rmse = root_mean_squared_error(actual_ratings, mf_scores)
        user_mf_ndcg = ndcg_at_k(actual_ratings, mf_scores, k=10)

        user_content_mae = mean_absolute_error(actual_ratings, content_scores)
        user_content_rmse = root_mean_squared_error(actual_ratings, content_scores)
        user_content_ndcg = ndcg_at_k(actual_ratings, content_scores, k=10)

        user_ml_ndcg = ndcg_at_k(actual_ratings, ml_scores, k=10)
        user_hybrid_ndcg = ndcg_at_k(actual_ratings, hybrid_scores, k=10)

        user_results.append(
            UserPairwiseResult(
                user_id=int(user_id),
                rating_count=rating_count,
                pairs_evaluated=user_pairs,
                baseline_accuracy=user_baseline_accuracy,
                heuristic_accuracy=user_heuristic_accuracy,
                ml_accuracy=user_ml_accuracy,
                matrix_factorization_accuracy=user_matrix_factorization_accuracy,
                content_accuracy=user_content_accuracy,
                hybrid_accuracy=user_hybrid_accuracy,
                baseline_mae=user_baseline_mae,
                baseline_rmse=user_baseline_rmse,
                baseline_ndcg=user_baseline_ndcg,

                heuristic_mae=user_heuristic_mae,
                heuristic_rmse=user_heuristic_rmse,
                heuristic_ndcg=user_heuristic_ndcg,

                matrix_factorization_mae=user_mf_mae,
                matrix_factorization_rmse=user_mf_rmse,
                matrix_factorization_ndcg=user_mf_ndcg,

                content_mae=user_content_mae,
                content_rmse=user_content_rmse,
                content_ndcg=user_content_ndcg,

                ml_ndcg=user_ml_ndcg,
                hybrid_ndcg=user_hybrid_ndcg,
            )
        )
        
        total_pairs += user_pairs

    if not baseline_user_accuracies:
        raise ValueError(
            "No valid test pairs were generated"
        )
    return PairwiseEvaluation(
        baseline_accuracy=float(
            np.mean(baseline_user_accuracies)
        ),
        heuristic_accuracy=float(
            np.mean(heuristic_user_accuracies)
        ),
        ml_accuracy=float(
            np.mean(ml_user_accuracies)
        ),
        matrix_factorization_accuracy=float(
            np.mean(matrix_factorization_user_accuracies)
        ),
        users_evaluated=len(
            baseline_user_accuracies
        ),
        pairs_evaluated=total_pairs,
        user_results=user_results,
        content_accuracy=float(
            np.mean(content_user_accuracies)
        ),
        hybrid_accuracy=float(
            np.mean(hybrid_user_accuracies)
        ),
    )
