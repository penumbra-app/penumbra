from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from src.collaborative.matrix_factorization import BiasedMatrixFactorization
from src.evaluation.metrics import comparison_credit
from src.evaluation.splits import build_user_evaluation_split
from src.hybrid.content_adapter import build_content_model_from_frames
from src.hybrid.genre_recommender import score_movies_by_genre
from src.hybrid.ml_reranker import (
    HybridCandidate,
    MovieFeatures,
    calculate_movie_popularity,
    load_ranker,
)
from src.load_data import load_movielens


SIGNALS = (
    "personal",
    "quality",
    "popularity",
    "mf",
    "content",
)

MODELS = (
    "old_ml",
    "hybrid",
    "mf",
    "content",
)


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def format_percent(value: float) -> str:
    if np.isnan(value):
        return "      —"
    return f"{value:8.3%}"


def quantile_bucket(
    value: float,
    low_cutoff: float,
    high_cutoff: float,
) -> str:
    if value <= low_cutoff:
        return "Low"
    if value <= high_cutoff:
        return "Medium"
    return "High"


def directional_credit(
    first_score: float,
    second_score: float,
    first_rating: float,
    second_rating: float,
) -> float:
    return comparison_credit(
        first_score=first_score,
        second_score=second_score,
        first_rating=first_rating,
        second_rating=second_rating,
    )


def model_pair_credit(
    first: dict[str, float],
    second: dict[str, float],
    model: str,
) -> float:
    return directional_credit(
        first[model],
        second[model],
        first["actual"],
        second["actual"],
    )


def pair_signal_gap(
    first: dict[str, float],
    second: dict[str, float],
    signal: str,
) -> float:
    return abs(first[signal] - second[signal])


def signal_direction_agreement(
    first: dict[str, float],
    second: dict[str, float],
    left: str,
    right: str,
) -> float:
    left_direction = np.sign(first[left] - second[left])
    right_direction = np.sign(first[right] - second[right])

    if left_direction == 0 and right_direction == 0:
        return 1.0

    if left_direction == 0 or right_direction == 0:
        return 0.5

    return float(left_direction == right_direction)


def print_table_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def build_evaluation_data(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
]:
    train_ratings_list: list[pd.DataFrame] = []
    profile_ratings_list: list[pd.DataFrame] = []

    user_test_data: dict[
        int,
        tuple[pd.DataFrame, pd.DataFrame, int],
    ] = {}

    for user_id in ratings["userId"].unique():
        user_ratings = ratings.loc[
            ratings["userId"] == user_id
        ].copy()

        if len(user_ratings) < 10:
            train_ratings_list.append(user_ratings)
            continue

        profile, train, test = build_user_evaluation_split(
            user_ratings
        )

        profile_ratings_list.append(profile)
        train_ratings_list.append(profile)
        train_ratings_list.append(train)

        if len(test) >= 2:
            user_test_data[int(user_id)] = (
                profile,
                test,
                len(user_ratings),
            )

    if not profile_ratings_list:
        raise ValueError("No profile ratings were generated")

    if not train_ratings_list:
        raise ValueError("No training ratings were generated")

    global_profile_ratings = pd.concat(
        profile_ratings_list,
        ignore_index=True,
    )

    all_train_ratings = pd.concat(
        train_ratings_list,
        ignore_index=True,
    )

    print("Training evaluation MF...")

    mf_model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=20,
        prior_strength=5.0,
        random_state=42,
    )
    mf_model.fit(all_train_ratings)

    print("Training hybrid profile-only MF...")

    hybrid_mf_model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=20,
        prior_strength=5.0,
        random_state=42,
    )
    hybrid_mf_model.fit(global_profile_ratings)

    old_ranker = load_ranker()

    hybrid_ranker = load_ranker(
        path="src/models/hybrid_v1_reranker.joblib"
    )

    movie_rows: list[dict[str, float]] = []
    pair_rows: list[dict[str, float]] = []

    total_users = len(user_test_data)

    for user_number, (
        user_id,
        (profile, test, rating_count),
    ) in enumerate(user_test_data.items(), start=1):

        if user_number % 50 == 0 or user_number == total_users:
            print(
                f"Evaluating user "
                f"{user_number:,}/{total_users:,}..."
            )

        test_movie_ids = (
            test["movieId"]
            .astype(int)
            .tolist()
        )

        mf_frame = mf_model.predict(
            user_ids=[user_id] * len(test_movie_ids),
            movie_ids=test_movie_ids,
        )

        mf_scores = (
            mf_frame
            .set_index("movie_id")["predicted_score"]
            .to_dict()
        )

        mf_confidence = (
            mf_frame
            .set_index("movie_id")["confidence"]
            .to_dict()
        )

        hybrid_mf_frame = hybrid_mf_model.predict(
            user_ids=[user_id] * len(test_movie_ids),
            movie_ids=test_movie_ids,
        )

        hybrid_mf_scores = (
            hybrid_mf_frame
            .set_index("movie_id")["predicted_score"]
            .to_dict()
        )

        hybrid_mf_confidence = (
            hybrid_mf_frame
            .set_index("movie_id")["confidence"]
            .to_dict()
        )

        content_model = build_content_model_from_frames(
            profile_ratings=profile,
            movies=movies,
        )

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

        hybrid_reference_ratings = (
            global_profile_ratings.loc[
                global_profile_ratings["userId"] != user_id
            ].copy()
        )

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

        user_movies: list[dict[str, float]] = []

        for movie in scored_movies:
            movie_id = movie.movie_id

            required = (
                movie_id in mf_scores
                and movie_id in mf_confidence
                and movie_id in hybrid_mf_scores
                and movie_id in hybrid_mf_confidence
                and movie_id in popularity
                and movie_id in hybrid_popularity
                and movie_id in hybrid_quality
            )

            if not required:
                continue

            actual_rows = test.loc[
                test["movieId"] == movie_id,
                "rating",
            ]

            if actual_rows.empty:
                continue

            actual_rating = float(actual_rows.iloc[0])

            old_features = MovieFeatures(
                personal_score=movie.personal_score,
                quality_score=movie.quality_score,
                popularity=popularity[movie_id],
            )

            old_raw = old_features.as_array().reshape(1, -1)
            old_scaled = old_ranker.scaler.transform(old_raw)

            old_ml_score = float(
                old_ranker.model.decision_function(
                    old_scaled
                )[0]
            )

            content_prediction = content_model.predict_one(
                user_id=user_id,
                movie_id=movie_id,
            )

            content_score = float(
                content_prediction.predicted_score
            )

            content_confidence = float(
                content_prediction.confidence
            )

            hybrid_candidate = HybridCandidate(
                movie_id=movie_id,
                content_score=content_score,
                collaborative_score=float(
                    hybrid_mf_scores[movie_id]
                ),
                quality_score=float(
                    hybrid_quality[movie_id]
                ),
                popularity=float(
                    hybrid_popularity[movie_id]
                ),
            )

            hybrid_raw = (
                hybrid_candidate
                .as_array()
                .reshape(1, -1)
            )

            hybrid_scaled = (
                hybrid_ranker.scaler.transform(
                    hybrid_raw
                )
            )

            hybrid_score = float(
                hybrid_ranker.model.decision_function(
                    hybrid_scaled
                )[0]
            )

            movie_popularity_count = int(
                (
                    all_train_ratings["movieId"]
                    == movie_id
                ).sum()
            )

            row = {
                "user_id": int(user_id),
                "movie_id": int(movie_id),
                "actual": actual_rating,

                "personal": float(
                    movie.personal_score
                ),
                "quality": float(
                    movie.quality_score
                ),
                "popularity": float(
                    popularity[movie_id]
                ),
                "mf": float(
                    mf_scores[movie_id]
                ),
                "content": content_score,

                "old_ml": old_ml_score,
                "hybrid": hybrid_score,

                "mf_confidence": float(
                    mf_confidence[movie_id]
                ),
                "hybrid_mf_confidence": float(
                    hybrid_mf_confidence[movie_id]
                ),
                "content_confidence": (
                    content_confidence
                ),

                "user_rating_count": int(
                    rating_count
                ),
                "profile_rating_count": int(
                    len(profile)
                ),
                "movie_rating_count": (
                    movie_popularity_count
                ),
            }

            movie_rows.append(row)
            user_movies.append(row)

        for first, second in combinations(
            user_movies,
            2,
        ):
            if first["actual"] == second["actual"]:
                continue

            old_credit = model_pair_credit(
                first,
                second,
                "old_ml",
            )

            hybrid_credit = model_pair_credit(
                first,
                second,
                "hybrid",
            )

            mf_credit = model_pair_credit(
                first,
                second,
                "mf",
            )

            content_credit = model_pair_credit(
                first,
                second,
                "content",
            )

            if old_credit > hybrid_credit:
                disagreement = "Old ML only"
            elif hybrid_credit > old_credit:
                disagreement = "Hybrid only"
            elif old_credit == 1.0:
                disagreement = "Both correct"
            elif old_credit == 0.0:
                disagreement = "Both wrong"
            else:
                disagreement = "Shared tie"

            pair = {
                "user_id": int(user_id),
                "old_ml_credit": old_credit,
                "hybrid_credit": hybrid_credit,
                "mf_credit": mf_credit,
                "content_credit": content_credit,
                "disagreement": disagreement,

                "actual_gap": abs(
                    first["actual"]
                    - second["actual"]
                ),

                "user_rating_count": int(
                    rating_count
                ),

                "movie_rating_count": float(
                    (
                        first["movie_rating_count"]
                        + second["movie_rating_count"]
                    )
                    / 2
                ),

                "mf_confidence": float(
                    (
                        first["mf_confidence"]
                        + second["mf_confidence"]
                    )
                    / 2
                ),

                "content_confidence": float(
                    (
                        first["content_confidence"]
                        + second["content_confidence"]
                    )
                    / 2
                ),
            }

            for signal in SIGNALS:
                pair[f"{signal}_gap"] = pair_signal_gap(
                    first,
                    second,
                    signal,
                )

                pair[f"{signal}_credit"] = directional_credit(
                    first[signal],
                    second[signal],
                    first["actual"],
                    second["actual"],
                )

            for left, right in combinations(
                SIGNALS,
                2,
            ):
                pair[
                    f"agree_{left}_{right}"
                ] = signal_direction_agreement(
                    first,
                    second,
                    left,
                    right,
                )

            pair_rows.append(pair)

    return movie_rows, pair_rows


def print_signal_correlations(
    movie_frame: pd.DataFrame,
) -> None:
    print_table_header(
        "1. Signal Correlation"
    )

    correlation = (
        movie_frame[list(SIGNALS)]
        .corr(method="pearson")
    )

    print(
        correlation.to_string(
            float_format=lambda value: f"{value:8.3f}"
        )
    )

    print()
    print(
        "Values near +1 mean the signals tend to move "
        "together; values near 0 indicate weak linear "
        "relationship."
    )


def print_signal_agreement(
    pair_frame: pd.DataFrame,
) -> None:
    print_table_header(
        "2. Pairwise Signal Direction Agreement"
    )

    print(
        f"{'Signals':<32} {'Agreement':>12}"
    )
    print("-" * 46)

    rows = []

    for left, right in combinations(
        SIGNALS,
        2,
    ):
        column = f"agree_{left}_{right}"
        agreement = float(
            pair_frame[column].mean()
        )
        rows.append(
            (agreement, left, right)
        )

    rows.sort(reverse=True)

    for agreement, left, right in rows:
        label = f"{left} / {right}"
        print(
            f"{label:<32} {agreement:>11.3%}"
        )


def print_disagreement_analysis(
    pair_frame: pd.DataFrame,
) -> None:
    print_table_header(
        "3. Old ML vs Hybrid V1 Disagreement"
    )

    order = (
        "Both correct",
        "Old ML only",
        "Hybrid only",
        "Both wrong",
        "Shared tie",
    )

    print(
        f"{'Outcome':<20} "
        f"{'Pairs':>12} "
        f"{'Share':>10}"
    )
    print("-" * 46)

    total = len(pair_frame)

    for category in order:
        count = int(
            (
                pair_frame["disagreement"]
                == category
            ).sum()
        )

        share = count / total if total else 0.0

        print(
            f"{category:<20} "
            f"{count:>12,} "
            f"{share:>9.3%}"
        )

    print()
    print("Signal gaps by disagreement bucket")
    print("-" * 96)

    header = (
        f"{'Outcome':<18}"
        f"{'Personal':>11}"
        f"{'Quality':>11}"
        f"{'Popularity':>12}"
        f"{'MF':>11}"
        f"{'Content':>11}"
        f"{'Actual':>11}"
    )

    print(header)
    print("-" * len(header))

    for category in order:
        subset = pair_frame.loc[
            pair_frame["disagreement"]
            == category
        ]

        if subset.empty:
            continue

        print(
            f"{category:<18}"
            f"{subset['personal_gap'].mean():>11.4f}"
            f"{subset['quality_gap'].mean():>11.4f}"
            f"{subset['popularity_gap'].mean():>12.4f}"
            f"{subset['mf_gap'].mean():>11.4f}"
            f"{subset['content_gap'].mean():>11.4f}"
            f"{subset['actual_gap'].mean():>11.4f}"
        )

    print()
    print("Confidence by disagreement bucket")
    print("-" * 70)

    print(
        f"{'Outcome':<18}"
        f"{'MF confidence':>18}"
        f"{'Content confidence':>22}"
    )

    print("-" * 58)

    for category in order:
        subset = pair_frame.loc[
            pair_frame["disagreement"]
            == category
        ]

        if subset.empty:
            continue

        print(
            f"{category:<18}"
            f"{subset['mf_confidence'].mean():>18.4f}"
            f"{subset['content_confidence'].mean():>22.4f}"
        )


def print_user_density_analysis(
    pair_frame: pd.DataFrame,
) -> None:
    print_table_header(
        "4. Sparse vs Dense Users"
    )

    users = (
        pair_frame[
            ["user_id", "user_rating_count"]
        ]
        .drop_duplicates()
    )

    low_cutoff = float(
        users["user_rating_count"]
        .quantile(1 / 3)
    )

    high_cutoff = float(
        users["user_rating_count"]
        .quantile(2 / 3)
    )

    working = pair_frame.copy()

    working["bucket"] = working[
        "user_rating_count"
    ].apply(
        lambda value: quantile_bucket(
            value,
            low_cutoff,
            high_cutoff,
        )
    )

    print(
        f"User-history tertiles: "
        f"Low <= {low_cutoff:.0f}, "
        f"Medium <= {high_cutoff:.0f}, "
        f"High > {high_cutoff:.0f}"
    )

    print()

    print(
        f"{'Bucket':<12}"
        f"{'Pairs':>12}"
        f"{'Old ML':>12}"
        f"{'Hybrid':>12}"
        f"{'MF':>12}"
        f"{'Content':>12}"
    )

    print("-" * 72)

    for bucket in ("Low", "Medium", "High"):
        subset = working.loc[
            working["bucket"] == bucket
        ]

        print(
            f"{bucket:<12}"
            f"{len(subset):>12,}"
            f"{format_percent(subset['old_ml_credit'].mean())}"
            f"{format_percent(subset['hybrid_credit'].mean())}"
            f"{format_percent(subset['mf_credit'].mean())}"
            f"{format_percent(subset['content_credit'].mean())}"
        )


def print_movie_popularity_analysis(
    pair_frame: pd.DataFrame,
) -> None:
    print_table_header(
        "5. Long-Tail vs Popular Movies"
    )

    low_cutoff = float(
        pair_frame["movie_rating_count"]
        .quantile(1 / 3)
    )

    high_cutoff = float(
        pair_frame["movie_rating_count"]
        .quantile(2 / 3)
    )

    working = pair_frame.copy()

    working["bucket"] = working[
        "movie_rating_count"
    ].apply(
        lambda value: quantile_bucket(
            value,
            low_cutoff,
            high_cutoff,
        )
    )

    print(
        f"Pair-average training-count tertiles: "
        f"Low <= {low_cutoff:.1f}, "
        f"Medium <= {high_cutoff:.1f}, "
        f"High > {high_cutoff:.1f}"
    )

    print()

    print(
        f"{'Bucket':<12}"
        f"{'Pairs':>12}"
        f"{'Old ML':>12}"
        f"{'Hybrid':>12}"
        f"{'MF':>12}"
        f"{'Content':>12}"
    )

    print("-" * 72)

    for bucket in ("Low", "Medium", "High"):
        subset = working.loc[
            working["bucket"] == bucket
        ]

        print(
            f"{bucket:<12}"
            f"{len(subset):>12,}"
            f"{format_percent(subset['old_ml_credit'].mean())}"
            f"{format_percent(subset['hybrid_credit'].mean())}"
            f"{format_percent(subset['mf_credit'].mean())}"
            f"{format_percent(subset['content_credit'].mean())}"
        )


def print_confidence_analysis(
    pair_frame: pd.DataFrame,
) -> None:
    print_table_header(
        "6. Confidence Analysis"
    )

    for confidence_name, label in (
        ("mf_confidence", "MF"),
        ("content_confidence", "Content"),
    ):
        low_cutoff = float(
            pair_frame[confidence_name]
            .quantile(1 / 3)
        )

        high_cutoff = float(
            pair_frame[confidence_name]
            .quantile(2 / 3)
        )

        working = pair_frame.copy()

        working["bucket"] = working[
            confidence_name
        ].apply(
            lambda value: quantile_bucket(
                value,
                low_cutoff,
                high_cutoff,
            )
        )

        print()
        print(
            f"{label} confidence tertiles: "
            f"Low <= {low_cutoff:.4f}, "
            f"Medium <= {high_cutoff:.4f}, "
            f"High > {high_cutoff:.4f}"
        )

        print(
            f"{'Bucket':<12}"
            f"{'Pairs':>12}"
            f"{'Old ML':>12}"
            f"{'Hybrid':>12}"
            f"{'MF':>12}"
            f"{'Content':>12}"
        )

        print("-" * 72)

        for bucket in (
            "Low",
            "Medium",
            "High",
        ):
            subset = working.loc[
                working["bucket"] == bucket
            ]

            print(
                f"{bucket:<12}"
                f"{len(subset):>12,}"
                f"{format_percent(subset['old_ml_credit'].mean())}"
                f"{format_percent(subset['hybrid_credit'].mean())}"
                f"{format_percent(subset['mf_credit'].mean())}"
                f"{format_percent(subset['content_credit'].mean())}"
            )


def print_rating_gap_analysis(
    pair_frame: pd.DataFrame,
) -> None:
    print_table_header(
        "7. Actual Rating-Gap Difficulty"
    )

    def rating_gap_bucket(
        gap: float,
    ) -> str:
        if gap <= 0.5:
            return "0.5"
        if gap <= 1.0:
            return "1.0"
        if gap <= 1.5:
            return "1.5"
        return "2.0+"

    working = pair_frame.copy()

    working["bucket"] = working[
        "actual_gap"
    ].apply(rating_gap_bucket)

    print(
        f"{'Rating gap':<12}"
        f"{'Pairs':>12}"
        f"{'Old ML':>12}"
        f"{'Hybrid':>12}"
        f"{'MF':>12}"
        f"{'Content':>12}"
    )

    print("-" * 72)

    for bucket in (
        "0.5",
        "1.0",
        "1.5",
        "2.0+",
    ):
        subset = working.loc[
            working["bucket"] == bucket
        ]

        print(
            f"{bucket:<12}"
            f"{len(subset):>12,}"
            f"{format_percent(subset['old_ml_credit'].mean())}"
            f"{format_percent(subset['hybrid_credit'].mean())}"
            f"{format_percent(subset['mf_credit'].mean())}"
            f"{format_percent(subset['content_credit'].mean())}"
        )


def print_complementarity(
    pair_frame: pd.DataFrame,
) -> None:
    print_table_header(
        "8. Conditional Signal Complementarity"
    )

    print(
        "Rescue rate = how often Signal B is correct "
        "when Signal A is wrong."
    )
    print(
        "Higher rescue rates indicate that B contains useful "
        "ranking information not captured by A."
    )

    print()

    print(
        f"{'Signal A wrong':<18}"
        f"{'Signal B':<14}"
        f"{'A-wrong pairs':>15}"
        f"{'B rescues':>14}"
        f"{'Rescue rate':>14}"
    )
    print("-" * 75)

    rows = []

    for signal_a in SIGNALS:
        a_credit = pair_frame[f"{signal_a}_credit"]

        # Restrict this diagnostic to genuine mistakes.
        wrong_mask = a_credit == 0.0
        wrong_count = int(wrong_mask.sum())

        if wrong_count == 0:
            continue

        for signal_b in SIGNALS:
            if signal_a == signal_b:
                continue

            b_credit = pair_frame.loc[
                wrong_mask,
                f"{signal_b}_credit",
            ]

            rescue_count = int(
                (b_credit == 1.0).sum()
            )

            rescue_rate = (
                rescue_count / wrong_count
            )

            rows.append(
                (
                    signal_a,
                    signal_b,
                    wrong_count,
                    rescue_count,
                    rescue_rate,
                )
            )

    for (
        signal_a,
        signal_b,
        wrong_count,
        rescue_count,
        rescue_rate,
    ) in rows:
        print(
            f"{signal_a:<18}"
            f"{signal_b:<14}"
            f"{wrong_count:>15,}"
            f"{rescue_count:>14,}"
            f"{rescue_rate:>13.3%}"
        )

    print()
    print("Best rescue signal for each failed signal")
    print("-" * 75)

    for signal_a in SIGNALS:
        candidates = [
            row
            for row in rows
            if row[0] == signal_a
        ]

        if not candidates:
            continue

        best = max(
            candidates,
            key=lambda row: row[4],
        )

        _, signal_b, wrong_count, rescue_count, rescue_rate = best

        print(
            f"When {signal_a:<10} is wrong -> "
            f"{signal_b:<10} rescues "
            f"{rescue_count:,}/{wrong_count:,} "
            f"({rescue_rate:.3%})"
        )

def print_summary(
    movie_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
) -> None:
    print_table_header(
        "9. Diagnostic Summary"
    )

    old_accuracy = float(
        pair_frame["old_ml_credit"].mean()
    )

    hybrid_accuracy = float(
        pair_frame["hybrid_credit"].mean()
    )

    old_only = int(
        (
            pair_frame["disagreement"]
            == "Old ML only"
        ).sum()
    )

    hybrid_only = int(
        (
            pair_frame["disagreement"]
            == "Hybrid only"
        ).sum()
    )

    print(
        f"Movie predictions retained: "
        f"{len(movie_frame):,}"
    )

    print(
        f"Held-out pairs retained:     "
        f"{len(pair_frame):,}"
    )

    print(
        f"Pair-weighted Old ML:        "
        f"{old_accuracy:.3%}"
    )

    print(
        f"Pair-weighted Hybrid V1:     "
        f"{hybrid_accuracy:.3%}"
    )

    print(
        f"Difference:                  "
        f"{(hybrid_accuracy - old_accuracy) * 100:+.3f} pp"
    )

    print()

    print(
        f"Pairs won only by Old ML:    "
        f"{old_only:,}"
    )

    print(
        f"Pairs won only by Hybrid:    "
        f"{hybrid_only:,}"
    )

    print(
        f"Net Old ML advantage:        "
        f"{old_only - hybrid_only:+,} pairs"
    )


def main() -> None:
    ratings, movies = load_movielens("data")

    print("Week 5 Comprehensive Error Analysis")
    print("=" * 84)

    print(f"Ratings:      {len(ratings):,}")
    print(
        f"Users:        "
        f"{ratings['userId'].nunique():,}"
    )
    print(
        f"Rated movies: "
        f"{ratings['movieId'].nunique():,}"
    )
    print(f"Movie rows:   {len(movies):,}")

    print()
    print(
        "Building predictions and held-out pair diagnostics..."
    )
    print()

    movie_rows, pair_rows = build_evaluation_data(
        ratings,
        movies,
    )

    if not movie_rows:
        raise ValueError(
            "No movie-level diagnostics were generated"
        )

    if not pair_rows:
        raise ValueError(
            "No pair-level diagnostics were generated"
        )

    movie_frame = pd.DataFrame(movie_rows)
    pair_frame = pd.DataFrame(pair_rows)

    print_signal_correlations(movie_frame)
    print_signal_agreement(pair_frame)
    print_disagreement_analysis(pair_frame)
    print_user_density_analysis(pair_frame)
    print_movie_popularity_analysis(pair_frame)
    print_confidence_analysis(pair_frame)
    print_rating_gap_analysis(pair_frame)
    print_complementarity(pair_frame)
    print_summary(movie_frame, pair_frame)


if __name__ == "__main__":
    main()