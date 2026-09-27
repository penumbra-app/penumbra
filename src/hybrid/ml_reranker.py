from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.hybrid.genre_recommender import score_movies_by_genre
from src.collaborative.matrix_factorization import BiasedMatrixFactorization
from src.hybrid.content_adapter import build_content_model_from_frames

@dataclass(frozen=True)
class HybridCandidate:
    movie_id: int

    content_score: float
    collaborative_score: float
    quality_score: float
    popularity: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.content_score,
                self.collaborative_score,
                self.quality_score,
                self.popularity,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class AblationCandidate:
    movie_id: int
    collaborative_score: float
    personal_score: float
    content_score: float
    quality_score: float
    popularity: float

    def as_array(self, features: list[str]) -> np.ndarray:
        values = {
            "collaborative": self.collaborative_score,
            "personal": self.personal_score,
            "content": self.content_score,
            "quality": self.quality_score,
            "popularity": self.popularity,
        }

        return np.array(
            [values[feature] for feature in features],
            dtype=float,
        )

@dataclass
class TrainedRanker:
    scaler: StandardScaler
    model: LogisticRegression


@dataclass(frozen=True)
class MovieFeatures:
    personal_score: float
    quality_score: float
    popularity: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.personal_score,
                self.quality_score,
                self.popularity,
            ],
            dtype=float,
        )
    
def calculate_movie_popularity(
    ratings: pd.DataFrame,
) -> dict[int, float]:
    rating_counts = ratings.groupby("movieId").size() # takes number of ratings for a movie

    return {
        int(movie_id): float(np.log1p(count)) # ln(1 + no. of ratings)
        for movie_id, count in rating_counts.items()
    }

def create_pairwise_example(
    first: MovieFeatures,
    second: MovieFeatures,
    first_rating: float,
    second_rating: float,
) -> tuple[np.ndarray, int]:
    if first_rating == second_rating:
        raise ValueError("Pair must contain different ratings")

    difference = first.as_array() - second.as_array()

    label = 1 if first_rating > second_rating else 0 # categorizing success vs failure

    return difference, label

def chronological_split(
    user_ratings: pd.DataFrame,
    profile_fraction: float = 0.60,
    train_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= profile_fraction <= 1.0:
        raise ValueError("profile_fraction must be between 0 and 1")

    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    if profile_fraction + train_fraction > 1.0:
        raise ValueError(
            "profile_fraction + train_fraction cannot exceed 1"
        )

    ordered = (
        user_ratings
        .sort_values(["timestamp", "movieId"], kind="stable")
        .reset_index(drop=True)
    )

    total = len(ordered)

    profile_end = int(total * profile_fraction)
    train_end = int(total * (profile_fraction + train_fraction))

    profile = ordered.iloc[:profile_end].copy()
    train = ordered.iloc[profile_end:train_end].copy()
    test = ordered.iloc[train_end:].copy()

    return profile, train, test

def build_pairwise_examples(
    movie_features: dict[int, MovieFeatures],
    actual_ratings: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    movie_ids = list(movie_features.keys())

    for i in range(len(movie_ids)):
        for j in range(i + 1, len(movie_ids)):
            first_id = movie_ids[i]
            second_id = movie_ids[j]

            first_rating = actual_ratings[first_id]
            second_rating = actual_ratings[second_id]

            if first_rating == second_rating:
                continue

            first_features = movie_features[first_id].as_array()
            second_features = movie_features[second_id].as_array()

            if first_rating > second_rating:
                preferred = first_features
                less_preferred = second_features
            else:
                preferred = second_features
                less_preferred = first_features

            difference = preferred - less_preferred

            feature_rows.append(difference)
            labels.append(1)

            feature_rows.append(-difference)
            labels.append(0)

    return (
        np.vstack(feature_rows),
        np.array(labels, dtype=int),
    )

def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
) -> TrainedRanker:
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    return TrainedRanker(
        scaler=scaler,
        model=model,
    )

def build_hybrid_pairwise_examples(
    candidates: dict[int, HybridCandidate],
    actual_ratings: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    movie_ids = list(candidates.keys())

    for i in range(len(movie_ids)):
        for j in range(i + 1, len(movie_ids)):
            first_id = movie_ids[i]
            second_id = movie_ids[j]

            first_rating = actual_ratings[first_id]
            second_rating = actual_ratings[second_id]

            if first_rating == second_rating:
                continue

            first_features = candidates[first_id].as_array()
            second_features = candidates[second_id].as_array()

            if first_rating > second_rating:
                difference = first_features - second_features
            else:
                difference = second_features - first_features

            feature_rows.append(difference)
            labels.append(1)

            feature_rows.append(-difference)
            labels.append(0)

    return (
        np.vstack(feature_rows),
        np.array(labels, dtype=int),
    )


def build_ablation_pairwise_examples(
    candidates: dict[int, AblationCandidate],
    actual_ratings: dict[int, float],
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = []
    labels = []

    movie_ids = list(candidates.keys())

    for i in range(len(movie_ids)):
        for j in range(i + 1, len(movie_ids)):
            first_id = movie_ids[i]
            second_id = movie_ids[j]

            first_rating = actual_ratings[first_id]
            second_rating = actual_ratings[second_id]

            if first_rating == second_rating:
                continue

            first_features = candidates[first_id].as_array(features)
            second_features = candidates[second_id].as_array(features)

            if first_rating > second_rating:
                difference = first_features - second_features
            else:
                difference = second_features - first_features

            feature_rows.append(difference)
            labels.append(1)

            feature_rows.append(-difference)
            labels.append(0)

    if not feature_rows:
        return (
            np.empty((0, len(features)), dtype=float),
            np.empty((0,), dtype=int),
        )

    return (
        np.vstack(feature_rows),
        np.array(labels, dtype=int),
    )

def build_user_training_examples(
    user_id: int,
    profile_ratings: pd.DataFrame,
    pairwise_ratings: pd.DataFrame,
    reference_ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(pairwise_ratings) < 2:
        return None

    train_movie_ids = (
        pairwise_ratings["movieId"]
        .astype(int)
        .tolist()
    )

    popularity = calculate_movie_popularity(
        reference_ratings
    )

    scored_movies = score_movies_by_genre(
        user_id=user_id,
        user_history=profile_ratings,
        reference_ratings=reference_ratings,
        movies=movies,
        movie_ids=train_movie_ids,
    )

    movie_features: dict[int, MovieFeatures] = {}
    actual_ratings: dict[int, float] = {}

    for movie in scored_movies:
        movie_id = movie.movie_id

        if movie_id not in popularity:
            continue

        rating_rows = pairwise_ratings.loc[
            pairwise_ratings["movieId"] == movie_id,
            "rating",
        ]

        if rating_rows.empty:
            continue

        movie_features[movie_id] = MovieFeatures(
            personal_score=movie.personal_score,
            quality_score=movie.quality_score,
            popularity=popularity[movie_id],
        )

        actual_ratings[movie_id] = float(
            rating_rows.iloc[0]
        )

    if len(movie_features) < 2:
        return None

    if len(set(actual_ratings.values())) < 2:
        return None

    return build_pairwise_examples(
        movie_features=movie_features,
        actual_ratings=actual_ratings,
    )

def build_user_hybrid_training_examples(
    user_id: int,
    profile_ratings: pd.DataFrame,
    pairwise_ratings: pd.DataFrame,
    reference_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    content_model,
    collaborative_model,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(pairwise_ratings) < 2:
        return None

    movie_ids = (
        pairwise_ratings["movieId"]
        .astype(int)
        .tolist()
    )

    popularity = calculate_movie_popularity(
        reference_ratings
    )

    # We still use the existing scorer for the population-level
    # quality_score, NOT for the new content_score.
    scored_movies = score_movies_by_genre(
        user_id=user_id,
        user_history=profile_ratings,
        reference_ratings=reference_ratings,
        movies=movies,
        movie_ids=movie_ids,
    )

    quality_by_movie = {
        movie.movie_id: movie.quality_score
        for movie in scored_movies
    }

    candidates: dict[int, HybridCandidate] = {}
    actual_ratings: dict[int, float] = {}

    for movie_id in movie_ids:
        if movie_id not in popularity:
            continue

        if movie_id not in quality_by_movie:
            continue

        rating_rows = pairwise_ratings.loc[
            pairwise_ratings["movieId"] == movie_id,
            "rating",
        ]

        if rating_rows.empty:
            continue

        candidate = build_hybrid_candidate(
            user_id=user_id,
            movie_id=movie_id,
            content_model=content_model,
            collaborative_model=collaborative_model,
            quality_score=quality_by_movie[movie_id],
            popularity=popularity[movie_id],
        )

        candidates[movie_id] = candidate

        actual_ratings[movie_id] = float(
            rating_rows.iloc[0]
        )

    if len(candidates) < 2:
        return None

    if len(set(actual_ratings.values())) < 2:
        return None

    return build_hybrid_pairwise_examples(
        candidates=candidates,
        actual_ratings=actual_ratings,
    )

def build_user_ablation_training_examples(
    user_id: int,
    profile_ratings: pd.DataFrame,
    pairwise_ratings: pd.DataFrame,
    reference_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    content_model,
    collaborative_model,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(pairwise_ratings) < 2:
        return None

    movie_ids = pairwise_ratings["movieId"].astype(int).tolist()

    popularity = calculate_movie_popularity(reference_ratings)

    scored_movies = score_movies_by_genre(
        user_id=user_id,
        user_history=profile_ratings,
        reference_ratings=reference_ratings,
        movies=movies,
        movie_ids=movie_ids,
    )

    personal_by_movie = {
        movie.movie_id: movie.personal_score
        for movie in scored_movies
    }

    quality_by_movie = {
        movie.movie_id: movie.quality_score
        for movie in scored_movies
    }

    candidates = {}
    actual_ratings = {}

    for movie_id in movie_ids:
        if movie_id not in popularity:
            continue

        if movie_id not in personal_by_movie:
            continue

        if movie_id not in quality_by_movie:
            continue

        rating_rows = pairwise_ratings.loc[
            pairwise_ratings["movieId"] == movie_id,
            "rating",
        ]

        if rating_rows.empty:
            continue

        candidate = build_ablation_candidate(
            user_id=user_id,
            movie_id=movie_id,
            content_model=content_model,
            collaborative_model=collaborative_model,
            personal_score=personal_by_movie[movie_id],
            quality_score=quality_by_movie[movie_id],
            popularity=popularity[movie_id],
        )

        candidates[movie_id] = candidate
        actual_ratings[movie_id] = float(rating_rows.iloc[0])

    if len(candidates) < 2:
        return None

    if len(set(actual_ratings.values())) < 2:
        return None

    return build_ablation_pairwise_examples(
        candidates=candidates,
        actual_ratings=actual_ratings,
        features=features,
    )

def build_training_dataset(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    user_splits: dict[
        int,
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}

    reference_parts: list[pd.DataFrame] = []

    for raw_user_id in ratings["userId"].unique():
        user_id = int(raw_user_id)

        user_ratings = ratings.loc[
            ratings["userId"] == user_id
        ].copy()

        if len(user_ratings) < 10:
            # These users are not used for pairwise ML training,
            # so they have no held-out test partition here.
            reference_parts.append(user_ratings)
            continue

        profile, train, _ = chronological_split(
            user_ratings
        )

        user_splits[user_id] = (
            profile,
            train,
        )

        reference_parts.append(profile)
        reference_parts.append(train)

    if not reference_parts:
        raise ValueError(
            "No leakage-safe reference ratings were generated"
        )

    global_reference_ratings = pd.concat(
        reference_parts,
        ignore_index=True,
    )

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    users_used = 0

    for user_id, (profile, train) in user_splits.items():
        # Do not let this user's own pairwise-training ratings
        # influence population-level quality/popularity features.
        user_reference_ratings = (
            global_reference_ratings.loc[
                global_reference_ratings["userId"]
                != user_id
            ]
        )

        result = build_user_training_examples(
            user_id=user_id,
            profile_ratings=profile,
            pairwise_ratings=train,
            reference_ratings=user_reference_ratings,
            movies=movies,
        )

        if result is None:
            continue

        X_user, y_user = result

        all_features.append(X_user)
        all_labels.append(y_user)

        users_used += 1

    if not all_features:
        raise ValueError(
            "No training examples were generated"
        )

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    return X, y, users_used

def build_hybrid_training_dataset(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:

    user_splits: dict[
        int,
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}

    profile_parts: list[pd.DataFrame] = []

    # 1. Split users into 60 / 20 / 20
    for raw_user_id in ratings["userId"].unique():
        user_id = int(raw_user_id)

        user_ratings = ratings.loc[
            ratings["userId"] == user_id
        ].copy()

        if len(user_ratings) < 10:
            continue

        profile, train, _ = chronological_split(
            user_ratings
        )

        user_splits[user_id] = (
            profile,
            train,
        )

        profile_parts.append(profile)

    if not profile_parts:
        raise ValueError(
            "No profile ratings were generated"
        )

    global_profile_ratings = pd.concat(
        profile_parts,
        ignore_index=True,
    )

    # 2. Train collaborative model ONCE using everybody's 60%
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

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    users_used = 0

    # 3. Build hybrid examples for each user
    for user_id, (profile, train) in user_splits.items():

        # Population features should not use this user's own ratings.
        other_user_reference_ratings = (
            global_profile_ratings.loc[
                global_profile_ratings["userId"] != user_id
            ]
            .copy()
        )

        # Content sees only this user's 60% profile.
        content_model = build_content_model_from_frames(
            profile_ratings=profile,
            movies=movies,
        )

        result = build_user_hybrid_training_examples(
            user_id=user_id,
            profile_ratings=profile,
            pairwise_ratings=train,
            reference_ratings=other_user_reference_ratings,
            movies=movies,
            content_model=content_model,
            collaborative_model=collaborative_model,
        )

        if result is None:
            continue

        X_user, y_user = result

        all_features.append(X_user)
        all_labels.append(y_user)

        users_used += 1

    if not all_features:
        raise ValueError(
            "No hybrid training examples were generated"
        )

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    return X, y, users_used

def build_ablation_training_dataset(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    user_splits: dict[
        int,
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}

    profile_parts: list[pd.DataFrame] = []

    for raw_user_id in ratings["userId"].unique():
        user_id = int(raw_user_id)

        user_ratings = ratings.loc[
            ratings["userId"] == user_id
        ].copy()

        if len(user_ratings) < 10:
            continue

        profile, train, _ = chronological_split(user_ratings)

        user_splits[user_id] = (profile, train)
        profile_parts.append(profile)

    if not profile_parts:
        raise ValueError("No profile ratings were generated")

    global_profile_ratings = pd.concat(
        profile_parts,
        ignore_index=True,
    )

    collaborative_model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=40,
        prior_strength=5.0,
        random_state=42,
        shrink_latent=True,
    )

    collaborative_model.fit(global_profile_ratings)

    all_features = []
    all_labels = []
    users_used = 0

    for user_id, (profile, train) in user_splits.items():
        reference_ratings = global_profile_ratings.loc[
            global_profile_ratings["userId"] != user_id
        ].copy()

        content_model = build_content_model_from_frames(
            profile_ratings=profile,
            movies=movies,
        )

        result = build_user_ablation_training_examples(
            user_id=user_id,
            profile_ratings=profile,
            pairwise_ratings=train,
            reference_ratings=reference_ratings,
            movies=movies,
            content_model=content_model,
            collaborative_model=collaborative_model,
            features=features,
        )

        if result is None:
            continue

        X_user, y_user = result

        if len(y_user) == 0:
            continue

        all_features.append(X_user)
        all_labels.append(y_user)
        users_used += 1

    if not all_features:
        raise ValueError(
            "No ablation training examples were generated"
        )

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    return X, y, users_used

def score_with_ranker(
    features: MovieFeatures,
    ranker: TrainedRanker,
) -> float:
    X = features.as_array().reshape(1, -1)

    X_scaled = ranker.scaler.transform(X)

    score = ranker.model.decision_function(X_scaled)[0]

    return float(score)

def rerank_candidates_with_ml(
    candidates,
    popularity: dict[int, float],
    ranker: TrainedRanker,
):
    scored = []

    for movie in candidates:
        features = MovieFeatures(
            personal_score=movie.personal_score,
            quality_score=movie.quality_score,
            popularity=popularity[movie.movie_id],
        )

        ml_score = score_with_ranker(
            features=features,
            ranker=ranker,
        )

        scored.append((movie, ml_score))

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored

def save_ranker(
    ranker: TrainedRanker,
    path: str = "src/models/ml_reranker.joblib"
) -> None:
    joblib.dump(ranker, path)


def load_ranker(
    path: str = "src/models/ml_reranker.joblib"
) -> TrainedRanker:
    return joblib.load(path)

@dataclass(frozen=True)
class MLRerankedRecommendation:
    movie_id: int
    title: str
    genres: str
    ml_score: float
    personal_score: float
    quality_score: float
    popularity: float


def recommend_with_ml(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    limit: int = 5,
) -> list[MLRerankedRecommendation]:
    ranker = load_ranker()

    user_history = ratings.loc[
        ratings["userId"] == user_id
    ].copy()

    if user_history.empty:
        raise ValueError(f"User {user_id} has no ratings")

    watched_ids = set(
        user_history["movieId"].astype(int)
    )

    unseen_movie_ids = (
        movies.loc[
            ~movies["movieId"].isin(watched_ids),
            "movieId",
        ]
        .astype(int)
        .tolist()
    )

    scored_movies = score_movies_by_genre(
        user_id=user_id,
        user_history=user_history,
        reference_ratings=ratings,
        movies=movies,
        movie_ids=unseen_movie_ids,
    )

    popularity = calculate_movie_popularity(ratings)

    recommendations: list[MLRerankedRecommendation] = []

    for movie in scored_movies:
        features = MovieFeatures(
            personal_score=movie.personal_score,
            quality_score=movie.quality_score,
            popularity=popularity[movie.movie_id],
        )

        raw_features = features.as_array().reshape(1, -1)

        scaled_features = ranker.scaler.transform(
            raw_features
        )

        ml_score = float(
            ranker.model.decision_function(
                scaled_features
            )[0]
        )

        recommendations.append(
            MLRerankedRecommendation(
                movie_id=movie.movie_id,
                title=movie.title,
                genres=movie.genres,
                ml_score=ml_score,
                personal_score=movie.personal_score,
                quality_score=movie.quality_score,
                popularity=popularity[movie.movie_id],
            )
        )

    recommendations.sort(
        key=lambda recommendation: recommendation.ml_score,
        reverse=True,
    )

    return recommendations[:limit]

def build_hybrid_candidate(
    user_id: int,
    movie_id: int,
    content_model,
    collaborative_model,
    quality_score: float,
    popularity: float,
) -> HybridCandidate:
    content_prediction = content_model.predict_one(
        user_id=user_id,
        movie_id=movie_id,
    )

    collaborative_prediction = collaborative_model.predict(
        user_ids=[user_id],
        movie_ids=[movie_id],
    )

    collaborative_score = float(
        collaborative_prediction.iloc[0]["predicted_score"]
    )

    return HybridCandidate(
        movie_id=movie_id,
        content_score=content_prediction.predicted_score,
        collaborative_score=collaborative_score,
        quality_score=quality_score,
        popularity=popularity,
    )

def build_ablation_candidate(
    user_id: int,
    movie_id: int,
    content_model,
    collaborative_model,
    personal_score: float,
    quality_score: float,
    popularity: float,
) -> AblationCandidate:
    content_prediction = content_model.predict_one(
        user_id=user_id,
        movie_id=movie_id,
    )

    collaborative_prediction = collaborative_model.predict(
        user_ids=[user_id],
        movie_ids=[movie_id],
    )

    collaborative_score = float(
        collaborative_prediction.iloc[0]["predicted_score"]
    )

    return AblationCandidate(
        movie_id=movie_id,
        collaborative_score=collaborative_score,
        personal_score=personal_score,
        content_score=content_prediction.predicted_score,
        quality_score=quality_score,
        popularity=popularity,
    )