from dataclasses import dataclass
from pathlib import Path
from xml.parsers.expat import model

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from src.collaborative.matrix_factorization import (
    BiasedMatrixFactorization,
)
from src.content.profiles import ProfileConfig
from src.content.scoring import ScoringConfig
from src.evaluation.splits import build_user_evaluation_split
from src.hybrid.content_adapter import (
    build_content_model_from_frames,
)
from src.hybrid.genre_recommender import (
    score_movies_by_genre,
)
from src.hybrid.ml_reranker import (
    calculate_movie_popularity,
)


GBT_FEATURES = [
    "collaborative",
    "personal",
    "content",
    "content_confidence",
    "quality",
    "popularity",
]


LINEAR_FEATURES = [
    "collaborative",
    "personal",
    "content",
    "quality",
    "popularity",
]


@dataclass(frozen=True)
class GBTMovieFeatures:
    collaborative_score: float
    personal_score: float
    content_score: float
    content_confidence: float
    quality_score: float
    popularity: float

    def as_array(
        self,
        features: list[str] | None = None,
    ) -> np.ndarray:
        if features is None:
            features = GBT_FEATURES

        values = {
            "collaborative": self.collaborative_score,
            "personal": self.personal_score,
            "content": self.content_score,
            "content_confidence": self.content_confidence,
            "quality": self.quality_score,
            "popularity": self.popularity,
        }

        return np.array(
            [
                values[feature]
                for feature in features
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class GBTExample:
    user_id: int
    movie_id: int
    features: GBTMovieFeatures
    actual_rating: float


@dataclass(frozen=True)
class UserGBTSplit:
    profile: pd.DataFrame
    development_train: pd.DataFrame
    validation: pd.DataFrame
    middle: pd.DataFrame
    test: pd.DataFrame


@dataclass
class TrainedGBTRanker:
    model: GradientBoostingRegressor
    feature_names: list[str]


def split_user_for_gbt(
    user_ratings: pd.DataFrame,
) -> UserGBTSplit:
    profile, middle, test = (
        build_user_evaluation_split(
            user_ratings
        )
    )

    middle = (
        middle
        .sort_values(
            ["timestamp", "movieId"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if len(middle) < 2:
        development_train = (
            middle.copy()
        )
        validation = middle.iloc[0:0].copy()

    else:
        cut = len(middle) // 2

        if cut == 0:
            cut = 1

        development_train = (
            middle.iloc[:cut].copy()
        )

        validation = (
            middle.iloc[cut:].copy()
        )

    return UserGBTSplit(
        profile=profile,
        development_train=development_train,
        validation=validation,
        middle=middle,
        test=test,
    )


def build_gbt_user_splits(
    ratings: pd.DataFrame,
) -> tuple[
    dict[int, UserGBTSplit],
    pd.DataFrame,
]:
    user_splits = {}
    profile_parts = []

    for raw_user_id in ratings[
        "userId"
    ].unique():
        user_id = int(raw_user_id)

        user_ratings = ratings.loc[
            ratings["userId"] == user_id
        ].copy()

        if len(user_ratings) < 10:
            continue

        split = split_user_for_gbt(
            user_ratings
        )

        user_splits[user_id] = split
        profile_parts.append(split.profile)

    if not profile_parts:
        raise ValueError(
            "No profile ratings were generated"
        )

    global_profile_ratings = pd.concat(
        profile_parts,
        ignore_index=True,
    )

    return (
        user_splits,
        global_profile_ratings,
    )


def fit_gbt_collaborative_model(
    global_profile_ratings: pd.DataFrame,
) -> BiasedMatrixFactorization:
    model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=40,
        prior_strength=5.0,
        random_state=42,
        shrink_latent=True,
    )

    model.fit(global_profile_ratings)

    return model


def build_user_gbt_examples(
    user_id: int,
    profile_ratings: pd.DataFrame,
    target_ratings: pd.DataFrame,
    global_profile_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    collaborative_model: BiasedMatrixFactorization,
    scoring_config: ScoringConfig | None = None,
    profile_config: ProfileConfig | None = None,
) -> list[GBTExample]:
    if target_ratings.empty:
        return []

    movie_ids = (
        target_ratings["movieId"]
        .astype(int)
        .tolist()
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
        user_history=profile_ratings,
        reference_ratings=reference_ratings,
        movies=movies,
        movie_ids=movie_ids,
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

    collaborative_predictions = (
        collaborative_model.predict(
            user_ids=[
                user_id
            ] * len(movie_ids),
            movie_ids=movie_ids,
        )
        .set_index("movie_id")[
            "predicted_score"
        ]
        .to_dict()
    )

    content_model = (
        build_content_model_from_frames(
            profile_ratings=profile_ratings,
            movies=movies,
            scoring_config=scoring_config,
            profile_config=profile_config,
        )
    )

    examples = []

    for movie_id in movie_ids:
        if movie_id not in popularity:
            continue

        if movie_id not in personal_by_movie:
            continue

        if movie_id not in quality_by_movie:
            continue

        if (
            movie_id
            not in collaborative_predictions
        ):
            continue

        rating_rows = target_ratings.loc[
            target_ratings["movieId"]
            == movie_id,
            "rating",
        ]

        if rating_rows.empty:
            continue

        content_prediction = (
            content_model.predict_one(
                user_id=user_id,
                movie_id=movie_id,
            )
        )

        features = GBTMovieFeatures(
            collaborative_score=float(
                collaborative_predictions[
                    movie_id
                ]
            ),
            personal_score=float(
                personal_by_movie[movie_id]
            ),
            content_score=float(
                content_prediction.predicted_score
            ),
            content_confidence=float(
                content_prediction.confidence
            ),
            quality_score=float(
                quality_by_movie[movie_id]
            ),
            popularity=float(
                popularity[movie_id]
            ),
        )

        examples.append(
            GBTExample(
                user_id=user_id,
                movie_id=movie_id,
                features=features,
                actual_rating=float(
                    rating_rows.iloc[0]
                ),
            )
        )

    return examples


def examples_to_pointwise_dataset(
    examples: list[GBTExample],
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError(
            "No GBT examples were generated"
        )

    if features is None:
        features = GBT_FEATURES

    X = np.vstack([
        example.features.as_array(
            features
        )
        for example in examples
    ])

    y = np.array(
        [
            example.actual_rating
            for example in examples
        ],
        dtype=float,
    )

    return X, y


def examples_to_pairwise_dataset(
    examples_by_user: dict[
        int,
        list[GBTExample],
    ],
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if features is None:
        features = LINEAR_FEATURES

    rows = []
    labels = []

    for examples in examples_by_user.values():
        for i in range(len(examples)):
            for j in range(
                i + 1,
                len(examples),
            ):
                first = examples[i]
                second = examples[j]

                if (
                    first.actual_rating
                    == second.actual_rating
                ):
                    continue

                first_array = (
                    first.features.as_array(
                        features
                    )
                )

                second_array = (
                    second.features.as_array(
                        features
                    )
                )

                if (
                    first.actual_rating
                    > second.actual_rating
                ):
                    difference = (
                        first_array
                        - second_array
                    )
                else:
                    difference = (
                        second_array
                        - first_array
                    )

                rows.append(difference)
                labels.append(1)

                rows.append(-difference)
                labels.append(0)

    if not rows:
        raise ValueError(
            "No pairwise examples were generated"
        )

    return (
        np.vstack(rows),
        np.array(labels, dtype=int),
    )


def train_gbt_ranker(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
) -> TrainedGBTRanker:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=20,
        random_state=42,
    )

    model.fit(X, y)

    return TrainedGBTRanker(
        model=model,
        feature_names=list(feature_names),
    )


def predict_gbt_score(
    ranker: TrainedGBTRanker,
    features: GBTMovieFeatures,
) -> float:
    raw = (
        features
        .as_array(
            ranker.feature_names
        )
        .reshape(1, -1)
    )

    return float(
        ranker.model.predict(raw)[0]
    )


def save_gbt_ranker(
    ranker: TrainedGBTRanker,
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        ranker,
        path,
    )


def load_gbt_ranker(
    path: str | Path,
) -> TrainedGBTRanker:
    return joblib.load(path)