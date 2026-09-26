from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from src.content.baselines import calculate_user_baseline
from src.content.errors import UnknownUserError
from src.content.features import FeaturePreference, aggregate_feature_preferences
from src.content.genres import GenrePreference, _unique_genres, aggregate_genre_preferences
from src.content.reliability import ProfileConfig, rating_weights
from src.content.schemas import MovieMetadata, UserRating


PROFILE_VERSION = "content-v3"


@dataclass(frozen=True)
class ProfileMetadata:
    ratings_used: int
    movies_with_metadata: int
    movies_with_genres: int
    ratings_without_movie_metadata: int
    ratings_without_genres: int
    reference_timestamp: int | None = None


@dataclass(frozen=True)
class UserTasteProfile:
    user_id: int
    baseline: float
    rating_count: int
    genre_preferences: tuple[GenrePreference, ...]
    profile_version: str
    metadata: ProfileMetadata
    config: ProfileConfig = ProfileConfig()
    feature_preferences: tuple[FeaturePreference, ...] = ()

    def preference_for(self, genre: str) -> float:
        key = genre.strip().casefold()
        for preference in self.genre_preferences:
            if preference.genre.casefold() == key:
                return preference.preference
        return 0.0

    def evidence_for(self, genre: str) -> int:
        key = genre.strip().casefold()
        for preference in self.genre_preferences:
            if preference.genre.casefold() == key:
                return preference.movie_count
        return 0

    def effective_evidence_for(self, genre: str) -> float:
        key = genre.strip().casefold()
        for preference in self.genre_preferences:
            if preference.genre.casefold() == key:
                return preference.evidence_weight
        return 0.0


def _index_movies(
    movies: Mapping[int, MovieMetadata] | Iterable[MovieMetadata],
) -> dict[int, MovieMetadata]:
    if isinstance(movies, Mapping):
        if any(key != movie.movie_id for key, movie in movies.items()):
            raise ValueError("Movie mapping keys must match movie IDs")
        return dict(movies)

    indexed: dict[int, MovieMetadata] = {}
    for movie in movies:
        if movie.movie_id in indexed:
            raise ValueError(f"Duplicate movie ID: {movie.movie_id}")
        indexed[movie.movie_id] = movie
    return indexed


def build_profile(
    user_id: int,
    ratings: Iterable[UserRating],
    movies: Mapping[int, MovieMetadata] | Iterable[MovieMetadata],
    config: ProfileConfig | None = None,
) -> UserTasteProfile:
    user_ratings = tuple(
        rating for rating in ratings if rating.user_id == user_id
    )
    if not user_ratings:
        raise UnknownUserError(user_id)
    if len({rating.movie_id for rating in user_ratings}) != len(user_ratings):
        raise ValueError("Duplicate ratings for a user and movie are not supported")

    active_config = config or ProfileConfig()
    weights, reference = rating_weights(user_ratings, active_config)
    movies_by_id = _index_movies(movies)
    baseline = calculate_user_baseline(user_ratings)
    preferences = aggregate_genre_preferences(
        baseline, movies_by_id, rating_weights=weights,
        regularization_strength=active_config.regularization_strength,
    )
    movies_with_metadata = sum(
        rating.movie_id in movies_by_id for rating in user_ratings
    )
    movies_with_genres = sum(
        bool(_unique_genres(movies_by_id[rating.movie_id].genres))
        for rating in user_ratings
        if rating.movie_id in movies_by_id
    )

    metadata = ProfileMetadata(
        ratings_used=len(user_ratings),
        movies_with_metadata=movies_with_metadata,
        movies_with_genres=movies_with_genres,
        ratings_without_movie_metadata=len(user_ratings) - movies_with_metadata,
        ratings_without_genres=movies_with_metadata - movies_with_genres,
        reference_timestamp=reference,
    )

    return UserTasteProfile(
        user_id=user_id,
        baseline=baseline.mean_rating,
        rating_count=baseline.rating_count,
        genre_preferences=preferences,
        profile_version=PROFILE_VERSION,
        metadata=metadata,
        config=active_config,
        feature_preferences=aggregate_feature_preferences(
            baseline, movies_by_id, weights, active_config.regularization_strength,
        ),
    )
