from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.content.errors import UnknownMovieError
from src.content.profiles import UserTasteProfile, build_profile
from src.content.reliability import ProfileConfig
from src.content.schemas import MovieMetadata, PredictionResult, UserRating, _validate_identifier
from src.content.scoring import ScoringConfig, predict_one


class ContentModel:
    def __init__(
        self,
        ratings: Iterable[UserRating],
        movies: Iterable[MovieMetadata],
        config: ScoringConfig | None = None,
        profile_config: ProfileConfig | None = None,
    ) -> None:
        self._ratings = tuple(ratings)
        self._movies_by_id: dict[int, MovieMetadata] = {}
        self._ratings_by_user: dict[int, tuple[UserRating, ...]] = {}
        self._profiles: dict[int, UserTasteProfile] = {}
        self.config = config or ScoringConfig()
        self._profile_config = profile_config or ProfileConfig()

        for movie in movies:
            if movie.movie_id in self._movies_by_id:
                raise ValueError(f"Duplicate movie ID: {movie.movie_id}")
            self._movies_by_id[movie.movie_id] = movie

        grouped_ratings: dict[int, list[UserRating]] = {}
        for rating in self._ratings:
            grouped_ratings.setdefault(rating.user_id, []).append(rating)
        self._ratings_by_user = {
            user_id: tuple(user_ratings)
            for user_id, user_ratings in grouped_ratings.items()
        }

    def build_profile(self, user_id: int) -> UserTasteProfile:
        _validate_identifier(user_id, "user_id")
        cached = self._profiles.get(user_id)
        if cached is not None:
            return cached

        user_ratings = self._ratings_by_user.get(user_id, ())

        profile = build_profile(
            user_id=user_id,
            ratings=user_ratings,
            movies=self._movies_by_id,
            config=self._profile_config,
        )
        self._profiles[user_id] = profile
        return profile

    def predict_one(
        self,
        user_id: int,
        movie_id: int,
        include_debug: bool = False,
    ) -> PredictionResult:
        movie = self._movies_by_id.get(movie_id)
        if movie is None:
            raise UnknownMovieError(movie_id)

        profile = self.build_profile(user_id)
        return predict_one(
            profile=profile,
            movie=movie,
            config=self.config,
            include_debug=include_debug,
        )

    def predict(
        self,
        user_ids: Sequence[int],
        movie_ids: Sequence[int],
        include_debug: bool = False,
    ) -> tuple[PredictionResult, ...]:
        ordered_user_ids = tuple(user_ids)
        ordered_movie_ids = tuple(movie_ids)

        for user_id in ordered_user_ids:
            _validate_identifier(user_id, "user_id")

        for movie_id in ordered_movie_ids:
            if movie_id not in self._movies_by_id:
                raise UnknownMovieError(movie_id)

        return tuple(
            self.predict_one(user_id, movie_id, include_debug)
            for user_id in ordered_user_ids
            for movie_id in ordered_movie_ids
        )

    def unseen_movie_ids(self, user_id: int) -> tuple[int, ...]:
        _validate_identifier(user_id, "user_id")
        user_ratings = self._ratings_by_user.get(user_id, ())

        rated_movie_ids = {rating.movie_id for rating in user_ratings}
        return tuple(
            movie_id
            for movie_id in sorted(self._movies_by_id)
            if movie_id not in rated_movie_ids
        )

    def predict_unseen(
        self,
        user_ids: Sequence[int],
        limit: int | None = None,
        include_debug: bool = False,
    ) -> tuple[PredictionResult, ...]:
        if limit is not None and (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 0
        ):
            raise ValueError("limit must be a non-negative integer or None")

        results: list[PredictionResult] = []
        for user_id in tuple(user_ids):
            movie_ids = self.unseen_movie_ids(user_id)
            selected_movie_ids = movie_ids if limit is None else movie_ids[:limit]
            results.extend(
                self.predict_one(user_id, movie_id, include_debug)
                for movie_id in selected_movie_ids
            )

        return tuple(results)
