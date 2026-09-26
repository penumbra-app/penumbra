from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _validate_identifier(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_bounded_number(
    value: float,
    field_name: str,
    minimum: float,
    maximum: float,
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(float(value)) or not minimum <= float(value) <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")


@dataclass(frozen=True)
class UserRating:
    user_id: int
    movie_id: int
    rating: float
    timestamp: int | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.user_id, "user_id")
        _validate_identifier(self.movie_id, "movie_id")
        _validate_bounded_number(self.rating, "rating", 0.0, 5.0)
        if self.timestamp is not None and (
            not isinstance(self.timestamp, int)
            or isinstance(self.timestamp, bool)
            or self.timestamp < 0
        ):
            raise ValueError("timestamp must be a non-negative integer or None")


@dataclass(frozen=True)
class MovieMetadata:
    movie_id: int
    title: str
    genres: tuple[str, ...] = ()
    directors: tuple[str, ...] = ()
    cast: tuple[str, ...] = ()
    runtime_minutes: float | None = None
    release_year: int | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.movie_id, "movie_id")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(self.directors, tuple) or any(
            not isinstance(value, str) for value in self.directors
        ):
            raise ValueError("directors must be a tuple of strings")
        if self.language is not None and not isinstance(self.language, str):
            raise ValueError("language must be a string or None")
        if self.runtime_minutes is not None and (
            not isinstance(self.runtime_minutes, (int, float))
            or isinstance(self.runtime_minutes, bool)
            or not math.isfinite(float(self.runtime_minutes))
            or self.runtime_minutes <= 0
        ):
            raise ValueError("runtime_minutes must be positive or None")
        if self.release_year is not None and (
            not isinstance(self.release_year, int)
            or isinstance(self.release_year, bool)
            or self.release_year <= 0
        ):
            raise ValueError("release_year must be positive or None")


@dataclass(frozen=True)
class ReasonSignal:
    feature_type: str
    feature_value: str
    strength: float
    evidence_count: float

    def __post_init__(self) -> None:
        if not self.feature_type.strip():
            raise ValueError("feature_type must be non-empty")
        if not self.feature_value.strip():
            raise ValueError("feature_value must be non-empty")
        _validate_bounded_number(self.strength, "strength", -1.0, 1.0)
        if not math.isfinite(float(self.evidence_count)) or self.evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")


@dataclass(frozen=True)
class FeatureDebug:
    feature_type: str
    candidate_values: tuple[str, ...]
    matched_values: tuple[str, ...]
    raw_component: float
    bounded_component: float
    weight: float
    weighted_adjustment: float
    support: float


@dataclass(frozen=True)
class PredictionDebug:
    baseline: float
    movie_genres: tuple[str, ...]
    matched_genres: tuple[str, ...]
    unknown_genres: tuple[str, ...]
    raw_genre_component: float
    bounded_genre_component: float
    genre_weight: float
    weighted_genre_adjustment: float
    unclamped_score: float
    was_clamped: bool
    feature_components: tuple[FeatureDebug, ...] = ()
    genre_support: float = 0.0
    effective_genre_evidence: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class PredictionResult:
    user_id: int
    movie_id: int
    predicted_score: float
    confidence: float
    reason_signals: tuple[ReasonSignal, ...] = field(default_factory=tuple)
    debug: PredictionDebug | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.user_id, "user_id")
        _validate_identifier(self.movie_id, "movie_id")
        _validate_bounded_number(
            self.predicted_score,
            "predicted_score",
            0.0,
            5.0,
        )
        _validate_bounded_number(self.confidence, "confidence", 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
