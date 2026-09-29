"""Deterministic settings for evidence shrinkage and gentle recency weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.content.schemas import UserRating


@dataclass(frozen=True)
class ProfileConfig:
    regularization_strength: float = 5.0
    recency_half_life_days: float | None = 365.0
    minimum_recency_weight: float = 0.5
    reference_timestamp: int | None = None
    max_cast_members: int = 10
    cold_start_score: float = 2.5

    def __post_init__(self) -> None:
        if (isinstance(self.max_cast_members, bool)
                or not isinstance(self.max_cast_members, int)
                or self.max_cast_members <= 0):
            raise ValueError("max_cast_members must be a positive integer")
        if (isinstance(self.cold_start_score, bool)
                or not isinstance(self.cold_start_score, (int, float))
                or not math.isfinite(self.cold_start_score)
                or not 0 <= self.cold_start_score <= 5):
            raise ValueError("cold_start_score must be finite and in [0, 5]")
        for name in ("regularization_strength", "minimum_recency_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < self.minimum_recency_weight <= 1:
            raise ValueError("minimum_recency_weight must be in (0, 1]")
        half_life = self.recency_half_life_days
        if half_life is not None and (
            isinstance(half_life, bool)
            or not isinstance(half_life, (int, float))
            or not math.isfinite(half_life)
            or half_life <= 0
        ):
            raise ValueError("recency_half_life_days must be positive or None")
        reference = self.reference_timestamp
        if reference is not None and (
            isinstance(reference, bool)
            or not isinstance(reference, int)
            or reference < 0
        ):
            raise ValueError(
                "reference_timestamp must be a non-negative integer or None"
            )


def rating_weights(
    ratings: tuple[UserRating, ...],
    config: ProfileConfig,
) -> tuple[dict[int, float], int | None]:
    """Use the latest observed timestamp by default, never the wall clock.

    An explicit reference is an as-of boundary; future observations are rejected.
    Missing dates get the floor weight in mixed histories, or 1 if all are absent.
    """
    latest = max(
        (r.timestamp for r in ratings if r.timestamp is not None), default=None,
    )
    reference = config.reference_timestamp
    if reference is None:
        reference = latest
    if latest is not None and reference is not None and latest > reference:
        raise ValueError("ratings must not be later than reference_timestamp")

    weights: dict[int, float] = {}
    for rating in ratings:
        if config.recency_half_life_days is None or reference is None:
            weight = 1.0
        elif rating.timestamp is None:
            weight = config.minimum_recency_weight
        else:
            age_days = (reference - rating.timestamp) / 86400
            weight = config.minimum_recency_weight + (
                1 - config.minimum_recency_weight
            ) * 2 ** (-age_days / config.recency_half_life_days)
        weights[rating.movie_id] = weight
    return weights, reference
