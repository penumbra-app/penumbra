"""Additional content signals, learned with the same shrinkage as genres."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.content.baselines import UserBaseline
from src.content.genres import _unique_genres
from src.content.schemas import MovieMetadata

FEATURE_TYPES = ("director", "runtime", "release_era", "language")


@dataclass(frozen=True)
class FeaturePreference:
    feature_type: str
    feature_value: str
    preference: float
    movie_count: int
    effective_evidence_count: float


def feature_values(movie: MovieMetadata, feature_type: str) -> tuple[str, ...]:
    """Use stable buckets; absent metadata never becomes a learned category."""
    if feature_type == "director":
        return tuple(value.casefold() for value in _unique_genres(movie.directors))
    if feature_type == "runtime":
        runtime = movie.runtime_minutes
        if runtime is None:
            return ()
        return ("under 90 min" if runtime < 90 else
                "90–119 min" if runtime < 120 else
                "120–149 min" if runtime < 150 else "150+ min",)
    if feature_type == "release_era":
        return () if movie.release_year is None else (f"{movie.release_year // 10 * 10}s",)
    if feature_type == "language":
        return (movie.language.strip().casefold(),) if movie.language and movie.language.strip() else ()
    raise ValueError(f"Unknown feature type: {feature_type}")


def aggregate_feature_preferences(
    baseline: UserBaseline,
    movies: Mapping[int, MovieMetadata],
    weights: Mapping[int, float],
    regularization_strength: float,
) -> tuple[FeaturePreference, ...]:
    totals: dict[tuple[str, str], float] = {}
    evidence: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for residual in baseline.residuals:
        movie = movies.get(residual.movie_id)
        if movie is None:
            continue
        weight = weights[residual.movie_id]
        for feature_type in FEATURE_TYPES:
            values = feature_values(movie, feature_type)
            for value in values:
                key = (feature_type, value)
                totals[key] = totals.get(key, 0.0) + weight * residual.residual / len(values)
                evidence[key] = evidence.get(key, 0.0) + weight
                counts[key] = counts.get(key, 0) + 1
    return tuple(
        FeaturePreference(kind, value, totals[(kind, value)] / (
            evidence[(kind, value)] + regularization_strength
        ), counts[(kind, value)], evidence[(kind, value)])
        for kind, value in sorted(totals)
    )
