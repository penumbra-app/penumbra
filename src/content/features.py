"""Additional content signals, learned with the same shrinkage as genres."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.content.baselines import UserBaseline
from src.content.genres import _unique_genres
from src.content.schemas import MovieMetadata

FEATURE_TYPES = ("director", "runtime", "release_era", "language", "cast")


@dataclass(frozen=True)
class FeaturePreference:
    feature_type: str
    feature_value: str
    preference: float
    movie_count: int
    effective_evidence_count: float


def feature_values(
    movie: MovieMetadata, feature_type: str, max_cast_members: int = 10,
) -> tuple[str, ...]:
    """Use stable buckets; absent metadata never becomes a learned category."""
    if feature_type == "director":
        return tuple(value.casefold() for value in _unique_genres(movie.directors))
    if feature_type == "cast":
        if (isinstance(max_cast_members, bool) or not isinstance(max_cast_members, int)
                or max_cast_members <= 0):
            raise ValueError("max_cast_members must be a positive integer")
        # The input is in credit order. Dedupe before taking the bounded prefix.
        return tuple(value.casefold() for value in
                     _unique_genres(movie.cast)[:max_cast_members])
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
    max_cast_members: int = 10,
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
            values = feature_values(movie, feature_type, max_cast_members)
            for value in values:
                key = (feature_type, value)
                totals[key] = totals.get(key, 0.0) + weight * residual.residual / len(values)
                # Cast shares both residual AND evidence: one film cannot supply
                # ten independent observations just because it has ten actors.
                evidence_weight = weight / len(values) if feature_type == "cast" else weight
                evidence[key] = evidence.get(key, 0.0) + evidence_weight
                counts[key] = counts.get(key, 0) + 1
    return tuple(
        FeaturePreference(kind, value, totals[(kind, value)] / (
            evidence[(kind, value)] + regularization_strength
        ), counts[(kind, value)], evidence[(kind, value)])
        for kind, value in sorted(totals)
    )
