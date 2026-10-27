from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.content.genres import _unique_genres
from src.content.features import FEATURE_TYPES, feature_values
from src.content.profiles import UserTasteProfile
from src.content.schemas import (
    FeatureDebug, MovieMetadata, PredictionDebug, PredictionResult, ReasonSignal,
)

if TYPE_CHECKING:
    from src.content.text import TextProfile, TextSignal


@dataclass(frozen=True)
class ScoringConfig:
    genre_weight: float = 1.0
    max_abs_genre_component: float = 1.0
    confidence_prior_count: float = 5.0

    cast_weight: float = 0.25
    max_abs_cast_component: float = 0.25
    director_weight: float = 0.5
    runtime_weight: float = 0.25
    release_era_weight: float = 0.25
    language_weight: float = 0.25
    max_abs_director_component: float = 0.5
    max_abs_runtime_component: float = 0.25
    max_abs_release_era_component: float = 0.25
    max_abs_language_component: float = 0.25

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
            if name.startswith("max_abs_") and value > 5:
                raise ValueError(f"{name} must not exceed 5 rating points")
        if self.confidence_prior_count == 0:
            raise ValueError("confidence_prior_count must be positive")


def _bounded_adjustment(component: float, weight: float, bound: float) -> float:
    # Check saturation before multiplication, including extremely large weights.
    if weight == 0 or bound == 0 or component == 0:
        return 0.0
    if abs(component) >= bound / weight:
        return math.copysign(bound, component)
    return component * weight


@dataclass(frozen=True)
class GenreMatch:
    genre: str
    preference: float
    evidence_count: int
    effective_evidence_count: float


@dataclass(frozen=True)
class GenreComponentResult:
    raw_contribution: float
    bounded_contribution: float
    matches: tuple[GenreMatch, ...]
    unknown_genres: tuple[str, ...]


def genre_component(
    movie: MovieMetadata,
    profile: UserTasteProfile,
    max_abs_contribution: float = 1.0,
) -> GenreComponentResult:
    if not math.isfinite(max_abs_contribution) or max_abs_contribution < 0:
        raise ValueError("max_abs_contribution must be finite and non-negative")

    movie_genres = _unique_genres(movie.genres)
    preference_map = {
        preference.genre.casefold(): preference
        for preference in profile.genre_preferences
    }
    matches: list[GenreMatch] = []
    unknown_genres: list[str] = []
    contributions: list[float] = []

    for genre in movie_genres:
        preference = preference_map.get(genre.casefold())
        if preference is None:
            unknown_genres.append(genre)
            contributions.append(0.0)
            continue

        matches.append(
            GenreMatch(
                genre=genre,
                preference=preference.preference,
                evidence_count=preference.movie_count,
                effective_evidence_count=preference.evidence_weight,
            )
        )
        contributions.append(preference.preference)

    raw_contribution = (
        float(sum(contributions) / len(contributions))
        if contributions
        else 0.0
    )
    bounded_contribution = max(
        -max_abs_contribution,
        min(max_abs_contribution, raw_contribution),
    )

    return GenreComponentResult(
        raw_contribution=raw_contribution,
        bounded_contribution=float(bounded_contribution),
        matches=tuple(matches),
        unknown_genres=tuple(unknown_genres),
    )


def _supported_reasons(
    kind: str, matches: list[tuple[str, float, float]],
    value_count: int, raw: float, adjustment: float,
) -> tuple[ReasonSignal, ...]:
    """Allocate the actual feature adjustment after averaging, caps and weight.

    Contributions explain the score before the final 0–5 clamp. A feature
    whose preferences cancel or whose weight is disabled emits no reasons.
    """
    if not value_count or raw == 0 or adjustment == 0:
        return ()
    # Allocate the net adjustment over supporting values in its direction.
    # Opposing values already reduce raw; this keeps attributions bounded even
    # when nearly cancelling preferences meet an extreme configured weight.
    supporting = [(v, p, e) for v, p, e in matches if p * raw > 0 and e > 0]
    total = sum(abs(p) for _, p, _ in supporting)
    return tuple(
        ReasonSignal(kind, value, contribution / 5.0, evidence, contribution)
        for value, preference, evidence in supporting
        for contribution in (adjustment * abs(preference) / total,)
    )


def predict_one(
    profile: UserTasteProfile,
    movie: MovieMetadata,
    config: ScoringConfig | None = None,
    include_debug: bool = False,
    *, text_signal: TextSignal | None = None,
) -> PredictionResult:
    active_config = config or ScoringConfig()
    component = genre_component(
        movie,
        profile,
        active_config.max_abs_genre_component,
    )
    weighted_adjustment = (
        _bounded_adjustment(component.bounded_contribution,
                            active_config.genre_weight,
                            active_config.max_abs_genre_component)
    )
    movie_genres = _unique_genres(movie.genres)
    prior = active_config.confidence_prior_count
    # Average support over ALL candidate genres. Unknown genres supply zero;
    # averaging prevents overlapping multi-genre evidence from being summed.
    genre_support = (
        sum(
            match.effective_evidence_count / (match.effective_evidence_count + prior)
            for match in component.matches
        ) / len(movie_genres)
        if movie_genres else 0.0
    )
    enabled = (
        active_config.genre_weight > 0
        and active_config.max_abs_genre_component > 0
    )
    reasons = _supported_reasons(
        "genre", [(m.genre, m.preference, m.effective_evidence_count)
                  for m in component.matches],
        len(movie_genres), component.raw_contribution, weighted_adjustment,
    )
    feature_debug: list[FeatureDebug] = []
    supports = [genre_support] if enabled and movie_genres else []
    extra_reasons: list[ReasonSignal] = []
    adjustments = [weighted_adjustment]
    preference_map = {
        (p.feature_type, p.feature_value): p for p in profile.feature_preferences
    }
    for kind in FEATURE_TYPES:
        values = feature_values(movie, kind)
        matches = [preference_map[(kind, value)] for value in values
                   if (kind, value) in preference_map]
        raw = sum(p.preference for p in matches) / len(values) if values else 0.0
        bound = getattr(active_config, f"max_abs_{kind}_component")
        weight = getattr(active_config, f"{kind}_weight")
        bounded = max(-bound, min(bound, raw))
        adjustment = _bounded_adjustment(bounded, weight, bound)
        support = (sum(p.effective_evidence_count / (p.effective_evidence_count + prior)
                       for p in matches) / len(values) if values else 0.0)
        if weight > 0 and bound > 0:
            if values:
                supports.append(support)
            extra_reasons.extend(_supported_reasons(
                kind, [(p.feature_value, p.preference, p.effective_evidence_count)
                       for p in matches], len(values), raw, adjustment,
            ))
        adjustments.append(adjustment)
        feature_debug.append(FeatureDebug(
            kind, values, tuple(p.feature_value for p in matches), raw,
            bounded, weight, adjustment, support,
        ))
    # Average feature support instead of counting the same ratings repeatedly.
    confidence = (profile.rating_count / (profile.rating_count + prior)
                  * sum(supports) / len(supports)) if supports else 0.0
    reasons += tuple(extra_reasons)
    if text_signal is not None and text_signal.evidence > 0:
        adjustments.append(text_signal.adjustment)
        text_support = text_signal.evidence / (text_signal.evidence + prior)
        # Max support avoids summing overlapping metadata evidence. This is
        # evidence strength, not a calibrated accuracy probability.
        confidence = max(confidence, profile.rating_count / (profile.rating_count + prior)
                         * text_support)
        if text_signal.adjustment:
            reasons += (ReasonSignal(
                "text", f"{text_signal.representation.upper()} similarity to rated movie text",
                text_signal.adjustment / 5, text_signal.evidence, text_signal.adjustment,
            ),)
        feature_debug.append(FeatureDebug(
            "text", (text_signal.representation,), ("rated movie text",),
            text_signal.raw_component, text_signal.adjustment, text_signal.weight,
            text_signal.adjustment, text_support,
        ))
    unclamped_score = profile.baseline + sum(adjustments)
    predicted_score = max(0.0, min(5.0, unclamped_score))
    debug = None

    if include_debug:
        debug = PredictionDebug(
            baseline=profile.baseline,
            feature_components=tuple(feature_debug),
            movie_genres=movie_genres,
            matched_genres=tuple(match.genre for match in component.matches),
            unknown_genres=component.unknown_genres,
            raw_genre_component=component.raw_contribution,
            bounded_genre_component=component.bounded_contribution,
            genre_weight=active_config.genre_weight,
            weighted_genre_adjustment=weighted_adjustment,
            unclamped_score=unclamped_score,
            was_clamped=predicted_score != unclamped_score,
            genre_support=genre_support,
            effective_genre_evidence=tuple(
                (match.genre, match.effective_evidence_count)
                for match in component.matches
            ),
        )

    return PredictionResult(
        user_id=profile.user_id,
        movie_id=movie.movie_id,
        predicted_score=float(predicted_score),
        confidence=float(confidence),
        reason_signals=reasons,
        debug=debug,
    )


def predict_batch(
    profile: UserTasteProfile,
    movies: Iterable[MovieMetadata],
    config: ScoringConfig | None = None,
    include_debug: bool = False,
    *, text_profile: TextProfile | None = None,
) -> tuple[PredictionResult, ...]:
    """Score candidates in input order, including movies outside a model catalog."""
    candidates = tuple(movies)
    if text_profile is None:
        return tuple(predict_one(profile, movie, config, include_debug) for movie in candidates)
    if text_profile.user_id is not None and text_profile.user_id != profile.user_id:
        raise ValueError("Text profile user must match the taste profile user")
    return tuple(predict_one(profile, movie, config, include_debug, text_signal=signal)
                 for movie, signal in zip(candidates, text_profile.signals(candidates), strict=True))
