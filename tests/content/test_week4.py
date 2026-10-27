
"""Regression tests for the current Content model contract."""

import json
from dataclasses import replace

import pytest

from src.content import (
    ContentModel,
    MovieMetadata,
    ProfileConfig,
    ScoringConfig,
    UserRating,
    build_profile,
    predict_one,
)
from src.data_processing.movielens import movie_metadata_from_record


def cast_profile(cast, config=None):
    ratings = (UserRating(1, 1, 5), UserRating(1, 2, 1))
    movies = (
        MovieMetadata(1, "Liked", cast=cast),
        MovieMetadata(2, "Disliked"),
    )
    return build_profile(1, ratings, movies, config)


def test_cast_deduplication_and_movie_level_evidence():
    profile = cast_profile((" A ", "a", "B"))
    learned = [
        p for p in profile.feature_preferences
        if p.feature_type == "cast"
    ]

    assert {p.feature_value for p in learned} == {"a", "b"}
    assert all(p.movie_count == 1 for p in learned)
    assert all(
        p.effective_evidence_count == pytest.approx(1.0)
        for p in learned
    )
    assert all(p.preference == pytest.approx(1 / 6) for p in learned)

    result = predict_one(
        profile,
        MovieMetadata(3, "New", cast=("a", "A", "B")),
        ScoringConfig(cast_weight=0.25),
    )

    assert result.predicted_score == pytest.approx(3 + 0.25 / 6)
    assert result.confidence == pytest.approx((2 / 7) * (1 / 6))
    assert len(result.reason_signals) == 2


def test_full_cast_is_used_without_old_cutoff():
    head = tuple(f"Actor {i}" for i in range(10))
    extended = head + tuple(f"Extra {i}" for i in range(100))

    short = cast_profile(head)
    long = cast_profile(extended)

    short_prefs = [
        p for p in short.feature_preferences
        if p.feature_type == "cast"
    ]
    long_prefs = [
        p for p in long.feature_preferences
        if p.feature_type == "cast"
    ]

    assert len(short_prefs) == 10
    assert len(long_prefs) == 110

    assert max(p.preference for p in long_prefs) < min(
        p.preference for p in short_prefs
    )
    assert any(p.feature_value == "extra 99" for p in long_prefs)

    prediction = predict_one(
        long,
        MovieMetadata(3, "New", cast=extended),
        ScoringConfig(cast_weight=0.25),
        True,
    )

    cast_debug = next(
        component
        for component in prediction.debug.feature_components
        if component.feature_type == "cast"
    )

    assert len(cast_debug.candidate_values) == 110


@pytest.mark.parametrize("config", [
    ScoringConfig(cast_weight=0),
    ScoringConfig(cast_weight=0.25, max_abs_cast_component=0),
])
def test_disabled_cast_has_no_adjustment_or_reasons(config):
    result = predict_one(
        cast_profile(("a",)),
        MovieMetadata(3, "New", cast=("a",)),
        config,
    )

    assert (
        result.predicted_score,
        result.confidence,
        result.reason_signals,
    ) == (3, 0, ())


def test_cast_score_is_capped_and_unknown_cast_stays_neutral():
    profile = cast_profile(("a",))

    saturated = predict_one(
        profile,
        MovieMetadata(3, "New", cast=("a",)),
        ScoringConfig(cast_weight=1e308),
        True,
    )

    assert saturated.predicted_score == pytest.approx(3.25)
    assert sum(
        reason.score_contribution
        for reason in saturated.reason_signals
    ) == pytest.approx(0.25)

    unknown = predict_one(
        profile,
        MovieMetadata(3, "New", cast=("z",)),
    )

    assert (
        unknown.predicted_score,
        unknown.confidence,
        unknown.reason_signals,
    ) == (3, 0, ())


def test_recency_reduces_cast_preference_and_evidence():
    profile = build_profile(
        1,
        (
            UserRating(1, 1, 5, 0),
            UserRating(1, 2, 1, 365 * 86400),
        ),
        (
            MovieMetadata(1, "Liked", cast=("a", "b")),
            MovieMetadata(2, "Disliked"),
        ),
    )

    cast = [
        p for p in profile.feature_preferences
        if p.feature_type == "cast"
    ]

    assert len(cast) == 2
    assert all(
        p.effective_evidence_count == pytest.approx(0.75)
        for p in cast
    )
    assert all(
        p.preference == pytest.approx(0.75 / 5.75)
        for p in cast
    )


def test_reason_contributions_reconstruct_unclamped_score():
    movies = (
        MovieMetadata(
            1, "Liked", ("Action", "Comedy"),
            cast=("a",), language="en",
        ),
        MovieMetadata(
            2, "Disliked", ("Drama",),
            cast=("b",), language="fr",
        ),
    )

    profile = build_profile(
        1,
        (UserRating(1, 1, 5), UserRating(1, 2, 1)),
        movies,
    )

    for weight in (0.25, 1e308):
        config = ScoringConfig(
            genre_weight=weight,
            cast_weight=weight,
            language_weight=weight,
        )

        candidates = (
            movies[0],
            movies[1],
            replace(movies[0], genres=("Action", "Unknown")),
        )

        for movie in candidates:
            result = predict_one(profile, movie, config, True)

            contribution = sum(
                r.score_contribution for r in result.reason_signals
            )

            assert contribution == pytest.approx(
                result.debug.unclamped_score - profile.baseline
            )
            assert all(
                r.evidence_count > 0
                for r in result.reason_signals
            )
            assert all(
                r.feature_value != "Unknown"
                for r in result.reason_signals
            )

            json.dumps(result.to_dict(), allow_nan=False)


def test_cold_start_is_consistent_across_apis():
    movies = (
        MovieMetadata(1, "M", ("Action",), cast=("a",)),
        MovieMetadata(2, "N"),
    )

    model = ContentModel(
        (UserRating(1, 1, 5),),
        movies,
        profile_config=ProfileConfig(cold_start_score=3),
    )

    profile = model.build_profile(99)
    single = model.predict_one(99, 1, True)

    assert model.predict((99,), (1,), True) == (single,)
    assert model.predict_unseen(
        (99,), limit=1, include_debug=True
    ) == (single,)
    assert predict_one(profile, movies[0], include_debug=True) == single

    assert model.unseen_movie_ids(99) == (1, 2)
    assert (
        single.predicted_score,
        single.confidence,
        single.reason_signals,
    ) == (3, 0, ())

    assert ContentModel(
        (), movies
    ).predict_one(1, 1).predicted_score == 2.5


def test_missing_history_metadata_keeps_baseline():
    model = ContentModel(
        (UserRating(1, 100, 5), UserRating(1, 101, 1)),
        (MovieMetadata(1, "M"),),
    )

    result = model.predict_one(1, 1)

    assert (
        result.predicted_score,
        result.confidence,
        result.reason_signals,
    ) == (3, 0, ())


@pytest.mark.parametrize(
    "missing",
    [None, "", "\\N", "NaN", "null", float("nan")],
)
def test_missing_metadata_sentinels(missing):
    movie = movie_metadata_from_record({
        "movieId": 1,
        "title": "M",
        "genres": missing,
        "cast": missing,
        "directors": missing,
        "language": missing,
        "runtime_minutes": missing,
        "release_year": missing,
    })

    assert movie == MovieMetadata(1, "M")


@pytest.mark.parametrize(
    "value",
    [-1, 6, True, float("nan"), float("inf")],
)
def test_cold_start_score_validation(value):
    with pytest.raises(ValueError):
        ProfileConfig(cold_start_score=value)


@pytest.mark.parametrize("user_id", [0, -1, True, 1.5])
def test_invalid_cold_user_ids(user_id):
    with pytest.raises(ValueError):
        ContentModel((), ()).build_profile(user_id)
