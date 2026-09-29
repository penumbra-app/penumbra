import json
from dataclasses import replace

import pytest

from src.content import ContentModel, MovieMetadata, ProfileConfig, ScoringConfig, UserRating, build_profile, predict_one
from src.data_processing.movielens import movie_metadata_from_record


def cast_profile(cast, config=None):
    return build_profile(1, (UserRating(1, 1, 5), UserRating(1, 2, 1)),
                         (MovieMetadata(1, "Liked", cast=cast), MovieMetadata(2, "Disliked")), config)


def test_cast_splits_residual_and_evidence_and_deduplicates():
    profile = cast_profile((" A ", "a", "B"))
    learned = [p for p in profile.feature_preferences if p.feature_type == "cast"]
    assert len(learned) == 2
    assert sum(p.effective_evidence_count for p in learned) == 1
    assert all(p.movie_count == 1 and p.preference == pytest.approx(1 / 5.5) for p in learned)
    config = ScoringConfig(cast_weight=0.25)
    result = predict_one(profile, MovieMetadata(3, "Candidate", cast=("a", "A", "B")), config, True)
    assert result.predicted_score == pytest.approx(3 + 0.25 / 5.5)
    assert result.confidence == pytest.approx((2 / 7) * (0.5 / 5.5))
    assert len(result.reason_signals) == 2


def test_large_cast_tail_cannot_change_profile_score_or_confidence():
    head = tuple(f"Actor {i}" for i in range(10))
    large = head + tuple(f"Extra {i}" for i in range(1000))
    assert cast_profile(head) == cast_profile(large)
    profile = cast_profile(head)
    config = ScoringConfig(cast_weight=0.25)
    first = predict_one(profile, MovieMetadata(3, "M", cast=head), config, True)
    second = predict_one(profile, MovieMetadata(3, "M", cast=large), config, True)
    assert first == second
    single = predict_one(cast_profile(("A",)), MovieMetadata(3, "M", cast=("A",)), config)
    assert first.predicted_score < single.predicted_score
    assert first.confidence < single.confidence


def test_custom_cast_cutoff_is_shared_between_training_and_prediction():
    profile = cast_profile(("a", "b", "c"), ProfileConfig(max_cast_members=2))
    result = predict_one(profile, MovieMetadata(3, "M", cast=("a", "b", "c")),
                         ScoringConfig(cast_weight=0.25), True)
    assert result.debug.feature_components[-1].candidate_values == ("a", "b")
    assert all(p.feature_value != "c" for p in profile.feature_preferences)


@pytest.mark.parametrize("config", [ScoringConfig(), ScoringConfig(cast_weight=0.25, max_abs_cast_component=0)])
def test_disabled_cast_supplies_no_score_confidence_or_reasons(config):
    result = predict_one(cast_profile(("a",)), MovieMetadata(3, "M", cast=("a",)), config)
    assert (result.predicted_score, result.confidence, result.reason_signals) == (3, 0, ())


def test_cast_cap_with_extreme_weight_and_unknown_cast():
    profile = cast_profile(("a",))
    result = predict_one(profile, MovieMetadata(3, "M", cast=("a",)), ScoringConfig(cast_weight=1e308), True)
    assert result.predicted_score == 3.25
    assert sum(r.score_contribution for r in result.reason_signals) == 0.25
    unknown = predict_one(profile, MovieMetadata(3, "M", cast=("z",)), ScoringConfig(cast_weight=0.25))
    assert (unknown.confidence, unknown.reason_signals, unknown.fallback_reason) == (0, (), "no_matching_evidence")


def test_recency_discounts_shared_cast_evidence():
    profile = build_profile(1, (UserRating(1, 1, 5, 0), UserRating(1, 2, 1, 365 * 86400)),
                            (MovieMetadata(1, "M", cast=("a", "b")), MovieMetadata(2, "N")))
    assert all(p.effective_evidence_count == 0.375 for p in profile.feature_preferences)
    assert all(p.preference == pytest.approx(0.75 / 5.375) for p in profile.feature_preferences)


def test_reason_contributions_reconstruct_score_with_caps_and_weights():
    movies = (MovieMetadata(1, "Liked", ("Action", "Comedy"), cast=("a",), language="en"),
              MovieMetadata(2, "Disliked", ("Drama",), cast=("b",), language="fr"))
    profile = build_profile(1, (UserRating(1, 1, 5), UserRating(1, 2, 1)), movies)
    for weight in (0.25, 1e308):
        config = ScoringConfig(genre_weight=weight, cast_weight=weight, language_weight=weight)
        for candidate in (movies[0], movies[1], replace(movies[0], genres=("Action", "Unknown"))):
            result = predict_one(profile, candidate, config, True)
            assert sum(reason.score_contribution for reason in result.reason_signals) == pytest.approx(
                result.debug.unclamped_score - profile.baseline)
            assert all(reason.evidence_count > 0 for reason in result.reason_signals)
            assert all(reason.feature_value != "Unknown" for reason in result.reason_signals)
            json.dumps(result.to_dict(), allow_nan=False)


def test_cold_start_is_consistent_across_all_apis_and_ignores_other_users():
    movies = (MovieMetadata(1, "M", ("Action",), cast=("a",)), MovieMetadata(2, "N"))
    config = ProfileConfig(cold_start_score=3)
    model = ContentModel((UserRating(1, 1, 5),), movies, profile_config=config)
    profile = model.build_profile(99)
    single = model.predict_one(99, 1, True)
    assert model.predict((99,), (1,), True) == (single,)
    assert model.predict_unseen((99,), limit=1, include_debug=True) == (single,)
    assert predict_one(profile, movies[0], include_debug=True) == single
    assert model.unseen_movie_ids(99) == (1, 2)
    assert (single.predicted_score, single.confidence, single.reason_signals) == (3, 0, ())
    assert single.fallback_reason == "cold_start_user"
    assert ContentModel((), movies).predict_one(1, 1).predicted_score == 2.5


def test_missing_history_metadata_keeps_baseline_and_no_invented_reasons():
    model = ContentModel((UserRating(1, 100, 5), UserRating(1, 101, 1)), (MovieMetadata(1, "M"),))
    result = model.predict_one(1, 1)
    assert (result.predicted_score, result.confidence, result.reason_signals) == (3, 0, ())
    assert result.fallback_reason == "missing_metadata"


@pytest.mark.parametrize("missing", [None, "", "\\N", "NaN", "null", float("nan")])
def test_loader_skips_missing_metadata_sentinels(missing):
    movie = movie_metadata_from_record(dict(movieId=1, title="M", genres=missing, cast=missing,
        directors=missing, language=missing, runtime_minutes=missing, release_year=missing))
    assert movie == MovieMetadata(1, "M")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_cast_cutoff_validation(value):
    with pytest.raises(ValueError):
        ProfileConfig(max_cast_members=value)


@pytest.mark.parametrize("value", [-1, 6, True, float("nan"), float("inf")])
def test_cold_start_score_validation(value):
    with pytest.raises(ValueError):
        ProfileConfig(cold_start_score=value)


@pytest.mark.parametrize("user_id", [0, -1, True, 1.5])
def test_invalid_cold_user_ids_are_not_accepted(user_id):
    with pytest.raises(ValueError):
        ContentModel((), ()).build_profile(user_id)
