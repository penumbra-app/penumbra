import unittest

from src.content.genres import GenrePreference
from src.content.profiles import ProfileMetadata, UserTasteProfile, build_profile
from src.content.reliability import ProfileConfig
from src.content.schemas import MovieMetadata
from src.content.scoring import ScoringConfig, genre_component, predict_one
from tests.fixtures import TOY_MOVIES, TOY_MOVIES_BY_ID, TOY_RATINGS


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        # Preserve the exact Week 1 arithmetic as an explicit unregularized mode.
        self.profile = build_profile(
            1, TOY_RATINGS, TOY_MOVIES,
            ProfileConfig(regularization_strength=0, recency_half_life_days=None),
        )

    def test_positive_scifi_movie_scores_above_baseline(self) -> None:
        result = predict_one(self.profile, TOY_MOVIES_BY_ID[8])

        self.assertAlmostEqual(result.predicted_score, 4.0)
        self.assertGreater(result.predicted_score, self.profile.baseline)

    def test_negative_drama_movie_scores_below_baseline(self) -> None:
        result = predict_one(self.profile, TOY_MOVIES_BY_ID[7])

        # The disliked 2000s era now contributes another -0.0625.
        self.assertAlmostEqual(result.predicted_score, 2.3125)
        self.assertLess(result.predicted_score, self.profile.baseline)

    def test_unknown_genre_returns_baseline(self) -> None:
        movie = MovieMetadata(11, "Unknown Genre", ("Documentary",))
        result = predict_one(self.profile, movie, include_debug=True)

        self.assertAlmostEqual(result.predicted_score, 3.25)
        self.assertEqual(result.debug.unknown_genres, ("Documentary",))
        self.assertEqual(result.debug.weighted_genre_adjustment, 0.0)

    def test_missing_genres_return_baseline(self) -> None:
        result = predict_one(
            self.profile,
            TOY_MOVIES_BY_ID[9],
            include_debug=True,
        )

        self.assertAlmostEqual(result.predicted_score, 3.25)
        self.assertEqual(result.debug.movie_genres, ())
        self.assertEqual(result.debug.bounded_genre_component, 0.0)

    def test_five_genre_prediction_matches_manual_calculation(self) -> None:
        result = predict_one(
            self.profile,
            TOY_MOVIES_BY_ID[10],
            include_debug=True,
        )

        self.assertAlmostEqual(result.debug.raw_genre_component, 0.15)
        self.assertAlmostEqual(result.predicted_score, 3.4)

    def test_genre_weight_scales_adjustment(self) -> None:
        result = predict_one(
            self.profile,
            TOY_MOVIES_BY_ID[8],
            config=ScoringConfig(genre_weight=0.5),
            include_debug=True,
        )

        self.assertAlmostEqual(result.debug.weighted_genre_adjustment, 0.375)
        self.assertAlmostEqual(result.predicted_score, 3.625)

    def test_genre_component_exposes_matches_and_evidence(self) -> None:
        component = genre_component(TOY_MOVIES_BY_ID[10], self.profile)
        matches = {match.genre: match for match in component.matches}

        self.assertEqual(matches["Sci-Fi"].evidence_count, 2)
        self.assertAlmostEqual(matches["Sci-Fi"].preference, 0.75)
        self.assertEqual(component.unknown_genres, ())

    def test_upper_score_is_clamped_to_five(self) -> None:
        profile = self._extreme_profile(4.75, 10.0)
        result = predict_one(
            profile,
            MovieMetadata(20, "Positive", ("Sci-Fi",)),
            include_debug=True,
        )

        self.assertEqual(result.predicted_score, 5.0)
        self.assertAlmostEqual(result.debug.unclamped_score, 5.75)
        self.assertTrue(result.debug.was_clamped)

    def test_lower_score_is_clamped_to_zero(self) -> None:
        profile = self._extreme_profile(0.25, -10.0)
        result = predict_one(
            profile,
            MovieMetadata(21, "Negative", ("Sci-Fi",)),
            include_debug=True,
        )

        self.assertEqual(result.predicted_score, 0.0)
        self.assertAlmostEqual(result.debug.unclamped_score, -0.75)
        self.assertTrue(result.debug.was_clamped)

    def test_debug_data_is_optional(self) -> None:
        result = predict_one(self.profile, TOY_MOVIES_BY_ID[8])

        self.assertIsNone(result.debug)

    def test_confidence_and_reasons_expose_genre_evidence(self) -> None:
        result = predict_one(self.profile, TOY_MOVIES_BY_ID[8])

        # The candidate's unseen 2020s era supplies zero support.
        self.assertAlmostEqual(result.confidence, (4 / 9) * ((2 / 7) / 2))
        self.assertEqual(result.reason_signals[0].feature_value, "Sci-Fi")
        self.assertEqual(result.reason_signals[0].evidence_count, 2)

    def _extreme_profile(
        self,
        baseline: float,
        preference: float,
    ) -> UserTasteProfile:
        return UserTasteProfile(
            user_id=50,
            baseline=baseline,
            rating_count=1,
            genre_preferences=(
                GenrePreference("Sci-Fi", preference, preference, 1),
            ),
            profile_version="test",
            metadata=ProfileMetadata(1, 1, 1, 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
