import unittest

from src.content.profiles import PROFILE_VERSION, build_profile
from src.content.reliability import ProfileConfig
from src.content.schemas import UserRating
from tests.fixtures import TOY_MOVIES, TOY_RATINGS


class ProfileTests(unittest.TestCase):
    def test_profile_contains_baseline_preferences_and_version(self) -> None:
        profile = build_profile(
            1, TOY_RATINGS, TOY_MOVIES,
            ProfileConfig(recency_half_life_days=None),
        )

        self.assertEqual(profile.user_id, 1)
        self.assertAlmostEqual(profile.baseline, 3.25)
        self.assertEqual(profile.rating_count, 4)
        self.assertEqual(profile.profile_version, PROFILE_VERSION)
        self.assertAlmostEqual(profile.preference_for("Sci-Fi"), 1.5 / 7)
        self.assertEqual(profile.evidence_for("Sci-Fi"), 2)

    def test_profile_metadata_tracks_missing_movie_and_genres(self) -> None:
        ratings = (
            UserRating(4, 1, 5.0),
            UserRating(4, 9, 3.0),
            UserRating(4, 999, 1.0),
        )
        profile = build_profile(4, ratings, TOY_MOVIES)

        self.assertEqual(profile.metadata.ratings_used, 3)
        self.assertEqual(profile.metadata.movies_with_metadata, 2)
        self.assertEqual(profile.metadata.movies_with_genres, 1)
        self.assertEqual(profile.metadata.ratings_without_movie_metadata, 1)
        self.assertEqual(profile.metadata.ratings_without_genres, 1)

    def test_unknown_genre_access_is_neutral(self) -> None:
        profile = build_profile(1, TOY_RATINGS, TOY_MOVIES)

        self.assertEqual(profile.preference_for("Documentary"), 0.0)
        self.assertEqual(profile.evidence_for("Documentary"), 0)

    def test_unknown_user_has_empty_profile(self) -> None:
        profile = build_profile(999, TOY_RATINGS, TOY_MOVIES)
        self.assertEqual((profile.baseline, profile.rating_count), (2.5, 0))
        self.assertEqual(profile.genre_preferences, ())
        self.assertEqual(profile.feature_preferences, ())


if __name__ == "__main__":
    unittest.main()
