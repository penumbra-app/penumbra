import json
import unittest
from dataclasses import replace

from src.content import (
    ContentModel, MovieMetadata, ProfileConfig, ScoringConfig, UserRating,
    build_profile, feature_values, predict_batch, predict_one,
)
from src.data_processing.movielens import movie_metadata_from_record


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.movies = (
            MovieMetadata(1, "Liked", ("Action",), directors=("A",),
                          runtime_minutes=95, release_year=1995, language="en"),
            MovieMetadata(2, "Disliked", ("Drama",), directors=("B",),
                          runtime_minutes=160, release_year=2020, language="fr"),
        )
        self.ratings = (UserRating(1, 1, 5), UserRating(1, 2, 1))
        self.profile = build_profile(1, self.ratings, self.movies)

    def test_each_feature_learns_positive_and_negative_preferences(self):
        for kind, positive, negative in (
            ("director", dict(directors=("a",)), dict(directors=("b",))),
            ("runtime", dict(runtime_minutes=110), dict(runtime_minutes=180)),
            ("release_era", dict(release_year=1999), dict(release_year=2025)),
            ("language", dict(language=" EN "), dict(language="fr")),
        ):
            with self.subTest(kind=kind):
                liked = predict_one(self.profile, MovieMetadata(3, "New", **positive), include_debug=True)
                disliked = predict_one(self.profile, MovieMetadata(4, "New", **negative))
                self.assertGreater(liked.predicted_score, 3)
                self.assertLess(disliked.predicted_score, 3)
                self.assertAlmostEqual(liked.confidence, (2 / 7) * (1 / 6))
                self.assertEqual(liked.reason_signals[0].feature_type, kind)
                learned = next(p for p in self.profile.feature_preferences if
                               p.feature_type == kind and p.preference > 0)
                self.assertAlmostEqual(learned.preference, 2 / 6)
                self.assertEqual(learned.movie_count, 1)
                json.dumps(liked.to_dict())

    def test_missing_and_unknown_metadata_are_neutral(self):
        for movie in (MovieMetadata(3, "Missing"), MovieMetadata(
            4, "Unknown", directors=("Z",), runtime_minutes=60,
            release_year=1900, language="de",
        )):
            result = predict_one(self.profile, movie)
            self.assertEqual(result.predicted_score, 3)
            self.assertEqual(result.confidence, 0)
            self.assertEqual(result.reason_signals, ())

    def test_runtime_and_decade_boundaries(self):
        for runtime, expected in ((89.9, "under 90 min"), (90, "90–119 min"),
                                  (120, "120–149 min"), (150, "150+ min")):
            self.assertEqual(feature_values(MovieMetadata(3, "M", runtime_minutes=runtime), "runtime"), (expected,))
        for year, expected in ((1999, "1990s"), (2000, "2000s")):
            self.assertEqual(feature_values(MovieMetadata(3, "M", release_year=year), "release_era"), (expected,))

    def test_multiple_directors_share_credit_and_duplicates_do_not_add_evidence(self):
        movies = (replace(self.movies[0], directors=(" A ", "a", "C")), self.movies[1])
        profile = build_profile(1, self.ratings, movies)
        preferences = [p for p in profile.feature_preferences if p.feature_type == "director" and p.preference > 0]
        self.assertEqual(len(preferences), 2)
        self.assertTrue(all(p.movie_count == 1 and p.preference == 1 / 6 for p in preferences))
        single = predict_one(profile, MovieMetadata(3, "M", directors=("a",)))
        duplicate = predict_one(profile, MovieMetadata(3, "M", directors=("A", "a")))
        self.assertEqual(single.predicted_score, duplicate.predicted_score)
        partial = predict_one(profile, MovieMetadata(3, "M", directors=("a", "unknown")))
        self.assertAlmostEqual(partial.confidence, single.confidence / 2)

    def test_caps_apply_after_even_extreme_weights_and_final_score_is_bounded(self):
        config = ScoringConfig(**{f"{kind}_weight": 1e308 for kind in
                                  ("genre", "director", "runtime", "release_era", "language")})
        for movie, sign in ((self.movies[0], 1), (self.movies[1], -1)):
            result = predict_one(self.profile, movie, config, include_debug=True)
            self.assertEqual(result.debug.weighted_genre_adjustment, sign)
            for component in result.debug.feature_components:
                self.assertEqual(component.weighted_adjustment,
                                 sign * getattr(config, f"max_abs_{component.feature_type}_component"))
            self.assertTrue(0 <= result.predicted_score <= 5)

    def test_disabled_features_have_no_score_reasons_or_confidence(self):
        for suffix in ("weight", "component"):
            fields = {(f"{kind}_weight" if suffix == "weight" else f"max_abs_{kind}_component"): 0
                      for kind in ("genre", "director", "runtime", "release_era", "language")}
            result = predict_one(self.profile, self.movies[0], ScoringConfig(**fields))
            self.assertEqual((result.predicted_score, result.confidence, result.reason_signals), (3, 0, ()))

    def test_new_config_rejects_invalid_values(self):
        for kind in ("director", "runtime", "release_era", "language"):
            for field in (f"{kind}_weight", f"max_abs_{kind}_component"):
                for value in (-1, float("nan"), float("inf"), True, "1"):
                    with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                        ScoringConfig(**{field: value})

    def test_recency_weights_apply_to_new_features(self):
        ratings = (replace(self.ratings[0], timestamp=0), replace(self.ratings[1], timestamp=365 * 86400))
        profile = build_profile(1, ratings, self.movies, ProfileConfig())
        director = next(p for p in profile.feature_preferences if p.feature_value == "a")
        self.assertAlmostEqual(director.effective_evidence_count, 0.75)
        self.assertAlmostEqual(director.preference, 1.5 / 5.75)

    def test_batch_and_catalog_match_single_prediction(self):
        model = ContentModel(self.ratings, self.movies)
        expected = predict_one(self.profile, self.movies[0], include_debug=True)
        self.assertEqual(model.predict_one(1, 1, True), expected)
        self.assertEqual(predict_batch(self.profile, [self.movies[0]], include_debug=True), (expected,))

    def test_loader_accepts_enriched_records_and_original_movielens(self):
        plain = movie_metadata_from_record(dict(movieId=1, title="M (1995)", genres="Action"))
        self.assertEqual((plain.release_year, plain.directors, plain.language), (1995, (), None))
        enriched = movie_metadata_from_record(dict(movieId=1, title="M (1995)",
            directors="A|B", runtime_minutes="100", language="en", release_year="2001"))
        self.assertEqual((enriched.directors, enriched.runtime_minutes, enriched.language, enriched.release_year),
                         (("A", "B"), 100, "en", 2001))
        missing = movie_metadata_from_record(dict(movieId=1, title="M", directors=float("nan"),
            runtime_minutes="", language=float("nan"), release_year=""))
        self.assertEqual(missing.directors, ())
        self.assertIsNone(missing.runtime_minutes)
        self.assertIsNone(missing.language)
