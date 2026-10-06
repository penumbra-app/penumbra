import json
import unittest
from dataclasses import replace

from src.content import (
    ContentModel, MovieMetadata, ProfileConfig, ScoringConfig, TextConfig,
    TextProfile, UserRating, build_profile, predict_batch,
    load_selected_model,
)
from src.content.text import movie_text
from src.data_processing.movielens import movie_metadata_from_record
from src.data_processing.text_metadata import with_keywords_as_of


class TextTests(unittest.TestCase):
    def setUp(self):
        self.movies = (
            MovieMetadata(1, "Liked", keywords=("space exploration",)),
            MovieMetadata(2, "Disliked", keywords=("romantic wedding",)),
            MovieMetadata(3, "New space", keywords=("space exploration",)),
            MovieMetadata(4, "New romance", keywords=("romantic wedding",)),
            MovieMetadata(5, "No text"),
            MovieMetadata(6, "Unseen words", keywords=("zebra acrobat",)),
        )
        self.ratings = (UserRating(1, 1, 5, 1), UserRating(1, 2, 1, 1))
        self.config = TextConfig()
        self.model = ContentModel(self.ratings, self.movies, text_config=self.config)

    def test_positive_and_negative_residuals_match_hand_calculation(self):
        positive = self.model.predict_one(1, 3, True)
        negative = self.model.predict_one(1, 4, True)
        self.assertAlmostEqual(positive.predicted_score, 3 + .5 * 2 / 6)
        self.assertAlmostEqual(negative.predicted_score, 3 - .5 * 2 / 6)
        self.assertAlmostEqual(positive.confidence, (2 / 7) * (1 / 6))
        self.assertEqual(positive.reason_signals[0].feature_type, "text")
        self.assertAlmostEqual(positive.reason_signals[0].evidence_count, 1)
        self.assertIsNone(self.model.predict_one(1, 3).debug)

    def test_no_text_or_out_of_vocabulary_is_exactly_neutral(self):
        core = ContentModel(self.ratings, self.movies)
        for movie_id in (5, 6):
            self.assertEqual(self.model.predict_one(1, movie_id, True),
                             core.predict_one(1, movie_id, True))

    def test_fit_excludes_candidate_vocabulary_and_candidate_order(self):
        profile = self.model.build_text_profile(1)
        vocabulary = dict(profile.vectorizer.vocabulary_)
        idf = profile.vectorizer.idf_.copy()
        self.assertNotIn("zebra", vocabulary)
        first = self.model.predict((1,), (6, 3, 3))
        second = self.model.predict((1,), (3, 6))
        self.assertEqual(first[1], second[0])
        self.assertEqual(first[1], first[2])
        self.assertEqual(vocabulary, profile.vectorizer.vocabulary_)
        self.assertTrue((idf == profile.vectorizer.idf_).all())
        self.assertIs(profile, self.model.build_text_profile(1))

    def test_disabled_text_matches_core_including_debug(self):
        core = ContentModel(self.ratings, self.movies)
        for config in (TextConfig(weight=0), TextConfig(max_adjustment=0)):
            disabled = ContentModel(self.ratings, self.movies, text_config=config)
            self.assertEqual(disabled.predict((1,), (3, 4), True), core.predict((1,), (3, 4), True))

    def test_empty_stopword_and_pruned_training_text_are_neutral(self):
        for text in ("", "the and is", "!!!", "a"):
            catalog = tuple(replace(m, keywords=(text,)) for m in self.movies)
            model = ContentModel(self.ratings, catalog, text_config=self.config)
            self.assertEqual(model.predict_one(1, 3).predicted_score, 3)
        for min_df in (2, 10):
            model = ContentModel(self.ratings, self.movies, text_config=TextConfig(min_df=min_df))
            self.assertEqual(model.predict_one(1, 3).predicted_score, 3)

    def test_cold_start_single_rating_and_missing_training_movie(self):
        cold = self.model.predict_one(42, 3)
        self.assertEqual((cold.predicted_score, cold.confidence, cold.reason_signals), (2.5, 0, ()))
        single = ContentModel(self.ratings[:1], self.movies, text_config=self.config)
        self.assertEqual(single.predict_one(1, 3).predicted_score, 5)
        self.assertEqual(single.predict_one(1, 3).reason_signals, ())
        missing = ContentModel(self.ratings, self.movies[2:], text_config=self.config)
        self.assertEqual(missing.predict_one(1, 3).predicted_score, 3)

    def test_keywords_are_deduplicated_ignoring_case(self):
        movies = tuple(replace(m, keywords=m.keywords * 4 + tuple(k.upper() for k in m.keywords))
                       for m in self.movies)
        duplicated = ContentModel(self.ratings, movies, text_config=self.config)
        self.assertEqual(duplicated.predict((1,), (3, 4)), self.model.predict((1,), (3, 4)))

    def test_plot_and_combined_sources_work_without_title_substitution(self):
        movies = tuple(replace(m, plot=" ".join(m.keywords), keywords=()) for m in self.movies)
        for source in ("plot", "combined"):
            model = ContentModel(self.ratings, movies, text_config=TextConfig(source=source))
            self.assertGreater(model.predict_one(1, 3).predicted_score, model.predict_one(1, 4).predicted_score)
        model = ContentModel(self.ratings, movies, text_config=TextConfig(source="keywords"))
        self.assertEqual(model.predict_one(1, 3).predicted_score, 3)
        self.assertEqual(movie_text(MovieMetadata(8, "space exploration", ("Sci-Fi",)), "combined"), "")

    def test_caps_extreme_weights_and_reasons_reconcile(self):
        for weight in (.25, 1, 1e308):
            model = ContentModel(self.ratings, self.movies, text_config=TextConfig(weight=weight))
            for result in model.predict((1,), (3, 4), True):
                self.assertLessEqual(abs(result.debug.unclamped_score - 3), .5)
                self.assertAlmostEqual(sum(r.score_contribution for r in result.reason_signals),
                                       result.debug.unclamped_score - 3)
                json.dumps(result.to_dict(), allow_nan=False)

    def test_blends_before_final_clamp(self):
        # Strong genre lift is clipped only after the negative text adjustment.
        movies = (replace(self.movies[0], genres=("Action",)), self.movies[1],
                  replace(self.movies[3], genres=("Action",)))
        ratings = (replace(self.ratings[0], rating=5), replace(self.ratings[1], rating=4))
        model = ContentModel(ratings, movies,
                             ScoringConfig(genre_weight=10), ProfileConfig(regularization_strength=0),
                             TextConfig(weight=1, regularization_strength=0, max_adjustment=1))
        result = model.predict_one(1, 4, True)
        self.assertAlmostEqual(result.debug.unclamped_score, 5)
        self.assertAlmostEqual(result.predicted_score, 5)

    def test_recency_reduces_old_text_evidence(self):
        ratings = (replace(self.ratings[0], timestamp=0),
                   replace(self.ratings[1], timestamp=365 * 86400))
        model = ContentModel(ratings, self.movies, text_config=self.config)
        self.assertAlmostEqual(model.predict_one(1, 3).reason_signals[0].evidence_count, .75)

    def test_text_confidence_respects_configured_prior(self):
        model = ContentModel(self.ratings, self.movies, ScoringConfig(confidence_prior_count=1),
                             text_config=self.config)
        self.assertAlmostEqual(model.predict_one(1, 3).confidence, (2 / 3) * (1 / 2))

    def test_public_batch_and_recommendation_paths_match(self):
        profile = self.model.build_profile(1)
        standalone = predict_batch(profile, self.movies[2:], text_profile=self.model.build_text_profile(1))
        self.assertEqual(standalone, self.model.predict((1,), (3, 4, 5, 6)))
        self.assertEqual(tuple(r.movie_id for r in self.model.recommend(1)), (3, 5, 6, 4))
        self.assertEqual(self.model.recommend(1, 1), (standalone[0],))
        self.assertEqual(self.model.predict((1,), ()), ())
        with self.assertRaises(ValueError):
            predict_batch(build_profile(2, (), self.movies), self.movies,
                          text_profile=self.model.build_text_profile(1))

    def test_lsa_is_deterministic_finite_and_handles_tiny_corpus(self):
        for count in (1, 2, 4):
            ratings = tuple(UserRating(1, m.movie_id, float(1 + i), i) for i, m in enumerate(self.movies[:count]))
            a = ContentModel(ratings, self.movies, text_config=TextConfig(representation="lsa", latent_dimensions=2))
            b = ContentModel(ratings, self.movies, text_config=TextConfig(representation="lsa", latent_dimensions=2))
            self.assertEqual(a.predict((1,), (3, 4, 5, 6)), b.predict((1,), (3, 4, 5, 6)))
            self.assertEqual(a.predict_one(1, 6).predicted_score, sum(r.rating for r in ratings) / count)

    def test_metadata_loader_and_invalid_settings(self):
        movie = movie_metadata_from_record(dict(movieId=1, title="M", keywords="Space|space|Robot", plot="An astronaut."))
        self.assertEqual(movie.keywords, ("Space", "Robot"))
        self.assertEqual(movie.plot, "An astronaut.")
        for value in (None, "", float("nan")):
            movie = movie_metadata_from_record(dict(movieId=1, title="M", keywords=value, plot=value))
            self.assertEqual((movie.keywords, movie.plot), ((), None))
        for kwargs in ({"keywords": ["space"]}, {"keywords": (3,)}, {"plot": 5}):
            with self.assertRaises(ValueError):
                MovieMetadata(1, "M", **kwargs)
        for kwargs in ({"source": "title"}, {"representation": "unknown"}, {"weight": True},
                       {"weight": -1}, {"weight": float("inf")}, {"max_adjustment": 6},
                       {"regularization_strength": float("nan")}, {"ngram_max": 3},
                       {"min_df": 0}, {"max_features": True}, {"latent_dimensions": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TextConfig(**kwargs)

    def test_tag_availability_and_other_user_filter(self):
        tags = [dict(movieId=1, userId=2, tag=" SPACE ", timestamp=9),
                dict(movieId=1, userId=2, tag="space", timestamp=10),
                dict(movieId=1, userId=1, tag="own preference", timestamp=9),
                dict(movieId=2, userId=2, tag="future", timestamp=11)]
        movies = with_keywords_as_of((MovieMetadata(1, "M"), MovieMetadata(2, "N")), tags, 10, 1)
        self.assertEqual(movies[0].keywords, ("space",))
        self.assertEqual(movies[1].keywords, ())

    def test_duplicate_and_mixed_user_ratings_are_rejected(self):
        for ratings in (self.ratings + self.ratings[:1], self.ratings + (UserRating(2, 3, 5),)):
            with self.assertRaises(ValueError):
                TextProfile(ratings, {m.movie_id: m for m in self.movies})

    def test_selected_preset_matches_frozen_validation_winner(self):
        from pathlib import Path
        from src.content.selection import DEFAULT_PRESET

        report = Path(__file__).resolve().parents[1] / "reports/week6/selected_config.json"
        self.assertEqual(json.loads(DEFAULT_PRESET.read_text()), json.loads(report.read_text()))
        model = load_selected_model(self.ratings, self.movies)
        self.assertGreater(model.predict_one(1, 3).predicted_score, model.predict_one(1, 4).predicted_score)
