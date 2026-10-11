import json
import unittest
from dataclasses import replace

from src.content import (
    ContentModel, MovieMetadata, ProfileConfig, ScoringConfig, UserRating,
    build_profile, predict_one,
)
from src.data_processing.movielens import movie_metadata_from_record


class Week4Tests(unittest.TestCase):
    def setUp(self):
        self.movies = (MovieMetadata(1, 'Liked', cast=('A',)),
                       MovieMetadata(2, 'Disliked', cast=('B',)))
        self.ratings = (UserRating(1, 1, 5), UserRating(1, 2, 1))
        self.profile = build_profile(1, self.ratings, self.movies)

    def test_cast_learns_both_directions_and_has_small_cap(self):
        for actor, sign in (('a', 1), ('b', -1)):
            result = predict_one(self.profile, MovieMetadata(3, 'New', cast=(actor,)),
                                 ScoringConfig(cast_weight=1e308), True)
            self.assertAlmostEqual(result.predicted_score, 3 + sign * 0.25)
            self.assertEqual(result.reason_signals[0].feature_type, 'cast')
            self.assertGreater(result.confidence, 0)

    def test_large_training_cast_shares_credit(self):
        large = replace(self.movies[0], cast=tuple(f'Actor {i}' for i in range(100)))
        profile = build_profile(1, self.ratings, (large, self.movies[1]))
        positive = [p for p in profile.feature_preferences if p.preference > 0]
        self.assertEqual(len(positive), 100)
        self.assertAlmostEqual(sum(p.preference for p in positive), 2 / 6)
        result = predict_one(profile, replace(large, movie_id=3))
        single = predict_one(self.profile, MovieMetadata(3, 'New', cast=('a',)))
        self.assertLess(result.predicted_score, single.predicted_score)
        self.assertLessEqual(result.confidence, single.confidence)

    def test_duplicate_and_unknown_cast_do_not_inflate_score_or_support(self):
        single = predict_one(self.profile, MovieMetadata(3, 'New', cast=('a',)))
        duplicate = predict_one(self.profile, MovieMetadata(3, 'New', cast=(' A ', 'a', '')))
        self.assertEqual(single, duplicate)
        large = predict_one(self.profile, MovieMetadata(
            3, 'New', cast=('a',) + tuple(f'Unknown {i}' for i in range(99))))
        self.assertAlmostEqual(large.predicted_score - 3, (2 / 6) / 100 * 0.25)
        self.assertLess(large.predicted_score, single.predicted_score)
        self.assertAlmostEqual(large.confidence, single.confidence / 100)

    def test_disabled_cancelled_missing_and_unknown_cast_have_no_reasons(self):
        for cast, config in ((('a',), ScoringConfig(cast_weight=0)),
                             (('a',), ScoringConfig(max_abs_cast_component=0)),
                             (('a', 'b'), ScoringConfig()),
                             ((), ScoringConfig()), (('z',), ScoringConfig())):
            result = predict_one(self.profile, MovieMetadata(3, 'New', cast=cast), config)
            self.assertEqual(result.predicted_score, 3)
            self.assertEqual(result.reason_signals, ())

    def test_reasons_reconcile_with_score_after_weighting_and_caps(self):
        movies = (replace(self.movies[0], genres=('Action',), language='en'),
                  replace(self.movies[1], genres=('Drama',), language='fr'))
        profile = build_profile(1, self.ratings, movies, ProfileConfig(regularization_strength=0))
        profile = replace(profile, baseline=4.9)
        candidate = MovieMetadata(3, 'New', ('Action', 'Unknown'), cast=('a', 'z'), language='en')
        for weight in (0, 0.1, 1, 1e308):
            result = predict_one(profile, candidate, ScoringConfig(cast_weight=weight), True)
            self.assertAlmostEqual(sum(r.score_contribution for r in result.reason_signals),
                                   result.debug.unclamped_score - profile.baseline)
            for r in result.reason_signals:
                self.assertGreater(r.evidence_count, 0)
                self.assertAlmostEqual(r.strength, r.score_contribution / 5)
            self.assertEqual(result.predicted_score, min(5, result.debug.unclamped_score))
            json.dumps(result.to_dict())

    def test_cold_start_all_apis_and_custom_default(self):
        config = ProfileConfig(cold_start_score=3)
        model = ContentModel((), self.movies, profile_config=config)
        profile = model.build_profile(42)
        self.assertEqual(profile, build_profile(42, (), self.movies, config))
        self.assertIs(profile, model.build_profile(42))
        self.assertEqual(model.unseen_movie_ids(42), (1, 2))
        for result in (*model.predict((42,), (1, 2)), *model.recommend(42)):
            self.assertEqual((result.predicted_score, result.confidence, result.reason_signals), (3, 0, ()))
        for invalid in (0, -1, True, '42'):
            with self.assertRaises(ValueError):
                model.recommend(invalid)
        for invalid in (-1, 6, float('nan'), True):
            with self.assertRaises(ValueError):
                ProfileConfig(cold_start_score=invalid)

    def test_missing_training_metadata_still_counts_toward_baseline(self):
        profile = build_profile(1, self.ratings, ())
        result = predict_one(profile, MovieMetadata(3, 'No metadata'))
        self.assertEqual(profile.metadata.ratings_without_movie_metadata, 2)
        self.assertEqual((result.predicted_score, result.confidence, result.reason_signals), (3, 0, ()))
        one_rating = build_profile(1, self.ratings[:1], self.movies)
        result = predict_one(one_rating, MovieMetadata(3, 'New', cast=('a',)))
        self.assertEqual(result.predicted_score, 5)
        self.assertEqual(result.reason_signals, ())
        self.assertLess(result.confidence, 0.1)

    def test_recommend_ranks_entire_catalog_and_excludes_seen(self):
        candidates = (MovieMetadata(3, 'Bad', cast=('b',)),
                      MovieMetadata(4, 'Good', cast=('a',)),
                      MovieMetadata(5, 'Also good', cast=('a',)))
        model = ContentModel(self.ratings, self.movies + candidates)
        self.assertEqual(tuple(r.movie_id for r in model.recommend(1)), (4, 5, 3))
        self.assertEqual(model.recommend(1, 1)[0].movie_id, 4)
        self.assertEqual(model.recommend(1, 0), ())
        self.assertEqual(ContentModel((), ()).recommend(1), ())
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                model.recommend(1, invalid)

    def test_cast_loader_validation_and_missing_values(self):
        for value, expected in ((' A |a|B', ('A', 'B')), (['A', 'B'], ('A', 'B')),
                                (None, ()), ('', ()), (float('nan'), ())):
            movie = movie_metadata_from_record(dict(movieId=1, title='M', cast=value))
            self.assertEqual(movie.cast, expected)
        for invalid in ('A', [1], (None,)):
            with self.assertRaises(ValueError):
                MovieMetadata(1, 'M', cast=invalid)
