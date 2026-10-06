import unittest
from dataclasses import replace

import numpy as np

from src.content import MovieMetadata, UserRating, TextConfig
from src.evaluate_content import (
    Candidate, blend, core_scores, paired_bootstrap, prepare_cases, select_best,
    summarize, temporal_split, text_evidence, user_metrics,
)


class EvaluationTests(unittest.TestCase):
    def test_split_keeps_timestamp_groups_and_orders_chronologically(self):
        ratings = tuple(UserRating(1, i + 1, 3, i // 3) for i in range(30))
        a, b, c = temporal_split(reversed(ratings))
        self.assertEqual((len(a), len(b), len(c)), (18, 6, 6))
        self.assertLess(max(r.timestamp for r in a), min(r.timestamp for r in b))
        self.assertLess(max(r.timestamp for r in b), min(r.timestamp for r in c))
        for invalid in ((), (UserRating(1, 1, 3),), (ratings[0], ratings[0])):
            with self.assertRaises(ValueError):
                temporal_split(invalid)

    def test_timestamp_groups_crossing_boundaries_are_not_divided(self):
        ratings = tuple(UserRating(1, i + 1, 3, i // 7) for i in range(30))
        a, b, c = temporal_split(ratings)
        self.assertEqual((len(a), len(b), len(c)), (21, 7, 2))
        movies = tuple(MovieMetadata(i + 1, "M") for i in range(30))
        same_time = tuple(replace(r, timestamp=1) for r in ratings)
        self.assertEqual(prepare_cases(same_time, movies, ()), ((), 1))

    def test_pairwise_ties_equal_ratings_and_ndcg(self):
        for scores, expected in (([1, 2, 3], 1), ([3, 2, 1], 0), ([2, 2, 2], .5)):
            self.assertEqual(user_metrics([1, 2, 3], scores)["pairwise_accuracy"], expected)
        self.assertAlmostEqual(user_metrics([1, 2, 3], [1, 2, 3])["ndcg_at_10"], 1)
        self.assertIsNone(user_metrics([3, 3], [1, 2])["pairwise_accuracy"])
        self.assertEqual(user_metrics([1, 1, 2], [1, 1, 2])["pairs"], 2)
        # Tie-averaged NDCG must not depend on input ordering.
        self.assertAlmostEqual(user_metrics([1, 2, 5], [3, 3, 3])["ndcg_at_10"],
                               user_metrics([5, 1, 2], [3, 3, 3])["ndcg_at_10"])

    def test_metrics_are_user_macro_and_bootstrap_is_paired(self):
        a, b = user_metrics([1, 2], [1, 2]), user_metrics([1, 2, 3, 4], [4, 3, 2, 1])
        self.assertEqual(summarize([a, b])["pairwise_accuracy"], .5)
        self.assertEqual(paired_bootstrap([a, b], [a, b])["ci95"], [0, 0])

    def test_selection_uses_validation_primary_metric_then_rmse(self):
        rows = [dict(candidate={"name": "a"}, metrics=dict(pairwise_accuracy=.6, rmse=1)),
                dict(candidate={"name": "b"}, metrics=dict(pairwise_accuracy=.7, rmse=2)),
                dict(candidate={"name": "c"}, metrics=dict(pairwise_accuracy=.7, rmse=1))]
        self.assertEqual(select_best(rows)["candidate"]["name"], "c")

    def test_holdout_labels_and_future_tags_cannot_change_predictions(self):
        ratings = tuple(UserRating(1, i + 1, 1. + i % 5, i + 1) for i in range(30))
        movies = tuple(MovieMetadata(i + 1, "M", ("Action",), keywords=("space" if i % 2 else "wedding",))
                       for i in range(30))
        cases, _ = prepare_cases(ratings, movies, ())
        changed = tuple(replace(r, rating=5 if r.rating < 3 else 1) if r.timestamp > 18 else r for r in ratings)
        tags = [dict(userId=2, movieId=1, tag="future revelation", timestamp=19)]
        other, _ = prepare_cases(changed, movies, tags)
        candidate = Candidate("text", text=TextConfig())
        for phase in ("validation", "test"):
            base_a, base_b = core_scores(cases, candidate, phase), core_scores(other, candidate, phase)
            text_a, text_b = text_evidence(cases, candidate, phase), text_evidence(other, candidate, phase)
            np.testing.assert_array_equal(blend(base_a, text_a, candidate.text)[0],
                                          blend(base_b, text_b, candidate.text)[0])

    def test_batched_evaluation_matches_public_model_when_text_changes_ranking(self):
        from src.content import ContentModel

        ratings = tuple(UserRating(1, i + 1, 5 if i % 2 else 1, i + 1) for i in range(30))
        movies = tuple(MovieMetadata(i + 1, "M", keywords=("space" if i % 2 else "wedding",))
                       for i in range(30))
        cases, _ = prepare_cases(ratings, movies, ())
        for representation in ("tfidf", "lsa"):
            candidate = Candidate("text", text=TextConfig(representation=representation))
            expected = blend(core_scores(cases, candidate, "test"),
                             text_evidence(cases, candidate, "test"), candidate.text)[0]
            case = cases[0]
            model = ContentModel(case.history, case.movies, candidate.scoring, candidate.profile, candidate.text)
            actual = [r.predicted_score for r in model.predict((1,), tuple(r.movie_id for r in case.test))]
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            self.assertGreater(actual[1], actual[0])
