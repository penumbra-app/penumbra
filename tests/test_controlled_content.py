import unittest
from dataclasses import replace

import numpy as np
from scipy import sparse

from src.content import ContentModel, MovieMetadata, ProfileConfig, TextConfig, UserRating
from src.content.experimental import (FieldTextIndex, combine, history_targets, personal_matrices,
                                     ridge_signal, separate_fields, shared_index, similarity_signal)
from src.content.text import movie_text
from src.evaluate_content import Case, candidate_from_dict
from src.experiment_content import Runner


class ControlledContentTests(unittest.TestCase):
    def setUp(self):
        self.movies = (
            MovieMetadata(1, "A", keywords=("space",), plot="astronaut journey"),
            MovieMetadata(2, "B", keywords=("wedding",), plot="romantic comedy"),
            MovieMetadata(3, "C", keywords=("space",), plot="galaxy journey"),
            MovieMetadata(4, "Empty"),
        )
        self.history = (UserRating(1, 1, 5, 100), UserRating(1, 2, 1, 100))
        self.config = TextConfig(weight=1)

    def test_personal_scorer_matches_public_api_and_signed_residuals(self):
        catalog = {m.movie_id: m for m in self.movies}
        h, c = personal_matrices(self.history, catalog, self.movies, self.config)
        r, w = history_targets(self.history, ProfileConfig())
        signal, mass = similarity_signal(h, c, r, w)
        result = combine(np.full(4, 3.), signal, self.config)
        public = ContentModel(self.history, self.movies, text_config=self.config)
        np.testing.assert_allclose(result, [p.predicted_score for p in public.predict((1,), (1, 2, 3, 4))])
        self.assertGreater(signal[0], 0)
        self.assertLess(signal[1], 0)
        self.assertEqual((signal[3], mass[3]), (0, 0))

    def test_catalog_index_uses_unrated_content_and_missing_text_is_zero(self):
        docs = sorted({movie_text(m, "combined") for m in self.movies})
        index = shared_index(self.movies, docs, self.config)
        self.assertEqual(index.for_movies([self.movies[3]]).nnz, 0)
        self.assertGreater(index.for_movies([self.movies[2]]).nnz, 0)
        # The index accepts metadata, no ratings, and contains each catalog text.
        np.testing.assert_allclose(index.for_movies([self.movies[2]]).toarray(),
                                   index.for_movies([self.movies[2], self.movies[1]])[:1].toarray())

    def test_field_blocks_have_expected_weighted_cosine_and_missing_fallback(self):
        plot = sparse.csr_matrix([[1., 0], [1., 0], [0., 0]])
        keywords = sparse.csr_matrix([[1., 0], [0., 1], [0., 1]])
        x = separate_fields(plot, keywords, .25)
        self.assertAlmostEqual((x @ x.T).toarray()[0, 1], .25)
        np.testing.assert_allclose(np.asarray(x.multiply(x).sum(axis=1)).ravel(), [1, 1, 1])

    def test_same_combined_text_in_different_fields_is_not_conflated(self):
        a, b = MovieMetadata(10, "A", plot="space"), MovieMetadata(11, "B", keywords=("space",))
        self.assertEqual(movie_text(a, "combined"), movie_text(b, "combined"))
        index = FieldTextIndex(sparse.eye(2, format="csr"), {("space", ""): 0, ("", "space"): 1})
        np.testing.assert_equal(index.for_movies([a, b]).toarray(), np.eye(2))

    def test_top_k_drops_weak_matches_without_looking_at_ratings(self):
        h, c = np.eye(3), np.array([[.8, .6, 0.]])
        residuals, weights = np.array([2., -2., 0.]), np.ones(3)
        score, mass = similarity_signal(h, c, residuals, weights, prior=5, top_k=1)
        self.assertAlmostEqual(score[0], 1.6 / 5.8)
        self.assertAlmostEqual(mass[0], .8)
        all_scores = similarity_signal(h, c, residuals, weights, top_k=10)
        normal = similarity_signal(h, c, residuals, weights)
        np.testing.assert_equal(all_scores, normal)

    def test_ridge_matches_closed_form_weighted_solution(self):
        h = np.array([[1., 0], [0., 1], [1., 1]])
        c = np.array([[1., .5], [0., 0]])
        residuals, weights, alpha = np.array([1., -1., 0.]), np.array([1., .5, 1.]), 2.
        expected = c @ np.linalg.solve(h.T @ (weights[:, None] * h) + alpha * np.eye(2),
                                       h.T @ (weights * residuals))
        for matrix in (h, sparse.csr_matrix(h)):
            np.testing.assert_allclose(ridge_signal(matrix, c, residuals, weights, alpha), expected, atol=1e-8)
        self.assertEqual(expected[1], 0)

    def test_holdout_rating_changes_do_not_change_predictions(self):
        candidate = candidate_from_dict({"name": "test", "profile": {}, "scoring": {},
                                         "text": {"weight": 1}})
        case = Case(1, self.history, (UserRating(1, 3, 5, 200), UserRating(1, 4, 1, 200)),
                    (UserRating(1, 3, 1, 300), UserRating(1, 4, 5, 300)), self.movies)
        modified = replace(case, validation=tuple(replace(r, rating=6. - r.rating) for r in case.validation),
                           test=tuple(replace(r, rating=6. - r.rating) for r in case.test))
        docs = sorted({movie_text(m, "combined") for m in self.movies})
        indexes = {"shared": shared_index(self.movies, docs, self.config)}
        for representation in ("personal_tfidf", "shared"):
            for scorer in ("similarity", "ridge"):
                config = {"representation": representation, "scorer": scorer, "alpha": 1., "top_k": 1}
                a = Runner((case,), candidate, indexes).score(config, "test")[2]
                b = Runner((modified,), candidate, indexes).score(config, "test")[2]
                np.testing.assert_equal(a, b)

    def test_other_user_ratings_do_not_enter_personal_targets(self):
        candidate = candidate_from_dict({"name": "test", "profile": {}, "scoring": {}, "text": {"weight": 1}})
        heldout = (UserRating(1, 3, 1, 200), UserRating(1, 4, 5, 200))
        a = Case(1, self.history, heldout, heldout, self.movies)
        b = Case(2, tuple(replace(r, user_id=2, rating=6.-r.rating) for r in self.history),
                 tuple(replace(r, user_id=2) for r in heldout),
                 tuple(replace(r, user_id=2) for r in heldout), self.movies)
        docs = sorted({movie_text(m, "combined") for m in self.movies})
        indexes = {"shared": shared_index(self.movies, docs, self.config)}
        config = {"representation": "shared", "scorer": "similarity"}
        alone = Runner((a,), candidate, indexes).score(config, "test")[2][0]
        together = Runner((a, b), candidate, indexes).score(config, "test")[2][0]
        np.testing.assert_equal(alone, together)

    def test_negative_cosine_and_zero_vectors_provide_no_evidence(self):
        score, mass = similarity_signal(np.eye(2), np.array([[-1., 0.], [0., 0.]]),
                                        np.array([2., -2.]), np.ones(2))
        np.testing.assert_equal(score, [0, 0])
        np.testing.assert_equal(mass, [0, 0])


if __name__ == "__main__":
    unittest.main()
