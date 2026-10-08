import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.content import ContentModel, MovieMetadata, TextConfig, UserRating
from src.evaluate_content import Candidate
from src.evaluate_catalog import case_catalog, confirm, score_models
from src.evaluation.catalog import (CatalogCase, aggregate, candidate_ids, development_cases,
                                    first_observed, full_catalog_metrics, future_cases, paired_inference)
from src.evaluation.confirmation import claim_once, register_plan, verify_plan


class CatalogMetricTests(unittest.TestCase):
    def test_known_positive_recall_ndcg_and_unjudged_coverage(self):
        outcomes = (UserRating(1, 3, 5, 20), UserRating(1, 4, 4, 20), UserRating(1, 6, 5, 20))
        metrics, top = full_catalog_metrics([2, 3, 4, 5], [0, 3, 2, 4], outcomes, ks=(2,))
        self.assertEqual(top, [5, 3])
        self.assertEqual(metrics['recall_at_2'], .5)
        self.assertAlmostEqual(metrics['ndcg_at_2'], (1/np.log2(3))/(1+1/np.log2(3)))
        self.assertEqual(metrics['recall_all_observed_positives_at_2'], 1/3)
        self.assertEqual(metrics['judged_fraction_at_2'], .5)
        self.assertEqual(metrics['positives_total'], 3)
        self.assertEqual(metrics['positives_eligible'], 2)

    def test_zero_positive_users_are_not_silently_counted_as_failures(self):
        row, _ = full_catalog_metrics([1], [5], (UserRating(1, 1, 1, 20),))
        self.assertIsNone(row['ndcg_at_10'])
        result = aggregate([row])
        self.assertEqual(result['users_without_eligible_positives'], 1)
        self.assertIsNone(result['recall_at_10'])
        empty, _ = full_catalog_metrics([], [], ())
        self.assertEqual(aggregate([empty])['users_without_outcomes'], 1)

    def test_ties_are_independent_of_input_order(self):
        outcomes = (UserRating(1, 2, 5, 20),)
        a = full_catalog_metrics([3, 2, 1], [4, 4, 4], outcomes)
        b = full_catalog_metrics([2, 1, 3], [4, 4, 4], outcomes)
        self.assertEqual(a, b)
        self.assertEqual(a[1], [1, 2, 3])

    def test_missing_nonfinite_and_duplicate_predictions_rejected(self):
        for ids, scores in (([1, 2], [1]), ([1], [float('nan')]), ([1, 1], [3, 4])):
            with self.assertRaises(ValueError):
                full_catalog_metrics(ids, scores, ())

    def test_large_catalog_distractors_change_metrics(self):
        outcomes = (UserRating(1, 1, 5, 20),)
        small, _ = full_catalog_metrics([1], [1], outcomes, ks=(1,))
        full, _ = full_catalog_metrics([1, 2], [1, 5], outcomes, ks=(1,))
        self.assertEqual((small['recall_at_1'], full['recall_at_1']), (1, 0))


class CatalogProtocolTests(unittest.TestCase):
    def setUp(self):
        self.movies = (MovieMetadata(1, 'A', keywords=('space',)),
                       MovieMetadata(2, 'B', keywords=('romance',)),
                       MovieMetadata(3, 'C', keywords=('space',)),
                       MovieMetadata(4, 'D', keywords=('future',)))
        self.history = (UserRating(1, 1, 5, 5), UserRating(1, 2, 1, 10))
        self.case = CatalogCase(1, 10, self.history, (UserRating(1, 3, 5, 20),))

    def test_candidates_exclude_history_and_not_yet_observed_movies(self):
        availability = first_observed((*self.history, UserRating(2, 3, 1, 9), UserRating(2, 4, 5, 11)))
        self.assertEqual(candidate_ids(self.case, availability, {1, 2, 3, 4}), (3,))

    def test_development_split_preserves_timestamp_groups(self):
        ratings = tuple(UserRating(1, i, 4, i//3) for i in range(1, 31))
        cases, _ = development_cases(ratings)
        self.assertTrue(all(r.timestamp <= cases[0].cutoff for r in cases[0].history))
        self.assertTrue(all(r.timestamp > cases[0].cutoff for r in cases[0].outcomes))

    def test_future_and_self_authored_tags_excluded(self):
        tags = [dict(userId=2, movieId=3, tag='allowed', timestamp=10),
                dict(userId=2, movieId=3, tag='future', timestamp=11),
                dict(userId=1, movieId=3, tag='own', timestamp=5)]
        catalog = case_catalog(self.case, self.movies, {1: 5, 2: 5, 3: 5, 4: 11}, tags)
        self.assertEqual({m.movie_id for m in catalog}, {1, 2, 3})
        self.assertEqual(next(m for m in catalog if m.movie_id==3).keywords, ('allowed', 'space'))

    def test_full_catalog_scores_match_serving_and_do_not_use_outcomes(self):
        preset = Candidate('test', text=TextConfig())
        other = CatalogCase(1, 10, self.history, (UserRating(1, 3, 1, 20),))
        actual = score_models(self.case, self.movies, (3, 4), preset)
        changed = score_models(other, self.movies, (3, 4), preset)
        np.testing.assert_array_equal(actual['selected'], changed['selected'])
        model = ContentModel(self.history, self.movies, preset.scoring, preset.profile, preset.text)
        np.testing.assert_allclose(actual['selected'], [p.predicted_score for p in model.predict((1,), (3, 4))])
        core = ContentModel(self.history, self.movies, preset.scoring, preset.profile)
        np.testing.assert_allclose(actual['structured_core'], [p.predicted_score for p in core.predict((1,), (3, 4))])

    def test_future_confirmation_excludes_old_repeated_and_out_of_window_events(self):
        for outcomes in ((UserRating(1, 3, 4, 10),), (UserRating(1, 3, 4, 31),),
                         (UserRating(1, 1, 4, 20),)):
            with self.assertRaises(ValueError):
                future_cases(self.history, outcomes, 10, 30, [1])
        cases, _ = future_cases(self.history, (), 10, 30, [1, 2])
        self.assertEqual(len(cases), 2)  # Do not hide users without observations.
        self.assertTrue(all(not c.outcomes for c in cases))

    def test_bootstrap_aligns_user_ids_and_rejects_different_cohorts(self):
        a = {2: {'ndcg': .2}, 1: {'ndcg': .5}}
        b = {1: {'ndcg': .6}, 2: {'ndcg': .3}}
        stats = paired_inference(a, b, 'ndcg', repeats=100)
        self.assertAlmostEqual(stats['mean_delta'], .1)
        np.testing.assert_allclose(stats['ci95'], [.1, .1])
        with self.assertRaises(ValueError):
            paired_inference(a, {3: {'ndcg': .2}}, 'ndcg')
        with self.assertRaises(ValueError):
            paired_inference(a, {1: {'ndcg': None}, 2: {'ndcg': .2}}, 'ndcg')


class ConfirmationGuardTests(unittest.TestCase):
    def test_confirmation_cli_refuses_early_labels_and_completes_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'ratings.csv').write_text('userId,movieId,rating,timestamp\n' +
                                           ''.join(f'1,{i},4,{i}\n' for i in range(1, 6)))
            (root/'movies.csv').write_text('movieId,title,genres,keywords\n' +
                                          ''.join(f'{i},Movie {i},Action,space\n' for i in range(1, 8)))
            (root/'tags.csv').write_text('userId,movieId,tag,timestamp\n')
            (root/'preset.json').write_text(json.dumps({'schema_version': 1, 'candidate': Candidate('test', text=TextConfig()).to_dict()}))
            (root/'outcomes.csv').write_text('userId,movieId,rating,timestamp\n1,6,5,101\n1,7,1,102\n')
            register_plan(root/'movies.csv', root/'ratings.csv', root/'tags.csv', root/'preset.json',
                          root/'plan.json', days=1, now=100)
            args = SimpleNamespace(plan=root/'plan.json', outcomes=str(root/'outcomes.csv'),
                                   output=root/'output', state_dir=root/'state')
            with patch('src.evaluation.confirmation.time.time', return_value=101), \
                 patch('src.evaluate_catalog.read_csv', side_effect=AssertionError('Outcome opened early')):
                with self.assertRaisesRegex(ValueError, 'still open'):
                    confirm(args)
            with patch('src.evaluation.confirmation.time.time', return_value=86500), patch('builtins.print'):
                report = confirm(args)
                self.assertEqual(report['protocol']['mode'], 'future_confirmation')
                self.assertEqual(report['metrics']['selected']['users'], 1)
                self.assertTrue((root/'output/results.json').is_file())
                with self.assertRaises(FileExistsError):
                    confirm(args)

    def test_frozen_files_source_window_and_single_open_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ratings = root/'ratings.csv'
            ratings.write_text('userId,movieId,rating,timestamp\n1,1,4,10\n')
            for name in ('movies.csv', 'tags.csv', 'preset.json'):
                (root/name).write_text('{}')
            with patch('src.evaluation.confirmation.source_hashes', return_value={'source': 'a'}):
                plan = register_plan(root/'movies.csv', ratings, root/'tags.csv', root/'preset.json',
                                     root/'plan.json', days=1, now=100)
                with self.assertRaises(ValueError):
                    verify_plan(plan, now=101)
                verify_plan(plan, now=86500)
                state = claim_once(plan, root/'state')
                self.assertTrue(state.is_file())
                with self.assertRaises(FileExistsError):
                    claim_once(plan, root/'state')
                (root/'tags.csv').write_text('changed')
                with self.assertRaises(ValueError):
                    verify_plan(plan, now=86500)
                (root/'tags.csv').write_text('{}')
            with patch('src.evaluation.confirmation.source_hashes', return_value={'source': 'b'}):
                with self.assertRaises(ValueError):
                    verify_plan(plan, now=86500)
            tampered = {**plan, 'outcome_end': 1}
            with self.assertRaises(ValueError):
                verify_plan(tampered, now=86500)


if __name__ == '__main__':
    unittest.main()
