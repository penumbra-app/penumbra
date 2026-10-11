import copy
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.collaborative.matrix_factorization import BiasedMatrixFactorization
from src.collaborative.pairwise_factorization import PairwiseMatrixFactorization
from src.collaborative.evaluation.pairwise_experiment import run_experiment, paired_interval
from src.collaborative.evaluation.benchmark import shared_split


def ratings():
    return pd.DataFrame([dict(userId=u, movieId=i, rating=float(5 if i % 2 else 1), timestamp=i)
                         for u in (1, 2, 3) for i in range(1, 16)])


def test_contract_reproducibility_and_roundtrip(tmp_path):
    kwargs = dict(baseline_epochs=1, n_epochs=1)
    model = PairwiseMatrixFactorization(**kwargs).fit(ratings())
    second = PairwiseMatrixFactorization(**kwargs).fit(ratings())
    users, movies = [1, 1, 999, 999, 1], [1, 999, 1, 999, 1]
    first = model.predict(users, movies)
    assert first.columns.tolist() == ['user_id', 'movie_id', 'predicted_score', 'confidence']
    assert first.user_id.tolist() == users
    assert first.movie_id.tolist() == movies
    assert first.predicted_score.between(.5, 5).all()
    assert first.confidence.between(0, 1).all()
    assert first.iloc[1].confidence == .1
    assert first.iloc[3].confidence == 0
    assert model.predict([], []).empty
    pd.testing.assert_frame_equal(first, second.predict(users, movies))
    target = tmp_path / 'pairwise.joblib'
    model.save(target)
    pd.testing.assert_frame_equal(first, PairwiseMatrixFactorization.load(target).predict(users, movies))
    with pytest.raises(ValueError):
        model.predict([1], [1, 2])
    with pytest.raises(ValueError):
        model.predict([-1], [1])


def test_pairwise_updates_improve_observed_order_without_mutating_baseline():
    rows = pd.DataFrame({'userId': [1, 1], 'movieId': [1, 2], 'rating': [5., 1.]})
    baseline = BiasedMatrixFactorization(n_epochs=1).fit(rows)
    original = copy.deepcopy(baseline.weights_)
    control = PairwiseMatrixFactorization(rank_weight=0, n_epochs=1)._fit_from_baseline(rows, baseline)
    ranked = PairwiseMatrixFactorization(rank_weight=5, n_epochs=1)._fit_from_baseline(rows, baseline)
    def margin(m):
        p = m.predict([1, 1], [1, 2]).predicted_score
        return p.iloc[0] - p.iloc[1]
    assert margin(ranked) > margin(control)
    np.testing.assert_array_equal(original.user_factors, baseline.weights_.user_factors)
    np.testing.assert_array_equal(original.movie_biases, baseline.weights_.movie_biases)
    assert ranked.training_history_[0]['sampled_pairs'] == 2


def test_equal_rating_history_has_no_fabricated_pairs():
    rows = ratings().assign(rating=3.)
    model = PairwiseMatrixFactorization(baseline_epochs=1, n_epochs=1).fit(rows)
    assert model.training_history_[0]['sampled_pairs'] == 0
    assert model.training_history_[0]['online_pairwise_loss'] is None


@pytest.mark.parametrize('settings', [dict(rank_weight=-1), dict(rank_weight=np.inf),
                                      dict(learning_rate=0), dict(n_epochs=True), dict(random_state=-1)])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        PairwiseMatrixFactorization(**settings)


def test_invalid_data_and_artifacts(tmp_path):
    fresh = PairwiseMatrixFactorization(baseline_epochs=1, n_epochs=1)
    with pytest.raises(RuntimeError):
        fresh.predict([1], [1])
    with pytest.raises(ValueError):
        fresh.save(tmp_path / 'unfitted.joblib')
    with pytest.raises(ValueError):
        fresh.fit(ratings().assign(rating=np.nan))
    with pytest.raises(ValueError, match='Duplicate'):
        fresh.fit(pd.concat([ratings(), ratings().iloc[:1]]))
    path = tmp_path / 'bad.joblib'
    joblib.dump({'format_version': 9}, path)
    with pytest.raises(ValueError):
        fresh.load(path)
    fitted = fresh.fit(ratings())
    protected = Path(__file__).parent / 'artifacts/selected_mf.joblib'
    before = hashlib.sha256(protected.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='overwrite'):
        fitted.save(protected)
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == before
    fitted.model_.weights_.user_factors[0, 0] = np.nan
    with pytest.raises(ValueError, match='weights'):
        fitted.save(path)


def test_experiment_selection_and_weights_ignore_test_labels():
    rows = ratings()
    settings = dict(seeds=(42,), rank_weights=(0., .1), baseline_epochs=1, pairwise_epochs=1)
    first, _, model, _ = run_experiment(rows, **settings)
    splits = shared_split(rows)
    held = set(map(tuple, splits.loc[splits.split == 'test', ['userId', 'movieId']].to_numpy()))
    changed = rows.copy()
    changed.loc[[tuple(pair) in held for pair in rows[['userId', 'movieId']].to_numpy()], 'rating'] = .5
    second, _, other, _ = run_experiment(changed, **settings)
    assert first['validation'] == second['validation']
    assert first['selected_rank_weight'] == second['selected_rank_weight']
    pd.testing.assert_frame_equal(model.predict([1]*15, list(range(1, 16))),
                                  other.predict([1]*15, list(range(1, 16))))
    assert first['test'] != second['test']
    assert not first['promoted']
    assert not second['improvement_supported']


def test_bootstrap_pairs_users_and_averages_seeds():
    records = [dict(seed=seed, user_id=u, model=model, ndcg_at_10=value)
               for seed in (42, 43) for u in (1, 2)
               for model, value in [('baseline', .5), ('selected', .6)]]
    result = paired_interval(records)
    assert result['users'] == 2
    assert result['mean_delta'] == pytest.approx(.1)
    assert result['ci95'] == pytest.approx([.1, .1])
