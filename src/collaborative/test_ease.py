import joblib
import numpy as np
import pandas as pd
import pytest

from src.collaborative.ease import EASE
from src.collaborative.evaluation.ease_experiment import run_experiment
from src.collaborative.evaluation.benchmark import shared_split


def small():
    return pd.DataFrame({'userId': [1, 1, 2, 2, 3, 3], 'movieId': [1, 2, 1, 3, 2, 3], 'rating': [5., 4., 4., 2., 1., 5.]})


def validation():
    return pd.DataFrame({'userId': [1, 2, 3], 'movieId': [3, 2, 1], 'rating': [4., 2., 3.]})


def test_user_space_solution_matches_direct_ease_and_column_regressions():
    model = EASE(regularization=2).fit(small())
    x = np.array([[1., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    inverse = np.linalg.inv(x.T @ x + 2*np.eye(3))
    weights = -inverse / np.diag(inverse)
    np.fill_diagonal(weights, 0)
    np.testing.assert_allclose(model.scores_, x @ weights, atol=1e-12)
    for target in range(3):
        others = [i for i in range(3) if i != target]
        expected = np.linalg.solve(x[:, others].T @ x[:, others] + 2*np.eye(2), x[:, others].T @ x[:, target])
        np.testing.assert_allclose(weights[others, target], expected)
    assert (np.diag(weights) == 0).all()


def test_single_item_cannot_copy_itself_and_no_likes_are_safe():
    one = pd.DataFrame({'userId': [1, 2], 'movieId': [1, 1], 'rating': [5., 4.]})
    np.testing.assert_allclose(EASE().fit(one).scores_, 0, atol=1e-12)
    model = EASE().fit(small().assign(rating=1.))
    assert np.count_nonzero(model.scores_) == 0
    assert model.predict_ranking([1], [1]).confidence.iloc[0] == 0
    model.calibrate(validation())
    assert model.calibration_['slope'] == 0
    assert model.predict([999], [999]).predicted_score.iloc[0] == 3


def test_output_contract_unknowns_duplicates_empty_and_calibration(tmp_path):
    model = EASE().fit(small())
    with pytest.raises(RuntimeError, match='Calibrate'):
        model.predict([1], [1])
    with pytest.raises(ValueError, match='disjoint'):
        model.calibrate(small())
    model.calibrate(validation())
    users, movies = [1, 1, 999, 1], [3, 999, 1, 3]
    pred = model.predict(users, movies)
    assert pred.columns.tolist() == ['user_id', 'movie_id', 'predicted_score', 'confidence']
    assert pred.user_id.tolist() == users and pred.movie_id.tolist() == movies
    assert pred.predicted_score.between(.5, 5).all()
    assert pred.confidence.between(0, 1).all()
    assert pred.confidence.iloc[1:3].tolist() == [0, 0]
    assert model.predict([], []).empty
    assert model.calibration_['slope'] >= 0
    path = tmp_path/'ease.joblib'
    model.save(path)
    restored = EASE.load(path)
    pd.testing.assert_frame_equal(pred, restored.predict(users, movies))
    pd.testing.assert_frame_equal(model.predict_ranking(users, movies), restored.predict_ranking(users, movies))
    with pytest.raises(ValueError):
        model.predict([1], [])
    with pytest.raises(ValueError):
        model.predict([True], [1])
    with pytest.raises(ValueError, match='overwrite'):
        model.save(tmp_path/'selected_mf.joblib')


@pytest.mark.parametrize('kwargs', [dict(regularization=0), dict(regularization=np.nan), dict(like_threshold=6), dict(prior_strength=-1)])
def test_bad_settings(kwargs):
    with pytest.raises(ValueError):
        EASE(**kwargs)


def test_bad_rows_and_memory_limit_and_unfitted():
    with pytest.raises(RuntimeError):
        EASE().predict_ranking([1], [1])
    with pytest.raises(ValueError):
        EASE().fit(small().assign(rating=True))
    with pytest.raises(ValueError):
        EASE().fit(small().assign(userId=-1))
    with pytest.raises(ValueError):
        EASE().fit(pd.concat([small(), small().iloc[:1]]))
    with pytest.raises(ValueError, match='memory'):
        EASE(max_working_mb=.000001).fit(small())


def test_corrupt_artifact_and_refit_resets_calibration(tmp_path):
    path = tmp_path/'ease.joblib'
    joblib.dump({'format_version': 7}, path)
    with pytest.raises(ValueError):
        EASE.load(path)
    model = EASE().fit(small()).calibrate(validation())
    model.fit(small())
    assert model.calibration_ is None
    model.scores_[0, 0] = np.nan
    joblib.dump({'format_version': 1, 'model_type': 'ease', 'model': model}, path)
    with pytest.raises(ValueError, match='scores'):
        EASE.load(path)


def test_selection_calibration_and_predictions_ignore_test_labels():
    rows = pd.DataFrame([dict(userId=u, movieId=i, rating=float(5 if (i+u)%2 else 1), timestamp=i)
                         for u in (1, 2, 3) for i in range(1, 16)])
    settings = dict(regularizations=(1., 10.), baseline_seeds=(42,), baseline_epochs=1)
    first, model, _, _ = run_experiment(rows, **settings)
    splits = shared_split(rows)
    keys = set(map(tuple, splits.loc[splits.split == 'test', ['userId', 'movieId']].to_numpy()))
    changed = rows.copy()
    changed.loc[[tuple(p) in keys for p in rows[['userId','movieId']].to_numpy()], 'rating'] = .5
    second, other, _, _ = run_experiment(changed, **settings)
    assert first['selected_settings'] == second['selected_settings']
    assert first['calibration'] == second['calibration']
    assert [c['validation_ranking'] for c in first['candidates']] == [c['validation_ranking'] for c in second['candidates']]
    np.testing.assert_array_equal(model.scores_, other.scores_)
    assert first['test'] != second['test']
    assert not second['ranking_improvement_supported']
