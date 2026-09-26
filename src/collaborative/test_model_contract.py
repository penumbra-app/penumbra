import joblib
import numpy as np
import pandas as pd
import pytest
from .matrix_factorization import BiasedMatrixFactorization
from .tune import tuning_split, select_configuration


@pytest.fixture
def ratings():
    return pd.DataFrame({"userId": [1, 1, 2, 2, 3, 3], "movieId": [10, 20, 10, 30, 20, 30],
                         "rating": [5., 2., 4., 3., 2., 4.]})


@pytest.mark.parametrize("settings", [
    {"n_factors": 0}, {"n_factors": 1.5}, {"n_epochs": -1}, {"n_epochs": True},
    {"learning_rate": 0}, {"learning_rate": np.inf}, {"regularization": -1},
    {"regularization": np.nan}, {"prior_strength": "5"},
    {"random_state": -1}, {"random_state": 2**32}, {"shrink_latent": 1}])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        BiasedMatrixFactorization(**settings)


@pytest.mark.parametrize("field,value", [("userId", 1.5), ("movieId", np.inf),
    ("userId", -1), ("rating", np.nan), ("rating", 0), ("rating", 5.5)])
def test_invalid_training_rows(ratings, field, value):
    ratings[field] = ratings[field].astype(float)
    ratings.loc[0, field] = value
    with pytest.raises(ValueError):
        BiasedMatrixFactorization().fit(ratings)


@pytest.mark.parametrize("bad_id", [1.5, np.nan, np.inf, -1, "1", True])
def test_invalid_prediction_ids(ratings, bad_id):
    model = BiasedMatrixFactorization(n_epochs=2).fit(ratings)
    with pytest.raises(ValueError):
        model.predict([bad_id], [10])
    with pytest.raises(ValueError):
        model.predict([1], [bad_id])


def test_batch_order_duplicates_and_unknowns(ratings):
    model = BiasedMatrixFactorization(n_epochs=2).fit(ratings)
    users, movies = [1, 99, 1, 99, 1], [20, 10, 99, 99, 20]
    batch = model.predict(users, movies)
    assert batch.user_id.tolist() == users
    assert batch.movie_id.tolist() == movies
    assert batch.iloc[0].equals(batch.iloc[4])
    for i, (uid, mid) in enumerate(zip(users, movies)):
        np.testing.assert_array_equal(batch.iloc[i].to_numpy(), model.predict([uid], [mid]).iloc[0].to_numpy())
    assert batch.predicted_score.between(.5, 5).all()
    assert batch.confidence.between(0, 1).all()
    assert model.predict([], []).columns.tolist() == batch.columns.tolist()
    with pytest.raises(ValueError):
        model.predict([1], [])


def test_reproducible_training(ratings):
    a = BiasedMatrixFactorization(n_epochs=3).fit(ratings)
    b = BiasedMatrixFactorization(n_epochs=3).fit(ratings)
    np.testing.assert_array_equal(a.weights_.user_factors, b.weights_.user_factors)
    assert a.training_history_ == b.training_history_


def test_save_load_exact_and_legacy(ratings, tmp_path):
    model = BiasedMatrixFactorization(n_epochs=2, shrink_latent=False).fit(ratings)
    path = tmp_path / "nested" / "model.joblib"
    model.save(path)
    restored = BiasedMatrixFactorization.load(path)
    users, movies = [1, 99, 1, 99], [10, 10, 99, 99]
    pd.testing.assert_frame_equal(model.predict(users, movies), restored.predict(users, movies), check_exact=True)
    assert model.get_params() == restored.get_params()
    assert model.metadata_ == restored.metadata_
    del model.shrink_latent
    joblib.dump(model, path)
    assert BiasedMatrixFactorization.load(path).shrink_latent is False


def test_bad_artifacts_and_unfitted(tmp_path):
    model = BiasedMatrixFactorization()
    with pytest.raises(RuntimeError):
        model.predict([1], [10])
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "model.joblib")
    for artifact in [model, {"format_version": 999}, "wrong type"]:
        joblib.dump(artifact, tmp_path / "bad.joblib")
        with pytest.raises(ValueError):
            BiasedMatrixFactorization.load(tmp_path / "bad.joblib")


def test_corrupted_weight_shape(ratings, tmp_path):
    model = BiasedMatrixFactorization(n_epochs=2).fit(ratings)
    model.weights_.user_factors = np.zeros((1, 1))
    joblib.dump(model, tmp_path / "bad.joblib")
    with pytest.raises(ValueError, match="weights"):
        BiasedMatrixFactorization.load(tmp_path / "bad.joblib")


def test_tuning_split_and_selection_ignore_test_labels():
    rows = pd.DataFrame({"userId": [1] * 10 + [2] * 5, "movieId": range(15),
                         "rating": [3.] * 15, "timestamp": [1] * 15})
    profile, validation, test = tuning_split(rows.sample(frac=1, random_state=1))
    assert validation.movieId.tolist() == [6, 7]
    assert test.movieId.tolist() == [8, 9]
    assert set(profile.movieId).isdisjoint(validation.movieId)
    assert set(profile.movieId).isdisjoint(test.movieId)
    assert len(profile.loc[profile.userId == 2]) == 5
    params = [dict(n_factors=2, n_epochs=2)]
    chosen, experiments = select_configuration(profile, validation, params)
    rows.loc[rows.movieId.isin(test.movieId), "rating"] = 5.
    profile2, validation2, _ = tuning_split(rows.sample(frac=1, random_state=1))
    chosen2, experiments2 = select_configuration(profile2, validation2, params)
    assert chosen == chosen2
    assert [e["validation"] for e in experiments] == [e["validation"] for e in experiments2]


@pytest.mark.parametrize("timestamp", [np.inf, "10"])
def test_tuning_rejects_invalid_timestamp(timestamp):
    rows = pd.DataFrame({"userId": [1] * 10, "movieId": range(10), "rating": [3.] * 10,
                         "timestamp": [timestamp] * 10})
    with pytest.raises(ValueError, match="Timestamps"):
        tuning_split(rows)
