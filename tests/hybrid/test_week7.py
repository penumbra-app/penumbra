from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.hybrid.ml_reranker as ml
from src.hybrid.week7 import FEATURES, build_feature_rows, collaborative_scores, validate_ranker
from src.hybrid.recommender import recommend_for_user


def test_equal_user_weights_balance_scaler_and_model():
    weights = ml.equal_user_pair_weights([2, 8])
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())
    X = np.array([[1, 2, 3], [-1, -2, -3]] + [[4, 1, 0], [-4, -1, 0]] * 4)
    ranker = ml.train_ranker(X, np.array([1, 0] * 5), sample_weight=weights)
    assert np.allclose(ranker.scaler.var_, np.average(X ** 2, axis=0, weights=weights))
    assert np.allclose(ranker.model.coef_, ml.train_ranker(X, np.array([1, 0] * 5), weights).model.coef_)


@pytest.mark.parametrize("counts", [[], [0], [-1, 2]])
def test_invalid_user_counts(counts):
    with pytest.raises(ValueError):
        ml.equal_user_pair_weights(counts)


def test_ablation_mf_fits_only_profiles_and_population_excludes_target(monkeypatch):
    ratings = pd.DataFrame({"userId": [1] * 20 + [2] * 20,
                            "movieId": list(range(20)) + list(range(100, 120)),
                            "rating": [3., 4.] * 20, "timestamp": list(range(20)) * 2})
    captured = {}
    class MF:
        def __init__(self, **kwargs):
            pass
        def fit(self, rows):
            captured["mf"] = set(rows.movieId)
    monkeypatch.setattr(ml, "BiasedMatrixFactorization", MF)
    monkeypatch.setattr(ml, "build_content_model_from_frames", lambda **kwargs: None)
    def examples(**kwargs):
        uid = kwargs["user_id"]
        profile, targets = kwargs["profile_ratings"], kwargs["pairwise_ratings"]
        assert set(profile.movieId).isdisjoint(targets.movieId)
        assert uid not in set(kwargs["reference_ratings"].userId)
        captured[uid] = set(targets.movieId)
        n = 2 if uid == 1 else 6
        return np.ones((n, 5)), np.array([1, 0] * (n // 2))
    monkeypatch.setattr(ml, "build_user_ablation_training_examples", examples)
    X, y, users, weights = ml.build_ablation_training_dataset(
        ratings, pd.DataFrame(), list(FEATURES), return_user_weights=True, train_fraction=.10,
    )
    assert captured["mf"] == set(range(12)) | set(range(100, 112))
    assert captured[1] == {12, 13} and captured[2] == {112, 113}
    assert users == 2 and X.shape == (8, 5)
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())


def test_artifact_feature_order_and_width():
    ranker = ml.train_ranker(np.array([[1] * 5, [-1] * 5]), np.array([1, 0]))
    validate_ranker(ranker, FEATURES)
    with pytest.raises(ValueError, match="feature order"):
        validate_ranker(ranker, tuple(reversed(FEATURES)))
    wrong = ml.train_ranker(np.array([[1] * 3, [-1] * 3]), np.array([1, 0]))
    with pytest.raises(ValueError, match="five-feature"):
        validate_ranker(wrong, FEATURES)


def test_missing_collaborative_scores_join_by_id():
    model = SimpleNamespace(predict=lambda **kw: pd.DataFrame(
        {"movie_id": [3, 1], "predicted_score": [np.nan, 4.]}))
    assert collaborative_scores(model, 7, [1, 2, 3]) == {1: 4., 2: 2.5, 3: 2.5}
    model.predict = lambda **kw: pd.DataFrame({"wrong": []})
    with pytest.raises(ValueError, match="require"):
        collaborative_scores(model, 7, [1])


def fixtures():
    movies = pd.DataFrame({"movieId": list(range(1, 16)),
                           "title": [f"Movie {i} (2000)" for i in range(1, 16)],
                           "genres": ["Drama", "Comedy", None] * 5})
    refs = pd.DataFrame({"userId": [2] * 15, "movieId": list(range(1, 16)),
                         "rating": [3., 4., 5.] * 5, "timestamp": list(range(15))})
    return movies, refs


@pytest.mark.parametrize("history_ids", [[], [1], [1, 2, 3, 4]])
def test_top10_cold_sparse_existing_missing_metadata_and_determinism(monkeypatch, history_ids):
    movies, refs = fixtures()
    history = pd.DataFrame({"userId": [1] * len(history_ids), "movieId": history_ids,
                            "rating": [4.] * len(history_ids), "timestamp": list(range(len(history_ids)))})
    ratings = pd.concat([refs, history], ignore_index=True)
    content = SimpleNamespace(predict_one=lambda **kw: SimpleNamespace(predicted_score=3.))
    import src.hybrid.week7 as serving
    monkeypatch.setattr(serving, "build_content_model_from_frames", lambda *a, **kw: content)
    ranker = ml.train_ranker(np.array([[1] * 5, [-1] * 5]), np.array([1, 0]))
    bundle = {"ranker": ranker, "features": FEATURES, "movies": pd.concat([movies, movies.iloc[:1]]),
              "reference_ratings": refs, "selected_config_path": "unused", "collaborative_model": None}
    first = recommend_for_user(1, ratings, movies, 10, week7_bundle=bundle)
    second = recommend_for_user(1, ratings, movies.iloc[::-1], 10, week7_bundle=bundle)
    assert first == second
    ids = [row.movie_id for row in first]
    assert len(ids) == 10 and len(set(ids)) == 10
    assert set(ids).isdisjoint(history_ids)
    assert all(np.isfinite(row.score) for row in first)


def test_feature_fallbacks_without_population_or_history():
    movies, refs = fixtures()
    content = SimpleNamespace(predict_one=lambda **kw: SimpleNamespace(predicted_score=np.nan))
    ids, X = build_feature_rows(1, refs.iloc[:0], refs.iloc[:0], movies, [3, 3], content, None)
    assert ids == [3]
    assert X.tolist() == [[2.5, 0., 2.5, 0., 0.]]


def test_offline_training_and_serving_use_equivalent_features():
    from src.content.model import ContentModel
    from src.content.text import TextConfig
    from src.data_processing.movielens import movie_metadata_from_records, user_ratings_from_records
    from src.collaborative.matrix_factorization import BiasedMatrixFactorization
    movies, refs = fixtures()
    movies["genres"] = movies.genres.fillna("")
    movies["plot"] = ["space adventure drama", "family comedy drama", "mystery detective drama"] * 5
    history = refs.iloc[:3].copy()
    history["userId"] = 1
    targets = refs.iloc[3:6].copy()
    targets["userId"] = 1
    content = ContentModel(user_ratings_from_records(history.to_dict("records")),
                           movie_metadata_from_records(movies.to_dict("records")),
                           text_config=TextConfig(weight=1.0))
    mf = BiasedMatrixFactorization(n_epochs=1).fit(pd.concat([refs, history]))
    offline_X, offline_y = ml.build_user_ablation_training_examples(
        1, history, targets, refs, movies, content, mf, list(FEATURES),
    )
    ids, serving_X = build_feature_rows(1, history, refs, movies, list(targets.movieId), content, mf)
    candidates = {mid: ml.AblationCandidate(mid, *row) for mid, row in zip(ids, serving_X)}
    X, y = ml.build_ablation_pairwise_examples(candidates, dict(zip(targets.movieId, targets.rating)), list(FEATURES))
    assert np.allclose(offline_X, X) and np.array_equal(offline_y, y)


@pytest.mark.parametrize("module", [
    "evaluate_week6_ablations", "evaluate_week6_gbt", "evaluate_week6_gbt_ablations",
    "week6_gbt_ablations", "train_week6_gbt", "week6_ablations",
])
def test_legacy_content_report_compatibility(module):
    from importlib import import_module
    from pathlib import Path
    scoring, profile, variant = import_module("src.scripts." + module).load_content_configuration(
        Path("docs/week6-content-repro.json"))
    assert profile.cold_start_score == 2.5
    assert variant == "without_language"


def test_paired_user_bootstrap_is_deterministic_and_paired():
    from src.scripts.verify_week7 import paired_bootstrap
    first = [{"user_id": uid, "pairwise_accuracy": .5, "ndcg_at_10": .7,
              "credit": n / 2, "pairs": n} for uid, n in [(1, 2), (2, 200)]]
    second = [dict(r, pairwise_accuracy=.6, ndcg_at_10=.8, credit=r["pairs"] * .6) for r in first]
    intervals = paired_bootstrap(first, second, repetitions=20)
    assert intervals == paired_bootstrap(first, second, repetitions=20)
    for interval in intervals["percentile_95_intervals"].values():
        assert interval == pytest.approx([.1, .1])
    with pytest.raises(ValueError, match="same ordered users"):
        paired_bootstrap(first, list(reversed(second)))


def test_bundle_missing_incomplete_and_changed_preset_fail_clearly(tmp_path):
    import joblib
    from src.hybrid.week7 import file_digest, load_bundle
    path = tmp_path / "serving.joblib"
    with pytest.raises(FileNotFoundError, match="Missing Week 7"):
        load_bundle(path)
    joblib.dump({"schema_version": 1}, path)
    with pytest.raises(ValueError, match="Incomplete"):
        load_bundle(path)
    preset = tmp_path / "preset.json"
    preset.write_text("{}")
    ranker = ml.train_ranker(np.array([[1] * 5, [-1] * 5]), np.array([1, 0]))
    bundle = {"schema_version": 1, "ranker": ranker, "features": FEATURES,
              "movies": pd.DataFrame(), "reference_ratings": pd.DataFrame(),
              "collaborative_model": SimpleNamespace(is_fitted_=True),
              "selected_config_path": str(preset), "selected_config_sha256": file_digest(preset)}
    joblib.dump(bundle, path)
    assert load_bundle(path)["features"] == FEATURES
    preset.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="preset changed"):
        load_bundle(path)
