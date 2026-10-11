import numpy as np
import pandas as pd
import pytest

from src.collaborative import BiasedMatrixFactorization
from src.collaborative.evaluation.benchmark import (
    ablation_models, evaluate_benchmark, shared_split, summarize_predictions,
)
from src.evaluation.content import evaluate_content
from src.evaluation.metrics import ranking_metrics


def fixture():
    return pd.DataFrame([
        dict(userId=u, movieId=i, rating=float(1 + (i + u) % 5), timestamp=i // 2)
        for u in (1, 2, 3) for i in range(1, 16)
    ] + [dict(userId=4, movieId=i, rating=3., timestamp=i) for i in range(1, 5)])


def test_same_assignments_and_hash_as_content_including_small_users_and_ties():
    ratings = fixture()
    movies = pd.DataFrame([dict(movieId=i, title=str(i), genres="Drama") for i in range(1, 16)])
    content, content_splits = evaluate_content(ratings, movies)
    report, splits, _ = evaluate_benchmark(ratings, {"n_epochs": 1})
    pd.testing.assert_frame_equal(content_splits, splits)
    assert report["split_sha256"] == content["split_sha256"]
    assert report["split_counts"] == content["split_counts"]


def test_test_labels_do_not_change_training_validation_or_selection():
    ratings = fixture()
    first, splits, _ = evaluate_benchmark(ratings, {"n_epochs": 1})
    test_keys = set(map(tuple, splits.loc[splits.split == "test", ["userId", "movieId"]].to_numpy()))
    changed = ratings.copy()
    mask = [tuple(pair) in test_keys for pair in changed[["userId", "movieId"]].to_numpy()]
    changed.loc[mask, "rating"] = .5
    second, _, _ = evaluate_benchmark(changed, {"n_epochs": 1})
    assert first["results"]["validation"] == second["results"]["validation"]
    assert first["selected_variant"] == second["selected_variant"]
    assert first["results"]["test"] != second["results"]["test"]


def test_ablations_isolate_contributions_and_preserve_full_model():
    model = BiasedMatrixFactorization(n_epochs=1).fit(fixture())
    original = model.predict([1, 999, 1], [1, 1, 999])
    variants = ablation_models(model)
    pd.testing.assert_frame_equal(original, model.predict([1, 999, 1], [1, 1, 999]))
    w = model.weights_
    u, m = w.user_to_idx[1], w.movie_to_idx[1]
    interaction = np.dot(w.user_factors[u], w.movie_factors[m])
    c = original.confidence.iloc[0]
    full = w.global_mean + w.user_biases[u] + w.movie_biases[m] + c * interaction
    for name, expected in {
        "full": full,
        "without_latent": full - c * interaction,
        "without_user_bias": full - w.user_biases[u],
        "without_movie_bias": full - w.movie_biases[m],
        "without_shrinkage": full + (1 - c) * interaction,
    }.items():
        pred = variants[name].predict([1], [1])
        assert pred.predicted_score.iloc[0] == pytest.approx(np.clip(expected, .5, 5))
        assert pred.confidence.iloc[0] == c


def test_ranking_aggregation_counts_and_no_relevant_users():
    rows = pd.DataFrame({"userId": [1, 1, 2, 2, 2], "rating": [5, 1, 3, 2, 1]})
    result, _ = summarize_predictions(rows, [5, 1, 1, 2, 3])
    assert result["pairwise_accuracy_macro"] == .5
    assert result["pairwise_accuracy_micro"] == .25
    assert result["ndcg_at_10"] == 1
    assert result["ndcg_users"] == 1
    assert result["pairs"] == 4
    result, _ = summarize_predictions(rows.assign(rating=1), [1]*5)
    assert result["ndcg_at_10"] is None
    assert result["pairwise_accuracy_macro"] is None


def test_ndcg_ties_crossing_cutoff_are_order_independent():
    actual = [5] + [1]*10
    expected = sum(1 / np.log2(i + 2) for i in range(10)) / 11
    for labels in (actual, actual[::-1]):
        assert ranking_metrics(labels, [3]*11)["ndcg_at_10"] == pytest.approx(expected)


@pytest.mark.parametrize("column,value", [("rating", np.inf), ("rating", 0), ("timestamp", np.nan), ("userId", -1)])
def test_invalid_data_rejected_before_splitting(column, value):
    rows = fixture()
    rows[column] = rows[column].astype(float)
    rows.loc[0, column] = value
    with pytest.raises(ValueError):
        shared_split(rows)


def test_duplicate_pairs_rejected():
    rows = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        shared_split(pd.concat([rows, rows.iloc[:1]]))
