import math

import pandas as pd
import pytest

from src.evaluation.content import evaluate_content, ranking_metrics
from src.evaluation.splits import build_user_evaluation_split


def test_ranking_metrics_exact_perfect_reverse_and_tied_scores():
    actual = [5, 4, 1]
    assert ranking_metrics(actual, actual)["pairwise_accuracy"] == 1
    assert ranking_metrics(actual, actual)["ndcg_at_10"] == 1
    assert ranking_metrics(actual, [1, 2, 3])["pairwise_accuracy"] == 0
    tied = ranking_metrics(actual, [3, 3, 3])
    assert tied["pairwise_accuracy"] == 0.5
    discounts = [1, 1 / math.log2(3), 0.5]
    assert tied["ndcg_at_10"] == pytest.approx((2 / 3) * sum(discounts) / sum(discounts[:2]))
    # A tie spanning the tenth position must include the entire block.
    assert ranking_metrics([5] + [1] * 10, [3] * 11)["ndcg_at_10"] == pytest.approx(
        sum(1 / math.log2(i + 2) for i in range(10)) / 11)


def test_unrankable_users_are_explicit_not_nan():
    result = ranking_metrics([1, 1], [1, 2])
    assert result["pairs"] == 0
    assert result["pairwise_accuracy"] is None
    assert result["ndcg_at_10"] is None


def fixtures():
    ratings = pd.DataFrame([dict(userId=1, movieId=i, rating=5 if i % 2 else 1, timestamp=i)
                            for i in range(1, 11)])
    movies = pd.DataFrame([dict(movieId=i, title=f"M {i}", genres="Action" if i % 2 else "Drama")
                           for i in range(1, 11)])
    return ratings, movies


def test_evaluation_reuses_shared_splits_and_cannot_learn_from_test_labels():
    ratings, movies = fixtures()
    report, splits = evaluate_content(ratings, movies)
    train, validation, test = build_user_evaluation_split(ratings)
    for label, expected in (("train", train), ("validation", validation), ("test", test)):
        assert splits.loc[splits.split == label, "movieId"].tolist() == expected.movieId.tolist()
    assert report["split_counts"] == {"train": 6, "validation": 2, "test": 2}
    assert report["selected_variant"] == "genre_only"
    assert report["unvalidated_features"] == ["cast", "director", "language", "release_era", "runtime"]
    changed = ratings.copy()
    changed.loc[changed.movieId.isin(test.movieId), "rating"] = 0
    second, _ = evaluate_content(changed, movies)
    assert report["results"]["validation"] == second["results"]["validation"]
    assert report["selected_config"] == second["selected_config"]
    assert report["results"]["test"] != second["results"]["test"]
    assert report["split_sha256"] != second["split_sha256"]


def test_evaluation_preserves_candidates_without_catalog_metadata():
    ratings, movies = fixtures()
    report, _ = evaluate_content(ratings, movies.iloc[:1])
    assert report["results"]["test"]["full"]["ratings"] == 2


def test_duplicate_ratings_and_missing_timestamps_rejected_before_split():
    ratings, movies = fixtures()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_content(pd.concat([ratings, ratings.iloc[:1]]), movies)
    ratings.loc[0, "timestamp"] = None
    with pytest.raises(ValueError, match="complete"):
        evaluate_content(ratings, movies)
