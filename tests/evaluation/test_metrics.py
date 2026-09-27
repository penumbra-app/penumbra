import math

import pytest

from src.evaluation.metrics import (
    comparison_credit,
    mean_absolute_error,
    ndcg_at_k,
    root_mean_squared_error,
)


def test_comparison_credit_correct_order() -> None:
    assert comparison_credit(
        first_score=4.5,
        second_score=3.0,
        first_rating=5.0,
        second_rating=2.0,
    ) == 1.0


def test_comparison_credit_incorrect_order() -> None:
    assert comparison_credit(
        first_score=2.0,
        second_score=4.0,
        first_rating=5.0,
        second_rating=2.0,
    ) == 0.0


def test_comparison_credit_predicted_tie_gets_half_credit() -> None:
    assert comparison_credit(
        first_score=3.0,
        second_score=3.0,
        first_rating=5.0,
        second_rating=2.0,
    ) == 0.5


def test_mae_perfect_predictions_are_zero() -> None:
    actual = [5.0, 4.0, 2.0]
    predicted = [5.0, 4.0, 2.0]

    assert mean_absolute_error(actual, predicted) == 0.0


def test_mae_matches_manual_calculation() -> None:
    actual = [5.0, 3.0, 1.0]
    predicted = [4.0, 3.5, 2.5]

    # Absolute errors: 1.0, 0.5, 1.5
    # MAE = 3.0 / 3 = 1.0
    assert mean_absolute_error(actual, predicted) == pytest.approx(1.0)


def test_mae_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        mean_absolute_error([5.0, 4.0], [5.0])


def test_mae_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        mean_absolute_error([], [])


def test_rmse_perfect_predictions_are_zero() -> None:
    actual = [5.0, 4.0, 2.0]
    predicted = [5.0, 4.0, 2.0]

    assert root_mean_squared_error(actual, predicted) == 0.0


def test_rmse_matches_manual_calculation() -> None:
    actual = [5.0, 3.0, 1.0]
    predicted = [4.0, 3.5, 2.5]

    # Squared errors: 1.0, 0.25, 2.25
    # RMSE = sqrt(3.5 / 3)
    expected = math.sqrt(3.5 / 3.0)

    assert root_mean_squared_error(actual, predicted) == pytest.approx(
        expected
    )


def test_rmse_penalizes_large_errors_more_than_mae() -> None:
    actual = [5.0, 5.0, 5.0]
    predicted = [5.0, 5.0, 1.0]

    mae = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)

    assert rmse > mae


def test_rmse_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        root_mean_squared_error([5.0, 4.0], [5.0])


def test_rmse_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        root_mean_squared_error([], [])


def test_ndcg_perfect_ranking_is_one() -> None:
    actual = [5.0, 4.0, 3.0, 2.0, 1.0]
    predicted = [0.9, 0.8, 0.7, 0.6, 0.5]

    assert ndcg_at_k(actual, predicted, k=5) == pytest.approx(1.0)


def test_ndcg_imperfect_ranking_is_below_one() -> None:
    actual = [5.0, 4.0, 3.0, 2.0, 1.0]
    predicted = [0.1, 0.2, 0.3, 0.4, 0.5]

    score = ndcg_at_k(actual, predicted, k=5)

    assert 0.0 < score < 1.0


def test_ndcg_only_uses_top_k_positions() -> None:
    actual = [5.0, 4.0, 3.0, 2.0]
    predicted = [0.9, 0.8, 0.1, 0.2]

    # Top two predicted movies are also the ideal top two.
    assert ndcg_at_k(actual, predicted, k=2) == pytest.approx(1.0)


def test_ndcg_allows_k_larger_than_number_of_movies() -> None:
    actual = [5.0, 4.0, 3.0]
    predicted = [0.9, 0.8, 0.7]

    assert ndcg_at_k(actual, predicted, k=10) == pytest.approx(1.0)


def test_ndcg_ties_are_deterministic() -> None:
    actual = [5.0, 4.0, 3.0]
    predicted = [1.0, 1.0, 1.0]

    first = ndcg_at_k(actual, predicted, k=3)
    second = ndcg_at_k(actual, predicted, k=3)

    assert first == second


def test_ndcg_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k([5.0], [1.0], k=0)

    with pytest.raises(ValueError):
        ndcg_at_k([5.0], [1.0], k=-1)


def test_ndcg_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k([5.0, 4.0], [0.9], k=10)


def test_ndcg_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k([], [], k=10)


def test_ndcg_zero_relevance_returns_zero() -> None:
    actual = [0.0, 0.0, 0.0]
    predicted = [0.9, 0.8, 0.7]

    assert ndcg_at_k(actual, predicted, k=3) == 0.0