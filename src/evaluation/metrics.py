import math

import numpy as np


def comparison_credit(
    first_score: float,
    second_score: float,
    first_rating: float,
    second_rating: float,
) -> float:
    actual_direction = np.sign(first_rating - second_rating)
    predicted_direction = np.sign(first_score - second_score)

    if predicted_direction == 0:
        return 0.5

    if predicted_direction == actual_direction:
        return 1.0

    return 0.0


def mean_absolute_error(
    actual: list[float],
    predicted: list[float],
) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")

    if not actual:
        raise ValueError("actual and predicted cannot be empty")

    errors = [
        abs(actual_rating - predicted_rating)
        for actual_rating, predicted_rating in zip(actual, predicted)
    ]

    return float(np.mean(errors))


def root_mean_squared_error(
    actual: list[float],
    predicted: list[float],
) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")

    if not actual:
        raise ValueError("actual and predicted cannot be empty")

    squared_errors = [
        (actual_rating - predicted_rating) ** 2
        for actual_rating, predicted_rating in zip(actual, predicted)
    ]

    return math.sqrt(float(np.mean(squared_errors)))


def ndcg_at_k(
    actual_relevance: list[float],
    predicted_scores: list[float],
    k: int = 10,
) -> float:
    if len(actual_relevance) != len(predicted_scores):
        raise ValueError(
            "actual_relevance and predicted_scores must have the same length"
        )

    if not actual_relevance:
        raise ValueError(
            "actual_relevance and predicted_scores cannot be empty"
        )

    if k <= 0:
        raise ValueError("k must be positive")

    limit = min(k, len(actual_relevance))

    predicted_order = sorted(
        range(len(predicted_scores)),
        key=lambda index: (-predicted_scores[index], index),
    )

    ideal_order = sorted(
        range(len(actual_relevance)),
        key=lambda index: (-actual_relevance[index], index),
    )

    def discounted_gain(order: list[int]) -> float:
        return sum(
            actual_relevance[index] / math.log2(rank + 2)
            for rank, index in enumerate(order[:limit])
        )

    dcg = discounted_gain(predicted_order)
    ideal_dcg = discounted_gain(ideal_order)

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg