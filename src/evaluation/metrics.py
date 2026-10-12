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


def ranking_metrics(actual, predicted, k: int = 10, like_threshold: float = 4.0) -> dict:
    """Binary NDCG with expected gain for score ties; pairwise ties get half credit."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if (actual.ndim != 1 or actual.shape != predicted.shape or not len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise ValueError("actual and predicted must be nonempty, finite, matching vectors")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not math.isfinite(like_threshold) or not 0 <= like_threshold <= 5:
        raise ValueError("like_threshold must be in [0, 5]")
    actual_diffs = actual[:, None] - actual
    score_diffs = predicted[:, None] - predicted
    mask = np.triu(actual_diffs != 0, k=1)
    pairs = int(mask.sum())
    correct = float(((np.sign(actual_diffs[mask]) == np.sign(score_diffs[mask])).sum()
                     + 0.5 * (score_diffs[mask] == 0).sum()))
    order = np.argsort(-predicted, kind="stable")
    gains = (actual[order] >= like_threshold).astype(float)
    ordered_scores = predicted[order]
    # Average gain within entire tied blocks, including blocks crossing rank k.
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1, len(gains)]
    expected = gains.copy()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        expected[start:end] = gains[start:end].mean()
    discounts = 1 / np.log2(np.arange(min(k, len(actual))) + 2)
    ideal = float(discounts[:min(int(gains.sum()), k)].sum())
    ndcg = float(np.dot(expected[:k], discounts) / ideal) if ideal else None
    return {"pairwise_accuracy": correct / pairs if pairs else None,
            "pairs": pairs, "pairwise_credit": correct, f"ndcg_at_{k}": ndcg}
