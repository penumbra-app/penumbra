from collections.abc import Sequence
from typing import Protocol


class RecommendationLike(Protocol):
    genres: str
    score: float


def _genre_set(genres: str) -> set[str]:
    if not genres:
        return set()

    return {
        genre.strip()
        for genre in genres.split("|")
        if genre.strip()
    }


def _jaccard_similarity(
    genres_a: set[str],
    genres_b: set[str],
) -> float:
    if not genres_a and not genres_b:
        return 0.0

    union = genres_a | genres_b

    if not union:
        return 0.0

    return len(genres_a & genres_b) / len(union)


def rerank_for_diversity(
    recommendations: Sequence[RecommendationLike],
    limit: int,
    diversity_weight: float = 0.15,
) -> list[RecommendationLike]:
    """
    Greedily select recommendations while penalizing movies that are
    too similar in genre to movies already selected.

    A larger diversity_weight produces a more varied final slate.
    """
    if limit <= 0:
        return []

    if not recommendations:
        return []

    remaining = list(recommendations)
    selected: list[RecommendationLike] = []

    while remaining and len(selected) < limit:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        best_index = 0
        best_adjusted_score = float("-inf")

        selected_genres = [
            _genre_set(movie.genres)
            for movie in selected
        ]

        for index, candidate in enumerate(remaining):
            candidate_genres = _genre_set(candidate.genres)

            max_similarity = max(
                _jaccard_similarity(
                    candidate_genres,
                    chosen_genres,
                )
                for chosen_genres in selected_genres
            )

            adjusted_score = (
                candidate.score
                - diversity_weight * max_similarity
            )

            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_index = index

        selected.append(remaining.pop(best_index))

    return selected