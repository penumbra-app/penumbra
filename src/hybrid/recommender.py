from dataclasses import dataclass

import pandas as pd

from .ml_reranker import recommend_with_ml
from .diversity import rerank_for_diversity


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    title: str
    genres: str
    score: float


def recommend_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    limit: int = 10,
) -> list[Recommendation]:
    if limit <= 0:
        return []

    candidate_limit = max(limit * 5, limit)

    ranked = recommend_with_ml(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        limit=candidate_limit,
    )

    candidates = [
        Recommendation(
            movie_id=item.movie_id,
            title=item.title,
            genres=item.genres,
            score=item.ml_score,
        )
        for item in ranked
    ]

    return rerank_for_diversity(
        candidates,
        limit=limit,
    )