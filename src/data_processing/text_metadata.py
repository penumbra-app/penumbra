"""Explicit as-of filtering for user-generated MovieLens keyword metadata."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from src.content.schemas import MovieMetadata


def with_keywords_as_of(
    movies: Iterable[MovieMetadata], tags: Iterable[Mapping],
    cutoff: int, exclude_user_id: int | None = None,
) -> tuple[MovieMetadata, ...]:
    """Never use future tags or the evaluation user's own tag preferences.

    Static keywords/plots supplied on MovieMetadata are assumed to be available
    at the cutoff; callers supplying enrichment must establish that provenance.
    """
    keywords: dict[int, set[str]] = {}
    for tag in tags:
        if int(tag["timestamp"]) > cutoff:
            continue
        if exclude_user_id is not None and int(tag["userId"]) == exclude_user_id:
            continue
        value = tag["tag"].strip().casefold()
        if value:
            keywords.setdefault(int(tag["movieId"]), set()).add(value)
    return tuple(replace(movie, keywords=tuple(sorted(
        {value.strip().casefold() for value in movie.keywords if value.strip()}
        | keywords.get(movie.movie_id, set())
    ))) for movie in movies)
