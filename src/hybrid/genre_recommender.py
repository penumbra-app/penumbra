from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")
SECONDS_PER_DAY = 86_400
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    title: str
    genres: str
    predicted_rating: float
    personal_score: float
    quality_score: float
    explanation: str


def extract_release_year(title: str) -> float:
    """
    Extract the final four-digit year from a MovieLens title.

    Returns NaN when no release year is present.
    """
    match = YEAR_PATTERN.search(title)

    if match is None:
        return math.nan

    return float(match.group(1))


def prepare_movie_data(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Add parsed release years and genre lists to the movie table.
    """
    prepared = movies.copy()

    prepared["release_year"] = prepared["title"].map(extract_release_year)

    prepared["genre_list"] = prepared["genres"].map(
        lambda value: [
            genre
            for genre in value.split("|")
            if genre and genre != "(no genres listed)"
        ]
    )

    return prepared


def calculate_movie_quality(
    ratings: pd.DataFrame,
    prior_strength: float = 10.0,
) -> pd.DataFrame:
    """
    Calculate a Bayesian-adjusted movie rating.

    Movies with very few ratings are pulled toward the dataset-wide
    average instead of being trusted immediately.
    """
    global_average = float(ratings["rating"].mean())

    statistics = (
        ratings.groupby("movieId", as_index=False)
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "size"),
        )
    )

    statistics["quality_rating"] = (
        statistics["rating_count"] * statistics["average_rating"]
        + prior_strength * global_average
    ) / (statistics["rating_count"] + prior_strength)

    return statistics


def calculate_recency_weights(
    timestamps: pd.Series,
    half_life_years: float,
) -> np.ndarray:
    """
    Give newer ratings more weight using exponential decay.

    A rating one half-life old receives weight 0.5.
    """
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive")

    newest_timestamp = float(timestamps.max())

    age_in_years = (
        newest_timestamp - timestamps.to_numpy(dtype=float)
    ) / (SECONDS_PER_DAY * DAYS_PER_YEAR)

    return np.power(0.5, age_in_years / half_life_years)


def calculate_year_weights(
    rated_years: np.ndarray,
    candidate_year: float,
    penalty_per_year: float,
    minimum_weight: float,
) -> np.ndarray:
    """
    Measure how relevant each rated movie's era is to a candidate movie.
    """
    if not 0 <= minimum_weight <= 1:
        raise ValueError("minimum_weight must be between 0 and 1")

    if penalty_per_year < 0:
        raise ValueError("penalty_per_year cannot be negative")

    weights = np.ones(len(rated_years), dtype=float)

    if math.isnan(candidate_year):
        return weights

    known_years = ~np.isnan(rated_years)
    differences = np.abs(rated_years[known_years] - candidate_year)

    weights[known_years] = np.maximum(
        minimum_weight,
        1.0 - penalty_per_year * differences,
    )

    return weights


def build_explanation(
    genre_details: list[tuple[str, float, float]],
) -> str:
    """
    Explain which candidate genres contributed most strongly.
    """
    positive_genres = sorted(
        (
            (genre, preference)
            for genre, preference, _ in genre_details
            if preference > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    if not positive_genres:
        return "This movie has a solid overall rating, although its genres are not a strong personal match yet."

    strongest = [genre for genre, _ in positive_genres[:2]]

    if len(strongest) == 1:
        return f"Recommended mainly because of your positive history with {strongest[0]} movies."

    return (
        "Recommended mainly because of your positive history with "
        f"{strongest[0]} and {strongest[1]} movies."
    )

def rerank_for_diversity(
    recommendations: list[Recommendation],
    limit: int,
    repetition_penalty: float = 0.08,
) -> list[Recommendation]:
    selected: list[Recommendation] = []
    remaining = recommendations.copy()

    while remaining and len(selected) < limit:
        best_index = 0
        best_score = float("-inf")

        for index, candidate in enumerate(remaining):
            candidate_genres = set(candidate.genres.split("|"))

            overlap_penalty = 0.0

            for chosen in selected:
                chosen_genres = set(chosen.genres.split("|"))
                overlap_penalty += (
                    len(candidate_genres & chosen_genres)
                    * repetition_penalty
                )

            reranked_score = (
                candidate.predicted_rating - overlap_penalty
            )

            if reranked_score > best_score:
                best_score = reranked_score
                best_index = index

        selected.append(remaining.pop(best_index))

    return selected

def score_movies_by_genre(
    user_id: int,
    user_history: pd.DataFrame,
    reference_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    movie_ids: list[int],
    recency_half_life_years: float = 4.0,
    year_penalty_per_year: float = 0.005,
    minimum_year_weight: float = 0.50,
    confidence_prior: float = 5.0,
    quality_weight: float = 0.20,
) -> list[Recommendation]:
    prepared_movies = prepare_movie_data(movies)
    quality = calculate_movie_quality(reference_ratings)

    history = (
        user_history.loc[user_history["userId"] == user_id]
        .merge(prepared_movies, on="movieId", how="inner")
        .copy()
    )

    if history.empty:
        raise ValueError(f"User {user_id} has no ratings")

    history["recency_weight"] = calculate_recency_weights(
        history["timestamp"],
        recency_half_life_years,
    )

    user_average = float(
        np.average(
            history["rating"],
            weights=history["recency_weight"],
        )
    )

    global_average = float(reference_ratings["rating"].mean())

    genre_history: dict[str, pd.DataFrame] = {}

    for genre in {
        genre
        for genres in history["genre_list"]
        for genre in genres
    }:
        genre_history[genre] = history.loc[
            history["genre_list"].map(lambda genres: genre in genres)
        ]

    genre_arrays = {
        genre: (rows["release_year"].to_numpy(dtype=float),
                rows["recency_weight"].to_numpy(dtype=float),
                rows["rating"].to_numpy())
        for genre, rows in genre_history.items()
    }

    candidates = (
        prepared_movies.loc[
            prepared_movies["movieId"].isin(movie_ids)
        ]
        .merge(quality, on="movieId", how="left")
        .dropna(subset=["quality_rating"])
    )

    results: list[Recommendation] = []

    for candidate in candidates.itertuples(index=False):
        genre_details: list[tuple[str, float, float]] = []

        for genre in candidate.genre_list:
            arrays = genre_arrays.get(genre)
            if arrays is None:
                continue
            rated_years, recency_weights, rated_values = arrays

            year_weights = calculate_year_weights(
                rated_years,
                float(candidate.release_year),
                year_penalty_per_year,
                minimum_year_weight,
            )

            combined_weights = (
                recency_weights
                * year_weights
            )

            evidence = float(combined_weights.sum())

            if evidence <= 0:
                continue

            weighted_genre_average = float(
                np.average(
                    rated_values,
                    weights=combined_weights,
                )
            )

            confidence = evidence / (evidence + confidence_prior)

            preference = (
                weighted_genre_average - user_average
            ) * confidence

            genre_details.append(
                (genre, preference, confidence)
            )

        if genre_details:
            personal_score = float(
                np.mean(
                    [
                        preference
                        for _, preference, _ in genre_details
                    ]
                )
            )
        else:
            personal_score = 0.0

        quality_score = float(
            candidate.quality_rating - global_average
        )

        predicted_rating = float(
            np.clip(
                user_average
                + personal_score
                + quality_weight * quality_score,
                0.5,
                5.0,
            )
        )

        results.append(
            Recommendation(
                movie_id=int(candidate.movieId),
                title=candidate.title,
                genres=candidate.genres,
                predicted_rating=predicted_rating,
                personal_score=personal_score,
                quality_score=quality_score,
                explanation=build_explanation(genre_details),
            )
        )

    return results


def recommend_by_genre(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    limit: int = 5,
    recency_half_life_years: float = 4.0,
    year_penalty_per_year: float = 0.005,
    minimum_year_weight: float = 0.50,
    confidence_prior: float = 5.0,
    quality_weight: float = 0.20,
) -> list[Recommendation]:
    """
    Rank unseen movies using the user's genre preferences.

    Personal genre evidence is weighted by:
    - how recently the rating was entered;
    - how similar the rated movie's year is to the candidate's year;
    - how much evidence exists for that genre.
    """
    if limit <= 0:
        return []

    prepared_movies = prepare_movie_data(movies)
    quality = calculate_movie_quality(ratings)

    history = (
        ratings.loc[ratings["userId"] == user_id]
        .merge(prepared_movies, on="movieId", how="inner")
        .copy()
    )

    if history.empty:
        raise ValueError(f"User {user_id} has no ratings")

    history["recency_weight"] = calculate_recency_weights(
        history["timestamp"],
        recency_half_life_years,
    )

    user_average = float(
        np.average(
            history["rating"],
            weights=history["recency_weight"],
        )
    )

    global_average = float(ratings["rating"].mean())
    watched_ids = set(history["movieId"])

    candidates = (
        prepared_movies.loc[~prepared_movies["movieId"].isin(watched_ids)]
        .merge(quality, on="movieId", how="left")
        .dropna(subset=["quality_rating"])
    )

    genre_history: dict[str, pd.DataFrame] = {}

    for genre in {
        genre
        for genres in history["genre_list"]
        for genre in genres
    }:
        genre_history[genre] = history.loc[
            history["genre_list"].map(lambda genres: genre in genres)
        ]

    recommendations: list[Recommendation] = []

    for candidate in candidates.itertuples(index=False):
        genre_details: list[tuple[str, float, float]] = []

        for genre in candidate.genre_list:
            matching_history = genre_history.get(genre)

            if matching_history is None or matching_history.empty:
                continue

            year_weights = calculate_year_weights(
                matching_history["release_year"].to_numpy(dtype=float),
                float(candidate.release_year),
                year_penalty_per_year,
                minimum_year_weight,
            )

            combined_weights = (
                matching_history["recency_weight"].to_numpy(dtype=float)
                * year_weights
            )

            evidence = float(combined_weights.sum())

            if evidence <= 0:
                continue

            weighted_genre_average = float(
                np.average(
                    matching_history["rating"],
                    weights=combined_weights,
                )
            )

            confidence = evidence / (evidence + confidence_prior)

            preference = (
                weighted_genre_average - user_average
            ) * confidence

            genre_details.append(
                (genre, preference, confidence)
            )

        if genre_details:
            personal_score = float(
                np.mean(
                    [preference for _, preference, _ in genre_details]
                )
            )
        else:
            personal_score = 0.0

        quality_score = float(candidate.quality_rating - global_average)

        predicted_rating = float(
            np.clip(
                user_average
                + personal_score
                + quality_weight * quality_score,
                0.5,
                5.0,
            )
        )

        recommendations.append(
            Recommendation(
                movie_id=int(candidate.movieId),
                title=candidate.title,
                genres=candidate.genres,
                predicted_rating=predicted_rating,
                personal_score=personal_score,
                quality_score=quality_score,
                explanation=build_explanation(genre_details),
            )
        )

    recommendations.sort(
        key=lambda recommendation: (
            recommendation.predicted_rating,
            recommendation.personal_score,
            recommendation.quality_score,
        ),
        reverse=True,
    )

    candidate_pool = recommendations[:50]

    return rerank_for_diversity(
        recommendations=candidate_pool,
        limit=limit,
    )