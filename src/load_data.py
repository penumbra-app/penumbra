from pathlib import Path

import pandas as pd


def load_movielens(
    data_directory: str | Path,
    movies_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(data_directory)

    ratings_path = directory / "ratings.csv"
    resolved_movies_path = (
        Path(movies_path)
        if movies_path is not None
        else directory / "movies.csv"
    )

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing file: {ratings_path}")

    if not resolved_movies_path.exists():
        raise FileNotFoundError(
            f"Missing file: {resolved_movies_path}"
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(resolved_movies_path)

    return ratings, movies