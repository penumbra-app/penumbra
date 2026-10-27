from pathlib import Path

import pandas as pd

def load_movielens(
    data_directory: str | Path, movies_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    directory = Path(data_directory)

    ratings_path = directory / "ratings.csv"
    movies_path = Path(movies_file) if movies_file is not None else directory / "movies.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing file: {ratings_path}")

    if not movies_path.exists():
        raise FileNotFoundError(f"Missing file: {movies_path}")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    return ratings, movies
