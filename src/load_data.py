from pathlib import Path

import pandas as pd

def load_movielens(
    data_directory: str | Path,
    movies_path: str | Path | None = None,
    *,
    movies_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens ratings and optionally an enriched movie catalog."""
    if movies_path is not None and movies_file is not None:
        raise ValueError(
            "Specify either movies_path or movies_file, not both"
        )

    directory = Path(data_directory)
    ratings_path = directory / "ratings.csv"

    selected_movies = (
        movies_path if movies_path is not None else movies_file
    )
    resolved_movies_path = (
        Path(selected_movies)
        if selected_movies is not None
        else directory / "movies.csv"
    )

    if not ratings_path.is_file():
        raise FileNotFoundError(f"Missing file: {ratings_path}")

    if not resolved_movies_path.is_file():
        raise FileNotFoundError(
            f"Missing file: {resolved_movies_path}"
        )

    return (
        pd.read_csv(ratings_path),
        pd.read_csv(resolved_movies_path),
    )
