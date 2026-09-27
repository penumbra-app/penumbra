import pandas as pd

from src.content.model import ContentModel
from src.data_processing.movielens import (
    movie_metadata_from_records,
    user_ratings_from_records,
)


def build_content_model_from_frames(
    profile_ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> ContentModel:
    ratings_objects = user_ratings_from_records(
        profile_ratings.to_dict("records")
    )

    movie_objects = movie_metadata_from_records(
        movies.to_dict("records")
    )

    return ContentModel(
        ratings=ratings_objects,
        movies=movie_objects,
    )