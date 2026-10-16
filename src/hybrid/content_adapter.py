import pandas as pd

from pathlib import Path

from src.content.model import ContentModel
from src.content.selection import load_selected_model
from src.content.profiles import ProfileConfig
from src.content.scoring import ScoringConfig
from src.data_processing.movielens import (
    movie_metadata_from_records,
    user_ratings_from_records,
)


def build_content_model_from_frames(
    profile_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    scoring_config: ScoringConfig | None = None,
    profile_config: ProfileConfig | None = None,
    use_selected_content: bool = False,
    selected_config_path: str | Path | None = None,
) -> ContentModel:
    ratings_objects = user_ratings_from_records(
        profile_ratings.to_dict("records")
    )

    movie_objects = movie_metadata_from_records(
        movies.to_dict("records")
    )

    if use_selected_content:
        if scoring_config is not None or profile_config is not None:
            raise ValueError(
                "Selected Content uses its frozen configuration."
            )

        if selected_config_path is None:
            return load_selected_model(ratings_objects, movie_objects)

        return load_selected_model(
            ratings_objects,
            movie_objects,
            config_path=selected_config_path,
        )

    if scoring_config is None:
        return ContentModel(
            ratings=ratings_objects,
            movies=movie_objects,
            profile_config=profile_config,
        )

    return ContentModel(
        ratings_objects,
        movie_objects,
        scoring_config,
        profile_config=profile_config,
    )