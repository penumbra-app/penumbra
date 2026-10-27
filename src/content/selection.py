"""Load the frozen, validation-selected Week 6 serving preset."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from src.content.model import ContentModel
from src.content.reliability import ProfileConfig
from src.content.schemas import MovieMetadata, UserRating
from src.content.scoring import ScoringConfig
from src.content.text import TextConfig


DEFAULT_PRESET = Path(__file__).with_name("selected_config.json")


def load_selected_model(
    ratings: Iterable[UserRating], movies: Iterable[MovieMetadata],
    config_path: str | Path = DEFAULT_PRESET,
) -> ContentModel:
    """Build the measured preset; callers provide metadata available as of use.

    Core constructor defaults stay stable for existing consumers. This factory
    and the content demo use the selected experiment configuration explicitly.
    Re-running evaluation writes a new report, never silently replaces this preset.
    """
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported content preset schema_version")
    config = data["candidate"]
    return ContentModel(
        ratings, movies, config=ScoringConfig(**config["scoring"]),
        profile_config=ProfileConfig(**config["profile"]),
        text_config=TextConfig(**config["text"]) if config["text"] else None,
    )
