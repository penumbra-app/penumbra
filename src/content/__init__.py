from src.content.features import FeaturePreference, feature_values
from src.content.baselines import (
    RatingResidual,
    UserBaseline,
    calculate_user_baseline,
    calculate_user_baselines,
)
from src.content.genres import (
    GenrePreference,
    aggregate_genre_preferences,
    normalized_genre_contributions,
)
from src.content.errors import UnknownMovieError, UnknownUserError
from src.content.model import ContentModel
from src.content.reliability import ProfileConfig
from src.content.profiles import (
    PROFILE_VERSION,
    ProfileMetadata,
    UserTasteProfile,
    build_profile,
)
from src.content.scoring import (
    GenreComponentResult,
    GenreMatch,
    ScoringConfig,
    genre_component,
    predict_one,
    predict_batch,
)
from src.content.schemas import (
    FeatureDebug,
    MovieMetadata,
    PredictionDebug,
    PredictionResult,
    ReasonSignal,
    UserRating,
)

__all__ = [
    "FeatureDebug",
    "FeaturePreference",
    "feature_values",
    "GenrePreference",
    "GenreComponentResult",
    "GenreMatch",
    "MovieMetadata",
    "PROFILE_VERSION",
    "PredictionDebug",
    "PredictionResult",
    "ProfileMetadata",
    "ProfileConfig",
    "RatingResidual",
    "ReasonSignal",
    "ScoringConfig",
    "ContentModel",
    "UnknownMovieError",
    "UnknownUserError",
    "UserBaseline",
    "UserRating",
    "UserTasteProfile",
    "aggregate_genre_preferences",
    "build_profile",
    "calculate_user_baseline",
    "calculate_user_baselines",
    "genre_component",
    "normalized_genre_contributions",
    "predict_one",
    "predict_batch",
]
