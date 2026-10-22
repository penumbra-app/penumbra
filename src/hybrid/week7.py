"""Explicit five-signal serving, with the frozen Week 7 preprocessing contract.

Bundles contain the upstream profile-only MF, catalog, population snapshot,
and reranker together. Never fit MF inside a recommendation request.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pickle
from numbers import Integral

import joblib
import numpy as np
import pandas as pd

from time import perf_counter

from src.hybrid.serving_cache import SERVING_CACHE, ranking_key
from src.hybrid.content_adapter import build_content_model_from_frames
from src.hybrid.genre_recommender import calculate_movie_quality, score_movies_by_genre
from src.hybrid.ml_reranker import AblationCandidate, calculate_movie_popularity

FEATURES = ("collaborative", "personal", "content", "quality", "popularity")
DEFAULT_BUNDLE = Path("src/models/week7_serving.joblib")


def file_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_ranker(ranker, features):
    if tuple(features) != FEATURES:
        raise ValueError("Week 7 feature order must be " + repr(FEATURES))
    if any(getattr(part, "n_features_in_", None) != len(FEATURES)
           for part in (ranker.scaler, ranker.model)):
        raise ValueError("Week 7 artifact must contain a fitted five-feature scaler and model")
    if not np.isfinite(ranker.model.coef_).all() or not np.isfinite(ranker.scaler.scale_).all():
        raise ValueError("Week 7 artifact contains nonfinite parameters")


def load_bundle(path=DEFAULT_BUNDLE, *, expected_sha256=None, resource_root=None):
    """Load only trusted local pickle/joblib artifacts. A digest checks integrity,
    not authenticity; obtain it from a separately trusted release manifest.
    Never expose this loader to user uploads or arbitrary remote paths.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing Week 7 serving bundle: {path}; restore the trusted frozen release artifact")
    if expected_sha256 is not None and file_digest(path) != expected_sha256:
        raise ValueError("Week 7 bundle checksum mismatch before deserialization")
    try:
        bundle = joblib.load(path)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError,
            KeyError, IndexError, AttributeError, ImportError) as error:
        raise ValueError(f"Cannot load Week 7 serving bundle: {path}") from error
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported Week 7 serving bundle schema")
    required = {"ranker", "features", "movies", "reference_ratings", "collaborative_model",
                "selected_config_path", "selected_config_sha256"}
    if not required <= bundle.keys():
        raise ValueError(f"Incomplete Week 7 serving bundle: missing {sorted(required - bundle.keys())}")
    if resource_root is not None:
        # Historical bundles contain Windows separators; never change process cwd.
        relative = Path(str(bundle["selected_config_path"]).replace("\\", "/"))
        base = Path(resource_root).resolve()
        resolved = (base / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(base):
            raise ValueError("Bundle configuration must stay within resource_root")
        bundle = dict(bundle, selected_config_path=str(resolved))
    validate_ranker(bundle["ranker"], bundle["features"])
    if file_digest(bundle["selected_config_path"]) != bundle["selected_config_sha256"]:
        raise ValueError("Frozen Content preset changed since bundle creation")
    if not getattr(bundle["collaborative_model"], "is_fitted_", False):
        raise ValueError("Week 7 bundle MF is not fitted")
    return bundle


def collaborative_scores(model, user_id, movie_ids, fallback=2.5, *, timings=None):
    """Join by movie ID, with an explicit neutral rating for missing scores.

    A malformed model contract raises; only absent/nonfinite score values fall back.
    """
    if model is None or not movie_ids:
        if timings is not None:
            timings["mf_missing_predictions"] = len(movie_ids)
            timings["mf_nonfinite_predictions"] = 0
        return {mid: fallback for mid in movie_ids}
    rows = model.predict(user_ids=[user_id] * len(movie_ids), movie_ids=movie_ids)
    if not {"movie_id", "predicted_score"} <= set(rows.columns):
        raise ValueError("Collaborative predictions require movie_id and predicted_score")
    if rows["movie_id"].duplicated().any():
        raise ValueError("Duplicate collaborative predictions")
    scores = rows.set_index("movie_id")["predicted_score"].to_dict()
    if timings is not None:
        timings["mf_missing_predictions"] = sum(mid not in scores for mid in movie_ids)
        timings["mf_nonfinite_predictions"] = sum(mid in scores and (pd.isna(scores[mid])
            or not np.isfinite(float(scores[mid]))) for mid in movie_ids)
    return {mid: float(scores[mid]) if mid in scores and pd.notna(scores[mid])
            and np.isfinite(float(scores[mid])) else fallback for mid in movie_ids}


def build_feature_rows(user_id, history, references, movies, movie_ids, content_model, mf, *, timings=None):
    """Same five signals/order as offline ablation training.

    Supported extensions for serving: missing population evidence -> zero
    quality/popularity, missing MF -> neutral 2.5, no history -> zero personal.
    """
    timings = {} if timings is None else timings
    feature_start = perf_counter()
    timings["feature_input_candidates"] = len(movie_ids)
    references = references.loc[references["userId"] != user_id]
    catalog = movies.drop_duplicates("movieId").copy()
    catalog["title"] = catalog["title"].fillna("").astype(str)
    catalog["genres"] = catalog["genres"].fillna("").astype(str)
    known = set(catalog["movieId"].astype(int))
    ids = list(dict.fromkeys(int(mid) for mid in movie_ids if int(mid) in known))
    timings["feature_candidates"] = len(ids)
    if not ids:
        timings["feature_total"] = perf_counter() - feature_start
        return ids, np.empty((0, len(FEATURES)))
    timings["feature_prepare"] = perf_counter() - feature_start
    phase_start = perf_counter()
    population_mean = float(references["rating"].mean()) if not references.empty else 2.5
    quality = calculate_movie_quality(references).set_index("movieId")["quality_rating"].to_dict()
    popularity = calculate_movie_popularity(references)
    timings["population"] = perf_counter() - phase_start
    phase_start = perf_counter()
    personal = {}
    usable_history = history.loc[history["userId"] == user_id]
    if not usable_history.empty and usable_history["movieId"].isin(known).any():
        personal = {row.movie_id: row.personal_score for row in score_movies_by_genre(
            user_id, usable_history, references, catalog, ids,
        )}
    timings["personal"] = perf_counter() - phase_start
    phase_start = perf_counter()
    mf_scores = collaborative_scores(mf, user_id, ids, timings=timings)
    timings["collaborative"] = perf_counter() - phase_start
    content_start = perf_counter()
    if callable(getattr(content_model, "predict", None)):
        predictions = content_model.predict(user_ids=[user_id], movie_ids=ids)
    else:
        predictions = [content_model.predict_one(user_id=user_id, movie_id=mid) for mid in ids]
    timings["content_predict"] = perf_counter() - content_start
    phase_start = perf_counter()
    if len(predictions) != len(ids):
        raise ValueError("Content prediction count does not match candidates")
    rows = []
    timings["content_nonfinite_predictions"] = 0
    for mid, cp in zip(ids, predictions, strict=True):
        if hasattr(cp, "movie_id") and cp.movie_id != mid:
            raise ValueError("Content prediction order does not match candidates")
        content_score = float(cp.predicted_score)
        if not np.isfinite(content_score):
            timings["content_nonfinite_predictions"] += 1
            content_score = 2.5
        rows.append(AblationCandidate(
            mid, mf_scores[mid], personal.get(mid, 0.0), content_score,
            quality.get(mid, population_mean) - population_mean,
            popularity.get(mid, 0.0),
        ).as_array(list(FEATURES)))
    X = np.vstack(rows)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite Week 7 features")
    timings["feature_assembly"] = perf_counter() - phase_start
    timings["feature_total"] = perf_counter() - feature_start
    return ids, X


@dataclass(frozen=True)
class Week7Recommendation:
    movie_id: int
    title: str
    genres: str
    ml_score: float


def score_week7_candidates(user_id, ratings, movies, *, bundle, use_cache=True,
                           timings=None):
    """Return all exact sorted (movie ID, raw score) pairs without diversity.

    Uses the same bounded cache and keys as individual serving.
    """
    recommend_start = perf_counter()
    timings = {} if timings is None else timings
    timings.clear()
    timings["cache_hit"] = False
    if not isinstance(user_id, Integral) or isinstance(user_id, (bool, np.bool_)) or user_id <= 0:
        raise ValueError("user_id must be a positive integer")
    for frame, columns in ((ratings, {"userId", "movieId", "rating", "timestamp"}),
                           (movies, {"movieId"})):
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not columns <= set(frame.columns):
            raise ValueError("Invalid serving DataFrame schema")
    for ids_column in (movies["movieId"], ratings["userId"], ratings["movieId"]):
        if len(ids_column) and (not pd.api.types.is_numeric_dtype(ids_column.dtype) or pd.api.types.is_complex_dtype(ids_column.dtype)
                or pd.api.types.is_bool_dtype(ids_column.dtype) or ids_column.isna().any()
                or not np.isfinite(ids_column.to_numpy(dtype=float)).all()
                or (ids_column <= 0).any() or (ids_column != np.floor(ids_column)).any()):
            raise ValueError("Serving IDs must be positive integers")
    validate_ranker(bundle["ranker"], bundle["features"])
    # Preserve training metadata; TF-IDF vocabulary is fit on user history only.
    catalog = bundle["movies"].drop_duplicates("movieId").copy()
    allowed = set(movies["movieId"].astype(int))
    history = ratings.loc[ratings["userId"] == user_id].copy()
    for column, upper in (("rating", 5.), ("timestamp", np.inf)):
        series = history[column]
        if len(series):
            if (not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype)
                    or pd.api.types.is_complex_dtype(series.dtype) or series.isna().any()):
                raise ValueError(f"Invalid history {column}")
            values = series.to_numpy(dtype=float)
            if (not np.isfinite(values).all() or (values < 0).any() or (values > upper).any()
                    or (column == "timestamp" and (values != np.floor(values)).any())):
                raise ValueError(f"Invalid history {column}")
    watched = set(history["movieId"].astype(int))
    ids = sorted(set(catalog["movieId"].astype(int)) & allowed - watched)
    timings["candidate_counts"] = dict(supplied_rows=len(movies), supplied_unique=len(allowed),
        frozen_rows=len(bundle["movies"]), frozen_unique=len(catalog), watched=len(watched), eligible=len(ids))
    timings["request_prepare"] = perf_counter() - recommend_start
    if not ids:
        timings["recommend_total"] = perf_counter() - recommend_start
        return ()
    phase_start = perf_counter()
    if bundle.get("selected_config_sha256") is not None and file_digest(bundle["selected_config_path"]) != bundle["selected_config_sha256"]:
        raise ValueError("Frozen Content preset changed since bundle creation")
    key = ranking_key(user_id, history, allowed, bundle) if use_cache else None
    timings["cache_fingerprint"] = perf_counter() - phase_start
    phase_start = perf_counter()
    ranked = SERVING_CACHE.get(key) if use_cache else None
    timings["cache_lookup"] = perf_counter() - phase_start
    timings["cache_key_lookup"] = timings["cache_fingerprint"] + timings["cache_lookup"]
    if ranked is None:
        phase_start = perf_counter()
        content = build_content_model_from_frames(
            history, catalog, use_selected_content=True,
            selected_config_path=bundle["selected_config_path"],
        )
        timings["content_build"] = perf_counter() - phase_start
        ids, X = build_feature_rows(user_id, history, bundle["reference_ratings"],
                                   catalog, ids, content, bundle["collaborative_model"],
                                   timings=timings)
        phase_start = perf_counter()
        ranker = bundle["ranker"]
        scores = ranker.model.decision_function(ranker.scaler.transform(X))
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite Week 7 ranking scores")
        timings["ranking"] = perf_counter() - phase_start
        phase_start = perf_counter()
        ranked = tuple(sorted(((mid, float(score)) for mid, score in zip(ids, scores)),
                              key=lambda row: (-row[1], row[0])))
        timings["sorting"] = perf_counter() - phase_start
        phase_start = perf_counter()
        if use_cache:
            SERVING_CACHE.put(key, ranked)
        timings["cache_store"] = perf_counter() - phase_start
        timings["sort_cache_store"] = timings["sorting"] + timings["cache_store"]
    else:
        timings["cache_hit"] = True
    timings["scoring_total"] = perf_counter() - recommend_start
    return ranked


def recommend_week7(user_id, ratings, movies, limit=10, *, bundle, use_cache=True,
                    timings=None):
    """Return the original ranked prefix; diversity remains the caller's job."""
    recommend_start = perf_counter()
    timings = {} if timings is None else timings
    if limit <= 0:
        timings.clear()
        timings["cache_hit"] = False
        timings["recommend_total"] = perf_counter() - recommend_start
        return []
    ranked = score_week7_candidates(user_id, ratings, movies, bundle=bundle,
                                    use_cache=use_cache, timings=timings)
    if not ranked:
        timings["recommend_total"] = perf_counter() - recommend_start
        return []
    phase_start = perf_counter()
    metadata = bundle["movies"].drop_duplicates("movieId").set_index("movieId")
    result = [Week7Recommendation(mid, str(metadata.loc[mid, "title"]),
                                 "" if pd.isna(metadata.loc[mid, "genres"]) else str(metadata.loc[mid, "genres"]),
                                 score) for mid, score in ranked[:limit]]
    timings["metadata"] = perf_counter() - phase_start
    timings["recommend_total"] = perf_counter() - recommend_start
    return result
