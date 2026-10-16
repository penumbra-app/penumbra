"""Deterministic Group V1 over the frozen five-signal Week 7 scores."""
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd

from .diversity import rerank_for_diversity
from .week7 import score_week7_candidates, validate_ranker


@dataclass(frozen=True)
class GroupRecommendation:
    movie_id: int
    title: str
    genres: str
    group_score: float
    member_scores: tuple[tuple[int, float], ...]
    min_satisfaction: float
    disagreement: float

    @property
    def score(self):
        """Adapter for existing diversity; never replace the aggregation score."""
        return self.group_score


def percentile_preferences(raw_scores):
    """Average tied ranks scaled as (rank-1)/(n-1); neutral .5 if constant.

    Values express relative preference within this pool, not stars or odds.
    A single candidate supplies no relative evidence and also receives .5.
    """
    values = np.asarray(raw_scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Raw scores must be a finite one-dimensional sequence')
    if not len(values):
        return np.empty(0, dtype=float)
    if len(values) == 1 or np.all(values == values[0]):
        return np.full(len(values), .5)
    return (pd.Series(values).rank(method='average').to_numpy() - 1) / (len(values) - 1)


def _frame(frame, columns, name):
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or not set(columns).issubset(frame.columns)):
        raise ValueError(f'{name} must be a DataFrame with unique columns including {columns}')


def _ids(values, name):
    if values.empty:
        return
    if (values.isna().any() or not pd.api.types.is_integer_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype) or (values <= 0).any()):
        raise ValueError(f'{name} must contain positive integer IDs')


def _validate(user_ids, ratings, movies, bundle, limit, strategy):
    if isinstance(user_ids, (str, bytes)):
        raise ValueError('user_ids must contain at least two distinct positive integers')
    try:
        users = list(user_ids)
    except TypeError as error:
        raise ValueError('user_ids must be an iterable of positive integers') from error
    if (len(users) < 2 or any(not isinstance(u, Integral) or isinstance(u, (bool, np.bool_))
                             or u <= 0 for u in users)):
        raise ValueError('Groups require at least two distinct positive integer user IDs')
    users = tuple(sorted(int(u) for u in users))
    if len(set(users)) != len(users):
        raise ValueError('Duplicate user IDs are not allowed')
    if not isinstance(limit, Integral) or isinstance(limit, (bool, np.bool_)) or limit < 0:
        raise ValueError('limit must be a nonnegative integer')
    if not isinstance(strategy, str) or strategy not in ('mean', 'least_misery', 'fairness_aware'):
        raise ValueError('strategy must be mean, least_misery, or fairness_aware')
    _frame(ratings, ('userId', 'movieId', 'rating', 'timestamp'), 'ratings')
    _frame(movies, ('movieId',), 'movies')
    _ids(ratings.userId, 'ratings.userId')
    _ids(ratings.movieId, 'ratings.movieId')
    _ids(movies.movieId, 'movies.movieId')
    for column, minimum, maximum in [('rating', 0., 5.), ('timestamp', 0., np.inf)]:
        values = ratings[column]
        if values.empty:
            continue
        if (not pd.api.types.is_numeric_dtype(values.dtype)
                or pd.api.types.is_bool_dtype(values.dtype)
                or pd.api.types.is_complex_dtype(values.dtype) or values.isna().any()):
            raise ValueError(f'ratings.{column} must be finite numeric values')
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or np.any(numeric < minimum) or np.any(numeric > maximum):
            raise ValueError(f'ratings.{column} values are out of range')
        if column == 'timestamp' and np.any(numeric != np.floor(numeric)):
            raise ValueError('ratings.timestamp must contain whole seconds')
    required = {'movies', 'reference_ratings', 'ranker', 'features',
                'selected_config_path', 'collaborative_model'}
    if not isinstance(bundle, Mapping) or not required.issubset(bundle):
        raise ValueError('bundle must contain the frozen Week 7 serving fields')
    _frame(bundle['movies'], ('movieId', 'title', 'genres'), 'bundle movies')
    _ids(bundle['movies'].movieId, 'bundle movies.movieId')
    _frame(bundle['reference_ratings'], ('userId', 'movieId', 'rating'), 'bundle reference ratings')
    try:
        validate_ranker(bundle['ranker'], bundle['features'])
    except (AttributeError, TypeError, KeyError) as error:
        raise ValueError('bundle must contain a fitted five-feature Week 7 ranker') from error
    return users


def recommend_group(user_ids, ratings, movies, *, bundle, limit=10, strategy='mean', timings=None):
    """Score one shared unseen pool, aggregate percentiles, then diversify.

    strategy='fairness_aware' uses a provisional .7 mean + .3 minimum heuristic.
    member_scores is an immutable tuple of (user_id, normalized preference).
    Final order is greedy diversity order; aggregation and adjusted-score ties
    retain ascending movie ID from the deterministic initial group ranking.
    """
    started = perf_counter()
    timings = {} if timings is None else timings
    timings.clear()
    users = _validate(user_ids, ratings, movies, bundle, limit, strategy)
    timings["validation"] = perf_counter() - started
    if not limit:
        timings["group_total"] = perf_counter() - started
        return []
    watched = set(ratings.loc[ratings.userId.isin(users), 'movieId'])
    ids = sorted((set(movies.movieId) & set(bundle['movies'].movieId)) - watched)
    timings["candidate_count"] = len(ids)
    timings["candidate_prepare"] = perf_counter() - started - timings["validation"]
    if not ids:
        timings["group_total"] = perf_counter() - started
        return []
    # Restrict scoring itself, not just its output: all members see this pool.
    shared = pd.DataFrame({'movieId': ids})
    preferences = []
    timings["members"] = []
    normalization_seconds = 0.
    for user in users:
        member = {}
        ranked = score_week7_candidates(user, ratings, shared, bundle=bundle, timings=member)
        timings["members"].append(dict(user_id=user, **member))
        normalize_start = perf_counter()
        scores = dict(ranked)
        if len(ranked) != len(ids) or set(scores) != set(ids):
            raise ValueError('Week 7 scoring did not return exactly the shared candidate pool')
        preferences.append(percentile_preferences([scores[mid] for mid in ids]))
        normalization_seconds += perf_counter() - normalize_start
    timings["normalization"] = normalization_seconds
    aggregation_start = perf_counter()
    values = np.asarray(preferences)
    means = values.mean(axis=0)
    minima = values.min(axis=0)
    if strategy == 'mean':
        aggregate = means
    elif strategy == 'least_misery':
        aggregate = minima
    else:
        aggregate = .7 * means + .3 * minima
    # Aggregate every candidate before selecting the diversity pool. No member
    # top-K truncation. Keep the existing five-times-limit diversity convention.
    order = sorted(range(len(ids)), key=lambda i: (-aggregate[i], ids[i]))
    timings["aggregation_sort"] = perf_counter() - aggregation_start
    metadata_start = perf_counter()
    metadata = bundle['movies'].drop_duplicates('movieId').set_index('movieId')
    candidates = []
    for i in order[:max(int(limit) * 5, int(limit))]:
        mid = ids[i]
        genres = metadata.loc[mid, 'genres']
        candidates.append(GroupRecommendation(
            int(mid), str(metadata.loc[mid, 'title']), '' if pd.isna(genres) else str(genres),
            float(aggregate[i]), tuple((user, float(values[j, i])) for j, user in enumerate(users)),
            float(minima[i]), float(values[:, i].max() - minima[i]),
        ))
    timings["group_metadata"] = perf_counter() - metadata_start
    diversity_start = perf_counter()
    result = rerank_for_diversity(candidates, int(limit), tie_breaker=lambda row: row.movie_id)
    timings["diversity"] = perf_counter() - diversity_start
    timings["group_total"] = perf_counter() - started
    return result
