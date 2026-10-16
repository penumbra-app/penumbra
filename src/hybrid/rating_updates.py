"""Pure request-history updates; never mutate the bundle, MF or population.

Persistence/authorization belongs to the caller. Pass the returned frame to
individual/group serving. Its existing content-based keys handle invalidation.
"""
from numbers import Integral, Real
import numpy as np
import pandas as pd


def update_rating(ratings, *, user_id, movie_id, operation, rating=None, timestamp=None):
    """Return an updated copy for add/edit/remove, rejecting ambiguous duplicates.

    Add requires rating and timestamp; edit requires rating and keeps the old
    timestamp unless supplied; remove accepts neither. Unknown users may add.
    Integer row indexes are preserved so unrelated users keep their cache keys.
    Duplicate (user,movie) events need an upstream event-ID contract, not silent
    last-write-wins. This function does not persist changes or refresh models.
    """
    for name, value in [('user_id', user_id), ('movie_id', movie_id)]:
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if operation not in ('add', 'edit', 'remove'):
        raise ValueError('operation must be add, edit or remove')
    required = {'userId', 'movieId', 'rating', 'timestamp'}
    if (not isinstance(ratings, pd.DataFrame) or not ratings.columns.is_unique
            or not required.issubset(ratings.columns) or not ratings.index.is_unique):
        raise ValueError('ratings requires unique row indexes and the four serving columns')
    mask = (ratings.userId == user_id) & (ratings.movieId == movie_id)
    count = int(mask.sum())
    if count > 1:
        raise ValueError('Ambiguous duplicate rating events; resolve by event ID upstream')
    if operation == 'add' and count:
        raise ValueError('Rating already exists; use edit')
    if operation != 'add' and not count:
        raise ValueError('Rating does not exist')
    if operation == 'remove':
        if rating is not None or timestamp is not None:
            raise ValueError('remove accepts neither rating nor timestamp')
        return ratings.loc[~mask].copy(deep=True)
    if (not isinstance(rating, Real) or isinstance(rating, (bool, np.bool_))
            or not np.isfinite(rating) or not 0 <= rating <= 5):
        raise ValueError('rating must be finite and between 0 and 5')
    if operation == 'add' and timestamp is None:
        raise ValueError('add requires timestamp')
    if timestamp is not None and (not isinstance(timestamp, Integral)
            or isinstance(timestamp, (bool, np.bool_)) or timestamp < 0):
        raise ValueError('timestamp must be nonnegative integer seconds')
    result = ratings.copy(deep=True)
    if operation == 'edit':
        result.loc[mask, 'rating'] = float(rating)
        if timestamp is not None:
            result.loc[mask, 'timestamp'] = int(timestamp)
        return result
    if not len(ratings.index) or pd.api.types.is_integer_dtype(ratings.index.dtype):
        new_index = int(ratings.index.max()) + 1 if len(ratings.index) else 0
    else:
        new_index = 'penumbra:new:0'
        while new_index in ratings.index:
            new_index += ':next'
    added = pd.DataFrame([dict(userId=int(user_id), movieId=int(movie_id),
                               rating=float(rating), timestamp=int(timestamp))], index=[new_index])
    return pd.concat([result, added], sort=False)
