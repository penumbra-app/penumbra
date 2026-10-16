"""Bounded, process-local memoization of exact Week 7 ranked scores.

No fitted Content/Collaborative objects are retained or modified. Keys are
content based, so in-place input/model changes cannot reuse stale rankings.
Callers must not mutate request inputs concurrently with a recommendation.
"""
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from threading import RLock
import pickle

import joblib
import pandas as pd


def frame_fingerprint(frame):
    """Include values, row order, index, column names/order and dtypes."""
    digest = sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(joblib.hash((tuple(frame.columns), tuple(map(str, frame.dtypes)),
                               tuple(frame.index.names))).encode())
    return digest.hexdigest()


def ranking_key(user_id, history, allowed_ids, bundle):
    """Fingerprint every input used by serving, including mutable model state.

    Hashing costs more than identity/version shortcuts, but also protects callers
    that edit DataFrames or model parameters in place. Unsupported custom model
    objects bypass caching rather than changing the prediction API's behavior.
    """
    try:
        preset = sha256(Path(bundle['selected_config_path']).read_bytes()).hexdigest()
        return joblib.hash((
            'week8-v1', user_id, frame_fingerprint(history), tuple(sorted(allowed_ids)),
            frame_fingerprint(bundle['movies']),
            frame_fingerprint(bundle['reference_ratings']),
            bundle['features'], bundle['ranker'], bundle['collaborative_model'],
            str(Path(bundle['selected_config_path']).resolve()), preset,
        ))
    except (OSError, TypeError, ValueError, pickle.PicklingError, AttributeError):
        return None


class RankingCache:
    """LRU with both entry and total candidate-count bounds; immutable values.

    Locks protect bookkeeping, not inference. Concurrent misses may compute the
    same key twice, avoiding serialization of unrelated users' expensive work.
    Errors are never cached. No files or user history are written to disk.
    """
    def __init__(self, max_entries=16, max_candidates=200_000):
        if max_entries < 0 or max_candidates < 0:
            raise ValueError('Cache bounds must be nonnegative')
        self.max_entries = max_entries
        self.max_candidates = max_candidates
        self._lock = RLock()
        self.clear()

    def clear(self):
        with self._lock:
            self._values = OrderedDict()
            self._candidates = 0
            self._hits = self._misses = self._evictions = 0

    def get(self, key):
        with self._lock:
            if key is None or key not in self._values:
                self._misses += 1
                return None
            self._hits += 1
            self._values.move_to_end(key)
            return self._values[key]

    def put(self, key, rows):
        if key is None or not self.max_entries or len(rows) > self.max_candidates:
            return
        value = tuple(rows)
        with self._lock:
            previous = self._values.pop(key, ())
            self._candidates -= len(previous)
            self._values[key] = value
            self._candidates += len(value)
            while (len(self._values) > self.max_entries
                   or self._candidates > self.max_candidates):
                _, removed = self._values.popitem(last=False)
                self._candidates -= len(removed)
                self._evictions += 1

    def info(self):
        with self._lock:
            return dict(entries=len(self._values), candidates=self._candidates,
                        hits=self._hits, misses=self._misses, evictions=self._evictions,
                        max_entries=self.max_entries, max_candidates=self.max_candidates)


SERVING_CACHE = RankingCache()
