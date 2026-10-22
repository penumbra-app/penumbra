"""Trusted release loading and process-local, versioned rating snapshots.

No persistence, authorization, model refresh, or distributed consistency is
provided. Bundle/model/catalog objects must remain immutable after startup.
"""
from importlib.metadata import version
import json
from numbers import Integral
from pathlib import Path
import sys
from threading import RLock
from time import perf_counter

from .week7 import FEATURES, file_digest, load_bundle
from .rating_updates import update_rating
from .recommender import recommend_for_user
from .group_recommender import recommend_group


def _resource(root, entry):
    relative = Path(entry['path'])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError('Release resource must stay within release root')
    if file_digest(path) != entry['sha256']:
        raise ValueError('Release resource checksum mismatch before deserialization')
    return path


def load_release(root, *, trusted_manifest):
    """Operator-controlled paths only, never user-supplied artifacts/manifests.

    Authenticate the release channel before calling. All resource checks and
    exact dependency checks precede joblib deserialization; fail closed.
    """
    root = Path(root).resolve()
    manifest = json.loads(Path(trusted_manifest).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or tuple(manifest.get('features', ())) != FEATURES:
        raise ValueError('Unsupported release schema or features')
    if manifest.get('python') != '.'.join(map(str, sys.version_info[:3])):
        raise ValueError('Release Python version mismatch')
    packages = manifest.get('packages', {})
    if not {'numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl'} <= packages.keys():
        raise ValueError('Release dependency contract is incomplete')
    for package, expected in packages.items():
        if version(package) != expected:
            raise ValueError(f'Release dependency mismatch: {package}')
    artifact = _resource(root, manifest['artifact'])
    config = _resource(root, manifest['config'])
    bundle = load_bundle(artifact, expected_sha256=manifest['artifact']['sha256'], resource_root=root)
    if (Path(bundle['selected_config_path']).resolve() != config or
            bundle['selected_config_sha256'] != manifest['config']['sha256']):
        raise ValueError('Release and embedded configuration disagree')
    return bundle


class RatingHistory:
    """Own a frame; snapshots and updates are atomic within this process.

    Never expose the internal frame. Requests retain their snapshot when an
    update arrives mid-inference. Backend persistence/versioning is separate.
    """
    def __init__(self, ratings):
        self._lock = RLock()
        self._ratings = ratings.copy(deep=True)
        self._revision = 0

    def snapshot(self):
        with self._lock:
            return self._revision, self._ratings.copy(deep=True)

    def update(self, *, expected_revision=None, **change):
        with self._lock:
            if expected_revision is not None and expected_revision != self._revision:
                raise ValueError('Rating revision conflict')
            revised = update_rating(self._ratings, **change)
            self._ratings = revised
            self._revision += 1
            return self._revision


class ServingMetrics:
    """Optional bounded aggregate counters; no user IDs or histories retained."""
    def __init__(self):
        self._lock = RLock()
        self._values = {}

    def record(self, kind, seconds, error=False, members=0):
        with self._lock:
            for key, value in [(kind+'_requests',1),(kind+'_seconds',seconds),
                               (kind+'_errors',int(error)),(kind+'_members',members)]:
                self._values[key] = self._values.get(key,0) + value

    def snapshot(self):
        from .serving_cache import SERVING_CACHE
        with self._lock:
            return dict(self._values, cache=SERVING_CACHE.info())


class ServingSession:
    """Integration facade; every request uses one atomic rating snapshot.

    Public functions remain available for callers managing their own snapshots.
    Share the immutable bundle/catalog; do not mutate model arrays concurrently.
    """
    def __init__(self, bundle, history, *, metrics=None):
        self.bundle = bundle
        self.history = history
        self.metrics = metrics

    def _call(self, kind, operation, members=0):
        start = perf_counter(); failed = True
        try:
            result = operation()
            failed = False
            return result
        finally:
            if self.metrics is not None:
                self.metrics.record(kind, perf_counter()-start, failed, members)

    def individual(self, user_id, limit=10):
        def run():
            if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 0:
                raise ValueError('limit must be a nonnegative integer')
            _, ratings = self.history.snapshot()
            return recommend_for_user(user_id, ratings, self.bundle['movies'], limit,
                                      week7_bundle=self.bundle)
        return self._call('individual',run,1)

    def group(self, user_ids, limit=10, *, strategy='mean'):
        def run():
            _, ratings = self.history.snapshot()
            return recommend_group(user_ids,ratings,self.bundle['movies'],limit=limit,
                                   strategy=strategy,bundle=self.bundle)
        # Validation belongs to the existing group API; metrics never consume generators.
        members = len(user_ids) if isinstance(user_ids,(list,tuple)) else 0
        return self._call('group',run,members)

    def update(self, **change):
        return self._call('update',lambda:self.history.update(**change))
