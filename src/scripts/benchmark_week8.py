"""Benchmark exact frozen serving behavior; no training or test-set evaluation.

Cold means an empty process-local serving cache, not an empty OS file cache.
Run from the repository root. Baseline source snapshots must precede Week 8.
"""
import argparse
import cProfile
import importlib.util
import io
import json
import platform
import pstats
import statistics
import sys
from contextlib import redirect_stdout
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.hybrid import week7
from src.hybrid.recommender import recommend_for_user
from src.hybrid.serving_cache import SERVING_CACHE
from src.load_data import load_movielens


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def signature(rows):
    return [(r.movie_id, r.title, r.genres, float(getattr(r, 'score', getattr(r, 'ml_score', 0))))
            for r in rows]


def summarize(samples):
    return {'samples_seconds': samples, 'median_seconds': statistics.median(samples),
            'min_seconds': min(samples), 'max_seconds': max(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path,
                        default=Path('reports/week8/baseline/week7.py'))
    parser.add_argument('--baseline-genre', type=Path,
                        default=Path('reports/week8/baseline/genre_recommender.py'))
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('reports/week8/serving_latency.json'))
    args = parser.parse_args()
    if args.runs < 2:
        parser.error('Use at least two samples')
    if args.output.exists():
        parser.error('Choose a new output path; existing benchmark reports are preserved')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = load_source('week8_baseline_serving', args.baseline_source)
    old_genre = load_source('week8_baseline_genre', args.baseline_genre)
    baseline.score_movies_by_genre = old_genre.score_movies_by_genre
    optimized = week7.recommend_week7
    started = perf_counter()
    bundle = week7.load_bundle()
    bundle_load = perf_counter() - started
    bundle_digest = week7.file_digest(week7.DEFAULT_BUNDLE)
    preset_digest = week7.file_digest(bundle['selected_config_path'])
    started = perf_counter()
    ratings, movies = load_movielens('data')
    data_load = perf_counter() - started
    sparse = ratings.loc[(ratings.userId != 23) | (ratings.index == ratings.loc[ratings.userId == 23].index[0])].copy()
    cases = [('existing', 23, ratings), ('sparse_one_rating', 23, sparse),
             ('cold_start', int(ratings.userId.max()) + 1, ratings)]
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'numpy': np.__version__, 'pandas': pd.__version__, 'runs': args.runs,
              'scope': 'End-to-end recommend_for_user, including diversity; excludes imports, CSV and bundle loading',
              'cold_definition': 'Empty application serving cache; OS caches uncontrolled',
              'bundle_load_seconds': bundle_load, 'data_load_seconds': data_load,
              'bundle_sha256': bundle_digest, 'preset_sha256': preset_digest,
              'baseline_source_sha256': sha256(args.baseline_source.read_bytes()).hexdigest(),
              'baseline_genre_sha256': sha256(args.baseline_genre.read_bytes()).hexdigest(),
              'cases': []}

    def call(fn, uid, frame):
        start = perf_counter()
        with patch.object(week7, 'recommend_week7', fn), redirect_stdout(io.StringIO()):
            result = recommend_for_user(uid, frame, movies, 10, week7_bundle=bundle)
        return perf_counter() - start, signature(result)

    for label, uid, frame in cases:
        baseline_samples, cold_samples, warm_samples, phases = [], [], [], []
        expected = None
        for _ in range(args.runs):
            seconds, rows = call(baseline.recommend_week7, uid, frame)
            baseline_samples.append(seconds)
            if expected is None:
                expected = rows
            assert rows == expected
            SERVING_CACHE.clear()
            timings = {}
            seconds, rows = call(partial(optimized, timings=timings), uid, frame)
            cold_samples.append(seconds)
            assert rows == expected and not timings['cache_hit']
            phases.append(dict(timings))
            timings = {}
            seconds, rows = call(partial(optimized, timings=timings), uid, frame)
            warm_samples.append(seconds)
            assert rows == expected and timings['cache_hit']
        # Compare every candidate and exact floating-point score, not only top 10.
        with redirect_stdout(io.StringIO()):
            full_baseline = baseline.recommend_week7(uid, frame, movies, len(movies), bundle=bundle)
        full_optimized = optimized(uid, frame, movies, len(movies), bundle=bundle)
        assert signature(full_baseline) == signature(full_optimized)
        # Exercise real ratings changes with a populated cache. Compare against
        # the pre-Week-8 implementation on each revised serving input.
        revised = frame.copy()
        target_rows = revised.index[revised.userId == uid]
        if len(target_rows):
            revised.loc[target_rows[0], 'rating'] = 0.5 if revised.loc[target_rows[0], 'rating'] != .5 else 5.
        else:
            revised = pd.concat([revised, pd.DataFrame([dict(userId=uid, movieId=expected[0][0],
                rating=5., timestamp=int(frame.timestamp.max()) + 1)])], ignore_index=True)
        t = {}
        revision_seconds, revised_rows = call(partial(optimized, timings=t), uid, revised)
        assert not t['cache_hit']
        _, baseline_revised = call(baseline.recommend_week7, uid, revised)
        assert revised_rows == baseline_revised
        t = {}
        _, repeated = call(partial(optimized, timings=t), uid, revised)
        assert t['cache_hit'] and repeated == revised_rows
        # New watched movie must leave the candidate pool immediately.
        added = pd.concat([revised, pd.DataFrame([dict(userId=uid, movieId=revised_rows[0][0],
            rating=4., timestamp=int(frame.timestamp.max()) + 2)])], ignore_index=True)
        t = {}
        _, added_rows = call(partial(optimized, timings=t), uid, added)
        assert not t['cache_hit'] and revised_rows[0][0] not in [r[0] for r in added_rows]
        _, added_baseline = call(baseline.recommend_week7, uid, added)
        assert added_rows == added_baseline
        # Profile separately so profiler overhead never enters benchmark samples.
        for name, fn in [('baseline', baseline.recommend_week7), ('optimized_miss', partial(optimized, use_cache=False))]:
            profile = cProfile.Profile()
            profile.runcall(call, fn, uid, frame)
            profile.dump_stats(str(args.output.parent / f'{label}_{name}.prof'))
            text = io.StringIO()
            pstats.Stats(profile, stream=text).strip_dirs().sort_stats('cumulative').print_stats(40)
            (args.output.parent / f'{label}_{name}_profile.txt').write_text(text.getvalue())
        record = {'case': label, 'user_id': uid, 'history_rows': int((frame.userId == uid).sum()),
                  'baseline': summarize(baseline_samples), 'optimized_cold': summarize(cold_samples),
                  'optimized_warm': summarize(warm_samples), 'cold_phase_samples': phases,
                  'exact_all_candidate_equality': True, 'candidate_count': len(full_baseline),
                  'exact_top10_diversity_equality': True, 'rating_change_misses_then_hits': True,
                  'new_watched_movie_excluded': True, 'revision_seconds': revision_seconds,
                  'top10': expected}
        report['cases'].append(record)
        print(label, 'baseline/cold/warm medians:',
              *(round(record[k]['median_seconds'], 6) for k in ['baseline', 'optimized_cold', 'optimized_warm']), flush=True)
    assert week7.file_digest(week7.DEFAULT_BUNDLE) == bundle_digest
    assert week7.file_digest(bundle['selected_config_path']) == preset_digest
    report['artifacts_unchanged'] = True
    report['cache_bounds'] = SERVING_CACHE.info()
    args.output.write_text(json.dumps(report, indent=2))
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
