"""Group V1 serving/proxy evaluation on frozen profile histories only.

No training, final-test evaluation, preference labels or strategy selection.
Cold = empty application RankingCache. Warm strategy comparisons share scoring.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.hybrid import week7
from src.hybrid.diversity import _genre_set, _jaccard_similarity
from src.hybrid.group_recommender import recommend_group
from src.hybrid.recommender import recommend_for_user
from src.hybrid.serving_cache import SERVING_CACHE


STRATEGIES = ('mean', 'least_misery', 'fairness_aware')


def proxy_metrics(rows):
    """Descriptive, model-derived proxies, never observed group satisfaction."""
    if not rows:
        return {'slate_size': 0}
    users = [uid for uid, _ in rows[0].member_scores]
    values = np.array([[dict(row.member_scores)[uid] for uid in users] for row in rows])
    genre_sets = [_genre_set(r.genres) for r in rows]
    distances = [1 - _jaccard_similarity(a, b) for i, a in enumerate(genre_sets) for b in genre_sets[i + 1:]]
    return dict(slate_size=len(rows), mean_member_preference=float(values.mean()),
        mean_minimum_member_preference=float(values.min(axis=1).mean()),
        worst_member_slate_mean=float(values.mean(axis=0).min()),
        mean_disagreement=float(np.mean([r.disagreement for r in rows])),
        per_member_mean={str(uid): float(values[:, j].mean()) for j, uid in enumerate(users)},
        per_member_top_quartile_fraction={str(uid): float((values[:, j] >= .75).mean()) for j, uid in enumerate(users)},
        members_with_top_quartile_representation=float((values >= .75).any(axis=0).mean()),
        mean_pairwise_genre_distance=float(np.mean(distances)) if distances else 0.,
        unique_genres=len(set().union(*genre_sets)))


def summary(samples):
    return dict(samples_seconds=samples, median_seconds=statistics.median(samples),
                minimum_seconds=min(samples), maximum_seconds=max(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('reports/group_v1/evaluation.json'))
    parser.add_argument('--baseline-source', type=Path, default=Path('reports/group_v1/baseline/week7.py'))
    args = parser.parse_args()
    if args.runs < 2:
        parser.error('Use at least two repeated samples')
    if args.output.exists():
        parser.error('Existing reports are preserved; choose a new output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bundle = week7.load_bundle()
    artifacts = {str(week7.DEFAULT_BUNDLE): week7.file_digest(week7.DEFAULT_BUNDLE),
                 str(bundle['selected_config_path']): week7.file_digest(bundle['selected_config_path'])}
    # Use only the original serving/profile snapshot, never test/validation
    # labels. This measures recommendation behavior, not held-out relevance.
    ratings = bundle['reference_ratings'].copy()
    movies = pd.read_csv('data/movies.csv')
    available = sorted(int(uid) for uid in ratings.userId.unique())
    existing = [uid for uid in (23, 24, 25) if uid in available]
    existing += [uid for uid in available if uid not in existing][:3-len(existing)]
    sparse, cold = max(available) + 1, max(available) + 2
    one = ratings.iloc[:1].copy()
    one['userId'] = sparse
    with_sparse = pd.concat([ratings, one], ignore_index=True)
    cases = [('two_existing', existing[:2], ratings), ('three_existing', existing[:3], ratings),
             ('with_sparse', [existing[0], sparse], with_sparse),
             ('with_cold_start', [existing[0], cold], ratings)]
    report = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        platform=platform.platform(), runs=args.runs, artifact_sha256=artifacts,
        input_policy='Frozen bundle reference/profile ratings only; no final-test file or labels read',
        methodology='Mean cold/warm timing interleaved; all strategies then compared on warm member-score caches',
        cold_definition='Empty process-local RankingCache, uncontrolled OS caches',
        timing_scope='recommend_group including validation, all members, normalization, aggregation and diversity; excludes imports and data/bundle loading',
        proxy_caveat='Relative model-derived preference proxies cannot prove group satisfaction or identify a winning strategy; no real group-choice labels',
        cases=[])

    def timed(users, frame, strategy):
        start = perf_counter()
        rows = recommend_group(users, frame, movies, bundle=bundle, strategy=strategy)
        return perf_counter() - start, rows

    for label, users, frame in cases:
        cold_samples, warm_samples = [], []
        strategy_samples = {strategy: [] for strategy in STRATEGIES}
        strategy_rows = {}
        for _ in range(args.runs):
            SERVING_CACHE.clear()
            seconds, first = timed(users, frame, 'mean')
            cold_samples.append(seconds)
            assert SERVING_CACHE.info()['misses'] == len(users)
            before = SERVING_CACHE.info()['hits']
            seconds, repeated = timed(users[::-1], frame, 'mean')
            warm_samples.append(seconds)
            assert first == repeated and SERVING_CACHE.info()['hits'] - before == len(users)
            for strategy in STRATEGIES:
                seconds, rows = timed(users, frame, strategy)
                strategy_samples[strategy].append(seconds)
                assert not strategy_rows.get(strategy) or strategy_rows[strategy] == rows
                strategy_rows[strategy] = rows
            watched = set(frame.loc[frame.userId.isin(users), 'movieId'])
            assert all(not watched.intersection(r.movie_id for r in rows) for rows in strategy_rows.values())
        eligible = (set(movies.movieId) & set(bundle['movies'].movieId)) - watched
        case = dict(case=label, users=users, history_rows={str(uid): int((frame.userId == uid).sum()) for uid in users},
            shared_candidate_count=len(eligible), cold_mean=summary(cold_samples), warm_mean=summary(warm_samples),
            strategies={strategy: dict(metrics=proxy_metrics(strategy_rows[strategy]),
                warm_runtime=summary(strategy_samples[strategy]),
                recommendations=[asdict(row) for row in strategy_rows[strategy]]) for strategy in STRATEGIES})
        report['cases'].append(case)
        print(label, 'cold/warm:', round(case['cold_mean']['median_seconds'], 4),
              round(case['warm_mean']['median_seconds'], 4), flush=True)
        # Checkpoint results so a later failed assertion does not lose samples.
        args.output.with_suffix('.partial.json').write_text(json.dumps(report, indent=2))

    # Rating-only edits retain the pool, invalidating only the changed member.
    users = existing[:2]
    SERVING_CACHE.clear()
    timed(users, ratings, 'mean')
    revised = ratings.copy()
    index = revised.index[revised.userId == users[0]][0]
    revised.loc[index, 'rating'] = .5 if revised.loc[index, 'rating'] != .5 else 5.
    before = SERVING_CACHE.info()
    _, updated = timed(users, revised, 'mean')
    after = SERVING_CACHE.info()
    assert after['misses'] - before['misses'] == 1 and after['hits'] - before['hits'] == 1
    SERVING_CACHE.clear()
    _, fresh = timed(users, revised, 'mean')
    assert updated == fresh
    watched_movie = updated[0].movie_id
    revised = pd.concat([revised, pd.DataFrame([dict(userId=users[0], movieId=watched_movie,
        rating=5., timestamp=int(revised.timestamp.max()) + 1)])], ignore_index=True)
    _, newly_watched = timed(users, revised, 'mean')
    assert watched_movie not in [r.movie_id for r in newly_watched]
    report['rating_invalidation'] = dict(changed_member_missed=True, unchanged_member_hit=True,
        exact_fresh_equality=True, newly_watched_excluded=True)

    # Compare full individual rankings with the pre-group implementation using
    # real frozen artifacts, without allowing shared cache results to mask errors.
    spec = importlib.util.spec_from_file_location('before_group_week7', args.baseline_source)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    original_recommend = week7.recommend_week7
    preservation = []
    for uid, frame in [(existing[0], ratings), (sparse, with_sparse), (cold, ratings)]:
        baseline = original.recommend_week7(uid, frame, movies, len(movies), bundle=bundle, use_cache=False)
        current = week7.recommend_week7(uid, frame, movies, len(movies), bundle=bundle, use_cache=False)
        assert [vars(r) for r in baseline] == [vars(r) for r in current]
        with patch.object(week7, 'recommend_week7', original.recommend_week7):
            old_slate = recommend_for_user(uid, frame, movies, 10, week7_bundle=bundle)
        assert recommend_for_user(uid, frame, movies, 10, week7_bundle=bundle) == old_slate
        assert week7.recommend_week7 is original_recommend
        preservation.append(dict(user_id=uid, full_candidate_count=len(baseline), exact_full_ranking=True,
                                 exact_individual_diversity=True))
    # Also measure individual warm performance after group use.
    recommend_for_user(existing[0], ratings, movies, 10, week7_bundle=bundle)
    individual = []
    for _ in range(args.runs):
        start = perf_counter()
        recommend_for_user(existing[0], ratings, movies, 10, week7_bundle=bundle)
        individual.append(perf_counter() - start)
    report['individual_preservation'] = preservation
    report['individual_warm_latency'] = summary(individual)
    assert all(week7.file_digest(path) == digest for path, digest in artifacts.items())
    report['artifacts_unchanged'] = True
    args.output.write_text(json.dumps(report, indent=2))
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
