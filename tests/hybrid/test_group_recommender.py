from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.hybrid import group_recommender as group
from src.hybrid import week7
from src.hybrid.recommender import recommend_for_user
from src.hybrid.serving_cache import SERVING_CACHE
from tests.hybrid.test_week8 import serving


@pytest.fixture
def mathematical(serving, monkeypatch):
    ratings, movies, bundle, _ = serving
    ratings = ratings.iloc[:0]
    movies = movies.iloc[:3].copy()
    bundle['movies'] = movies.copy()
    scores = {1: [3., 2., 1.], 2: [1., 2., 3.], 3: [2., 3., 1.]}
    calls = []
    def score(uid, frame, shared, **kwargs):
        ids = list(shared.movieId)
        calls.append((uid, ids))
        return tuple((mid, scores[uid][mid - 1]) for mid in ids)
    monkeypatch.setattr(group, 'score_week7_candidates', score)
    return ratings, movies, bundle, scores, calls


def request(data, users=(1, 2), **kwargs):
    ratings, movies, bundle, *_ = data
    return group.recommend_group(users, ratings, movies, bundle=bundle, **kwargs)


@pytest.mark.parametrize('raw, expected', [
    ([10., 20., 30.], [0., .5, 1.]), ([10., 10., 30.], [.25, .25, 1.]),
    ([3., 3., 3.], [.5, .5, .5]), ([100.], [.5]), ([], []),
    ([-10., 0., 10.], [0., .5, 1.])])
def test_percentiles(raw, expected):
    assert group.percentile_preferences(raw).tolist() == expected


@pytest.mark.parametrize('raw', [[1., np.nan], [np.inf], [[1., 2.]]])
def test_nonfinite_and_matrix_scores_rejected(raw):
    with pytest.raises(ValueError):
        group.percentile_preferences(raw)


@pytest.mark.parametrize('strategy, winner, score', [
    ('mean', 1, .5), ('least_misery', 2, .5), ('fairness_aware', 2, .5)])
def test_conflicting_preferences_and_exact_aggregation(mathematical, strategy, winner, score):
    result = request(mathematical, strategy=strategy, limit=3)
    assert result[0].movie_id == winner and result[0].group_score == score
    for row in result:
        values = [value for _, value in row.member_scores]
        expected = {'mean': np.mean(values), 'least_misery': min(values),
                    'fairness_aware': .7 * np.mean(values) + .3 * min(values)}[strategy]
        assert row.group_score == expected == row.score
        assert row.min_satisfaction == min(values)
        assert row.disagreement == max(values) - min(values)


def test_three_users_and_order_independence(mathematical):
    result = request(mathematical, users=(3, 1, 2))
    assert result == request(mathematical, users=(2, 3, 1))
    assert all(tuple(uid for uid, _ in row.member_scores) == (1, 2, 3) for row in result)


def test_identical_and_constant_preferences(mathematical):
    mathematical[3][2] = mathematical[3][1]
    assert request(mathematical)[0].group_score == 1.
    assert all(row.disagreement == 0 for row in request(mathematical))
    mathematical[3][1] = mathematical[3][2] = [7., 7., 7.]
    result = request(mathematical)
    assert [r.movie_id for r in result] == [1, 2, 3]
    assert all(r.group_score == .5 and r.disagreement == 0 for r in result)


def test_shared_filtering_dedup_and_single_candidate(mathematical):
    ratings, movies, bundle, _, calls = mathematical
    ratings = pd.DataFrame({'userId': [1, 2, 9], 'movieId': [1, 2, 3],
                            'rating': [4., 4., 4.], 'timestamp': [1, 1, 1]})
    movies = pd.concat([movies, movies.iloc[:1], pd.DataFrame({'movieId': [999]})])
    result = group.recommend_group([1, 2], ratings, movies, bundle=bundle)
    assert [r.movie_id for r in result] == [3]
    assert all(ids == [3] for _, ids in calls)
    assert result[0].member_scores == ((1, .5), (2, .5))
    assert result[0].group_score == .5


def test_no_candidates_and_zero_limit(mathematical):
    ratings, movies, bundle, _, calls = mathematical
    assert group.recommend_group([1, 2], ratings, movies.iloc[:0], bundle=bundle) == []
    assert request(mathematical, limit=0) == []
    watched = pd.DataFrame({'userId': [1, 2, 1], 'movieId': [1, 2, 3],
                            'rating': [4.] * 3, 'timestamp': [0] * 3})
    assert group.recommend_group([1, 2], watched, movies, bundle=bundle) == []
    assert calls == []


@pytest.mark.parametrize('users', [[], [1], [1, 1], [0, 2], [-1, 2], [True, 2],
                                  [1., 2], ['1', 2], None, '12', 12])
def test_invalid_users(mathematical, users):
    with pytest.raises(ValueError):
        request(mathematical, users=users)


@pytest.mark.parametrize('options', [{'limit': -1}, {'limit': True}, {'limit': 1.5},
                                    {'strategy': 'median'}, {'strategy': None}, {'strategy': []}])
def test_invalid_options(mathematical, options):
    with pytest.raises(ValueError):
        request(mathematical, **options)


@pytest.mark.parametrize('change', ['missing', 'ids', 'nan', 'rating', 'timestamp', 'bundle', 'catalog'])
def test_invalid_tables_and_bundle(serving, change):
    ratings, movies, bundle, _ = serving
    if change == 'missing':
        ratings = ratings.drop(columns='rating')
    elif change == 'ids':
        ratings['userId'] = ratings.userId.astype(float)
    elif change == 'nan':
        ratings.loc[0, 'rating'] = np.nan
    elif change == 'rating':
        ratings.loc[0, 'rating'] = 9.
    elif change == 'timestamp':
        ratings.loc[0, 'timestamp'] = -1
    elif change == 'bundle':
        bundle = {}
    else:
        movies = movies.astype({'movieId': float})
    with pytest.raises(ValueError):
        group.recommend_group([1, 3], ratings, movies, bundle=bundle)


def test_immutable_output(mathematical):
    row = request(mathematical)[0]
    with pytest.raises(FrozenInstanceError):
        row.group_score = 0
    with pytest.raises(TypeError):
        row.member_scores[0][1] = 0


def test_diversity_after_aggregation_and_scores_preserved(mathematical):
    ratings, movies, bundle, scores, _ = mathematical
    scores[1] = scores[2] = [3., 2.9, 2.8]
    # Percentile gaps of .5 dwarf the diversity penalty; use a larger tied
    # low-end group where diversity can visibly change a near-tied score.
    from src.hybrid.diversity import rerank_for_diversity
    candidates = [group.GroupRecommendation(i, str(i), genres, score, ((1, score), (2, score)), score, 0)
                  for i, genres, score in [(1, 'Drama', 1.), (2, 'Drama', .95), (3, 'Comedy', .9)]]
    result = rerank_for_diversity(candidates, 3, tie_breaker=lambda row: row.movie_id)
    assert [r.movie_id for r in result] == [1, 3, 2]
    assert [r.group_score for r in result] == [1., .9, .95]
    actual = request(mathematical, limit=3)
    assert actual == rerank_for_diversity(sorted(actual, key=lambda r: (-r.group_score, r.movie_id)),
                                         3, tie_breaker=lambda row: row.movie_id)


def test_adjusted_diversity_ties_use_movie_id():
    from src.hybrid.diversity import rerank_for_diversity
    # .625 - .15 = .475 exactly equals the no-overlap alternative.
    rows = [group.GroupRecommendation(i, str(i), genres, score, (), score, 0.)
            for i, genres, score in [(1, 'Drama', 1.), (3, 'Drama', .625), (2, 'Comedy', .475)]]
    assert [r.movie_id for r in rerank_for_diversity(rows, 3, tie_breaker=lambda r: r.movie_id)] == [1, 2, 3]


def test_compromise_outside_each_members_top10_is_scored(serving, monkeypatch):
    ratings, movies, bundle, _ = serving
    movies = pd.DataFrame({'movieId': range(1, 23), 'title': [f'M{i}' for i in range(22)], 'genres': ['Drama'] * 22})
    bundle['movies'] = movies
    calls = []
    def scores(uid, frame, shared, **kwargs):
        values = list(range(100, 90, -1)) + list(range(10))
        if uid == 2:
            values = values[10:] + values[:10]
        values += [50, -1]
        calls.append(len(shared))
        return tuple(zip(range(1, 23), values))
    monkeypatch.setattr(group, 'score_week7_candidates', scores)
    result = group.recommend_group([1, 2], ratings.iloc[:0], movies, bundle=bundle,
                                    limit=1, strategy='least_misery')
    assert result[0].movie_id == 21 and calls == [22, 22]


def test_missing_member_scores_fail_instead_of_silently_dropping(mathematical, monkeypatch):
    monkeypatch.setattr(group, 'score_week7_candidates', lambda *a, **k: ((1, 3.),))
    with pytest.raises(ValueError, match='shared candidate pool'):
        request(mathematical)


def test_serving_cache_reuse_and_rating_invalidations(serving):
    ratings, movies, bundle, calls = serving
    # user 2 has watched the full fixture catalog, so use a cold user 3.
    before = group.recommend_group([1, 3], ratings, movies, bundle=bundle)
    assert len(calls) == 2
    assert group.recommend_group([3, 1], ratings, movies, bundle=bundle) == before
    assert len(calls) == 2 and SERVING_CACHE.info()['hits'] == 2
    ratings.loc[ratings.index[ratings.userId == 1][0], 'rating'] = 5.
    changed = group.recommend_group([1, 3], ratings, movies, bundle=bundle)
    assert len(calls) == 3  # Unchanged cold member reuses its own ranking.
    SERVING_CACHE.clear()
    assert group.recommend_group([1, 3], ratings, movies, bundle=bundle) == changed
    ratings = pd.concat([ratings, pd.DataFrame([dict(userId=1, movieId=changed[0].movie_id,
                                                   rating=5., timestamp=100)])], ignore_index=True)
    updated = group.recommend_group([1, 3], ratings, movies, bundle=bundle)
    assert changed[0].movie_id not in [r.movie_id for r in updated]


def test_group_pipeline_diversifies_after_percentile_aggregation(mathematical):
    ratings, _, bundle, scores, calls = mathematical
    movies = pd.DataFrame({'movieId': range(1, 21), 'title': [str(i) for i in range(20)],
                           'genres': ['Drama'] * 20})
    movies.loc[2, 'genres'] = 'Comedy'
    bundle['movies'] = movies
    scores[1] = scores[2] = list(range(20, 0, -1))
    result = group.recommend_group([1, 2], ratings, movies, bundle=bundle, limit=3)
    assert [r.movie_id for r in result[:2]] == [1, 3]
    assert result[1].group_score == pytest.approx(17/19)
    assert calls == [(1, list(range(1, 21))), (2, list(range(1, 21)))]


def test_evaluation_proxy_metrics_have_explicit_member_coverage():
    from src.scripts.evaluate_group_v1 import proxy_metrics
    rows = [group.GroupRecommendation(1, 'A', 'Drama', .5, ((1, 1.), (2, 0.)), 0., 1.),
            group.GroupRecommendation(2, 'B', 'Comedy', .5, ((1, .5), (2, .5)), .5, 0.)]
    result = proxy_metrics(rows)
    assert result['mean_member_preference'] == .5
    assert result['mean_minimum_member_preference'] == .25
    assert result['worst_member_slate_mean'] == .25
    assert result['mean_disagreement'] == .5
    assert result['per_member_top_quartile_fraction'] == {'1': .5, '2': 0.}
    assert result['members_with_top_quartile_representation'] == .5
    assert result['mean_pairwise_genre_distance'] == 1.
    assert result['unique_genres'] == 2
    assert proxy_metrics([]) == {'slate_size': 0}


@pytest.mark.parametrize('history_size', [0, 1, 3])
def test_real_content_pipeline_and_individual_preservation(serving, monkeypatch, history_size):
    ratings, movies, bundle, _ = serving
    ratings = ratings.loc[(ratings.userId != 1) | ratings.index.isin(ratings.index[ratings.userId == 1][:history_size])]
    from src.hybrid.content_adapter import build_content_model_from_frames
    monkeypatch.setattr(week7, 'build_content_model_from_frames', build_content_model_from_frames)
    bundle['selected_config_path'] = str(Path('src/content/selected_config.json'))
    before = recommend_for_user(1, ratings, movies, 10, week7_bundle=bundle)
    result = group.recommend_group([1, 3], ratings, movies, bundle=bundle)
    assert len(result) == 10
    assert all(0 <= value <= 1 for row in result for _, value in row.member_scores)
    assert result == group.recommend_group([3, 1], ratings, movies, bundle=bundle)
    assert recommend_for_user(1, ratings, movies, 10, week7_bundle=bundle) == before
    raw = week7.score_week7_candidates(1, ratings, movies, bundle=bundle)
    assert [(r.movie_id, r.ml_score) for r in week7.recommend_week7(1, ratings, movies, len(movies), bundle=bundle)] == list(raw)
