"""Week 8 serving regression tests: exact output and cache lifecycle."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.hybrid import week7
from src.hybrid.recommender import recommend_for_user
from src.hybrid.serving_cache import RankingCache, SERVING_CACHE, ranking_key


class IdentityScaler:
    n_features_in_ = 5
    scale_ = np.ones(5)

    def transform(self, values):
        return values


class FixedModel:
    n_features_in_ = 5

    def __init__(self):
        self.coef_ = np.array([[.3, .7, .4, .1, .05]])

    def decision_function(self, values):
        return values @ self.coef_[0]


class FixedMF:
    def __init__(self):
        self.bias = .2

    def predict(self, user_ids, movie_ids):
        return pd.DataFrame({'movie_id': movie_ids,
                             'predicted_score': [3 + self.bias] * len(movie_ids)})


@pytest.fixture
def serving(tmp_path, monkeypatch):
    SERVING_CACHE.clear()
    movies = pd.DataFrame({'movieId': range(1, 21),
                           'title': [f'Movie {i} ({1990 + i})' for i in range(1, 21)],
                           'genres': ['Drama', 'Comedy', 'Drama|Comedy', 'Action'] * 5,
                           'plot': ['sample plot'] * 20})
    refs = pd.DataFrame({'userId': [2] * 20, 'movieId': range(1, 21),
                         'rating': [2., 3., 4., 5.] * 5, 'timestamp': range(20)})
    history = refs.iloc[:3].copy()
    history['userId'] = 1
    ratings = pd.concat([refs, history], ignore_index=True)
    preset = tmp_path / 'preset.json'
    preset.write_text('{}')
    bundle = dict(features=week7.FEATURES, movies=movies.copy(), reference_ratings=refs,
                  ranker=SimpleNamespace(scaler=IdentityScaler(), model=FixedModel()),
                  collaborative_model=FixedMF(), selected_config_path=str(preset))
    calls = []

    def build(history, catalog, **kwargs):
        calls.append(len(history))
        value = float(history.rating.mean()) if len(history) else 2.5
        return SimpleNamespace(predict=lambda user_ids, movie_ids: [
            SimpleNamespace(movie_id=mid, predicted_score=value) for mid in movie_ids])

    monkeypatch.setattr(week7, 'build_content_model_from_frames', build)
    yield ratings, movies, bundle, calls
    SERVING_CACHE.clear()


def recommend(data, **kwargs):
    ratings, movies, bundle, _ = data
    return week7.recommend_week7(1, ratings, movies, bundle=bundle, **kwargs)


@pytest.mark.parametrize('n', [0, 1, 3])
def test_exact_cold_warm_and_diversity_with_limit_changes(serving, n):
    ratings, movies, bundle, calls = serving
    ratings = ratings.loc[(ratings.userId != 1) | ratings.index.isin(ratings.index[ratings.userId == 1][:n])]
    serving = ratings, movies, bundle, calls
    expected = recommend(serving, limit=100, use_cache=False)
    cold = recommend(serving, limit=100)
    warm = recommend(serving, limit=100)
    assert expected == cold == warm
    assert SERVING_CACHE.info()['hits'] == 1
    warm.clear()  # Mutating the returned list must not corrupt cached state.
    assert recommend(serving, limit=100) == expected
    for limit in [1, 3, 10, 30]:
        assert recommend(serving, limit=limit) == expected[:limit]
        actual = recommend_for_user(1, ratings, movies, limit, week7_bundle=bundle)
        from src.hybrid.diversity import rerank_for_diversity
        from src.hybrid.recommender import Recommendation
        pool = [Recommendation(r.movie_id, r.title, r.genres, r.ml_score)
                for r in expected[:max(limit * 5, limit)]]
        assert actual == rerank_for_diversity(pool, limit)
    assert set(r.movie_id for r in cold).isdisjoint(ratings.loc[ratings.userId == 1, 'movieId'])


@pytest.mark.parametrize('change', ['rating', 'timestamp', 'add', 'delete', 'reorder',
                                  'catalog', 'population', 'preset', 'ranker', 'mf', 'allowed'])
def test_mutations_miss_and_equal_fresh_computation(serving, change):
    ratings, movies, bundle, calls = serving
    recommend(serving)
    recommend(serving)
    assert len(calls) == 1
    row = ratings.index[ratings.userId == 1][0]
    if change == 'rating':
        ratings.loc[row, 'rating'] = 5.
    elif change == 'timestamp':
        ratings.loc[row, 'timestamp'] += 1000
    elif change == 'add':
        ratings.loc[len(ratings)] = [1, 10, 5., 100]
    elif change == 'delete':
        ratings.drop(index=row, inplace=True)
    elif change == 'reorder':
        ratings.sort_values('timestamp', ascending=False, inplace=True)
    elif change == 'catalog':
        bundle['movies'].loc[3, ['title', 'genres', 'plot']] = ['Changed (2020)', 'Mystery', 'changed plot']
    elif change == 'population':
        bundle['reference_ratings'].loc[0, 'rating'] = 5.
    elif change == 'preset':
        from pathlib import Path
        Path(bundle['selected_config_path']).write_text('{"new": true}')
    elif change == 'ranker':
        bundle['ranker'].model.coef_[0, 0] += .1
    elif change == 'mf':
        bundle['collaborative_model'].bias += .3
    elif change == 'allowed':
        movies.drop(index=movies.index[-1], inplace=True)
    changed = recommend(serving)
    assert len(calls) == 2
    assert changed == recommend(serving, use_cache=False)
    assert changed == recommend(serving)
    assert len(calls) == 3


def test_other_users_history_does_not_invalidate_frozen_population(serving):
    ratings, _, _, calls = serving
    expected = recommend(serving)
    ratings.loc[0, 'rating'] = 1.
    assert recommend(serving) == expected
    assert len(calls) == 1


def test_unsupported_fingerprint_bypasses_cache(serving, monkeypatch):
    monkeypatch.setattr(week7, 'ranking_key', lambda *args: None)
    assert recommend(serving) == recommend(serving)
    assert len(serving[-1]) == 2


def test_failures_never_populate_cache(serving, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('bad input')
    monkeypatch.setattr(week7, 'build_content_model_from_frames', fail)
    for _ in range(2):
        with pytest.raises(ValueError, match='bad input'):
            recommend(serving)
    assert SERVING_CACHE.info()['entries'] == 0


def test_empty_requests_and_opt_out(serving):
    assert recommend(serving, limit=0) == []
    ratings, movies, bundle, calls = serving
    assert week7.recommend_week7(1, ratings, movies.iloc[:0], bundle=bundle) == []
    assert not calls
    recommend(serving, use_cache=False)
    assert SERVING_CACHE.info()['entries'] == 0


def test_ties_missing_metadata_and_duplicate_catalog_rows(serving):
    ratings, movies, bundle, _ = serving
    bundle['ranker'].model.coef_[:] = 0
    bundle['movies'].loc[3, 'genres'] = None
    bundle['movies'].loc[4, 'title'] = None
    bundle['movies'] = pd.concat([bundle['movies'], bundle['movies'].iloc[:1]])
    expected = recommend(serving, limit=100, use_cache=False)
    assert recommend(serving, limit=100) == recommend(serving, limit=100) == expected
    assert [r.movie_id for r in expected] == list(range(4, 21))
    assert expected[0].genres == ''
    assert expected[1].title == str(bundle['movies'].loc[4, 'title'])


def test_schema_dtype_changes_and_missing_preset_invalidate_key(serving):
    ratings, movies, bundle, _ = serving
    history = ratings.loc[ratings.userId == 1]
    allowed = set(movies.movieId)
    original = ranking_key(1, history, allowed, bundle)
    assert original is not None
    assert ranking_key(1, history.copy(), allowed, bundle) == original
    changed = history.copy()
    changed['rating'] = changed.rating.astype('float32')
    assert ranking_key(1, changed, allowed, bundle) != original
    from pathlib import Path
    Path(bundle['selected_config_path']).unlink()
    assert ranking_key(1, history, allowed, bundle) is None


def test_genre_scores_match_pre_week8_golden():
    import json
    from pathlib import Path
    from src.hybrid.genre_recommender import score_movies_by_genre
    fixture = json.loads((Path(__file__).parent / 'fixtures/week8_genre_golden.json').read_text())
    for case in fixture:
        actual = score_movies_by_genre(case['user_id'], pd.DataFrame(case['history']),
            pd.DataFrame(case['references']), pd.DataFrame(case['movies']), case['movie_ids'])
        assert [vars(row) for row in actual] == case['expected']


def test_profile_reports_hit_and_inference_phases(serving):
    timings = {}
    recommend(serving, timings=timings)
    assert not timings['cache_hit']
    for phase in ['content_build', 'personal', 'collaborative', 'content_predict',
                  'population', 'feature_assembly', 'ranking', 'metadata']:
        assert timings[phase] >= 0
    recommend(serving, timings=timings)
    assert timings['cache_hit'] and 'content_build' not in timings


def test_lru_entry_candidate_bounds_and_clear():
    cache = RankingCache(max_entries=2, max_candidates=3)
    cache.put('a', [(1, 1.)])
    cache.put('b', [(2, 2.)])
    assert cache.get('a') == ((1, 1.),)
    cache.put('c', [(3, 3.)])
    assert cache.get('b') is None
    cache.put('d', [(4, 4.), (5, 5.)])
    assert cache.info()['candidates'] <= 3
    cache.put('oversize', [(i, float(i)) for i in range(4)])
    assert cache.get('oversize') is None
    cache.clear()
    assert cache.info()['entries'] == cache.info()['hits'] == 0


def test_concurrent_cache_bookkeeping_is_bounded():
    cache = RankingCache(max_entries=4, max_candidates=8)
    def request(i):
        cache.put(str(i), [(i, float(i))])
        rows = cache.get(str(i))
        assert rows is None or rows == ((i, float(i)),)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(request, range(100)))
    assert cache.info()['entries'] <= 4
    assert cache.info()['candidates'] <= 8
