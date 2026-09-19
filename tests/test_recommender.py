from types import SimpleNamespace

import pandas as pd
import pytest

import src.hybrid.recommender as recommender_module
from src.hybrid.recommender import Recommendation, recommend_for_user


def make_fake_ranked_recommendations():
    return [
        SimpleNamespace(
            movie_id=10,
            title="Movie A",
            genres="Drama",
            ml_score=2.5,
        ),
        SimpleNamespace(
            movie_id=20,
            title="Movie B",
            genres="Comedy",
            ml_score=1.8,
        ),
        SimpleNamespace(
            movie_id=30,
            title="Movie C",
            genres="Action",
            ml_score=0.9,
        ),
    ]


def test_recommend_for_user_returns_recommendations(monkeypatch):
    def fake_recommend_with_ml(
        user_id,
        ratings,
        movies,
        limit,
    ):
        return make_fake_ranked_recommendations()[:limit]

    monkeypatch.setattr(
        recommender_module,
        "recommend_with_ml",
        fake_recommend_with_ml,
    )

    ratings = pd.DataFrame()
    movies = pd.DataFrame()

    result = recommend_for_user(
        user_id=1,
        ratings=ratings,
        movies=movies,
        limit=2,
    )

    assert len(result) == 2

    assert all(
        isinstance(item, Recommendation)
        for item in result
    )


def test_recommend_for_user_preserves_ranking(monkeypatch):
    def fake_recommend_with_ml(
        user_id,
        ratings,
        movies,
        limit,
    ):
        return make_fake_ranked_recommendations()[:limit]

    monkeypatch.setattr(
        recommender_module,
        "recommend_with_ml",
        fake_recommend_with_ml,
    )

    result = recommend_for_user(
        user_id=1,
        ratings=pd.DataFrame(),
        movies=pd.DataFrame(),
        limit=3,
    )

    assert [item.movie_id for item in result] == [
        10,
        20,
        30,
    ]

    assert [item.score for item in result] == [
        2.5,
        1.8,
        0.9,
    ]


def test_recommend_for_user_converts_fields_correctly(monkeypatch):
    def fake_recommend_with_ml(
        user_id,
        ratings,
        movies,
        limit,
    ):
        return [
            SimpleNamespace(
                movie_id=42,
                title="Test Movie",
                genres="Sci-Fi|Drama",
                ml_score=3.25,
            )
        ]

    monkeypatch.setattr(
        recommender_module,
        "recommend_with_ml",
        fake_recommend_with_ml,
    )

    result = recommend_for_user(
        user_id=7,
        ratings=pd.DataFrame(),
        movies=pd.DataFrame(),
        limit=1,
    )

    assert result == [
        Recommendation(
            movie_id=42,
            title="Test Movie",
            genres="Sci-Fi|Drama",
            score=3.25,
        )
    ]


def test_recommend_for_user_passes_arguments_through(monkeypatch):
    captured = {}

    ratings = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [5],
            "rating": [4.0],
        }
    )

    movies = pd.DataFrame(
        {
            "movieId": [5],
            "title": ["Example"],
            "genres": ["Drama"],
        }
    )

    def fake_recommend_with_ml(
        user_id,
        ratings,
        movies,
        limit,
    ):
        captured["user_id"] = user_id
        captured["ratings"] = ratings
        captured["movies"] = movies
        captured["limit"] = limit

        return []

    monkeypatch.setattr(
        recommender_module,
        "recommend_with_ml",
        fake_recommend_with_ml,
    )

    result = recommend_for_user(
        user_id=123,
        ratings=ratings,
        movies=movies,
        limit=8,
    )

    assert result == []

    assert captured["user_id"] == 123
    assert captured["ratings"] is ratings
    assert captured["movies"] is movies

    expected_candidate_limit = max(8 * 5, 8)
    assert captured["limit"] == expected_candidate_limit


def test_recommend_for_user_propagates_model_errors(monkeypatch):
    def fake_recommend_with_ml(
        user_id,
        ratings,
        movies,
        limit,
    ):
        raise ValueError(f"User {user_id} has no ratings")

    monkeypatch.setattr(
        recommender_module,
        "recommend_with_ml",
        fake_recommend_with_ml,
    )

    with pytest.raises(
        ValueError,
        match="User 999 has no ratings",
    ):
        recommend_for_user(
            user_id=999,
            ratings=pd.DataFrame(),
            movies=pd.DataFrame(),
        )

def test_recommend_for_user_applies_diversity(monkeypatch):
    def fake_recommend_with_ml(
        user_id,
        ratings,
        movies,
        limit,
    ):
        return [
            SimpleNamespace(
                movie_id=1,
                title="Movie A",
                genres="Drama|Sci-Fi",
                ml_score=1.00,
            ),
            SimpleNamespace(
                movie_id=2,
                title="Movie B",
                genres="Drama|Sci-Fi",
                ml_score=0.95,
            ),
            SimpleNamespace(
                movie_id=3,
                title="Movie C",
                genres="Comedy",
                ml_score=0.90,
            ),
        ]

    monkeypatch.setattr(
        recommender_module,
        "recommend_with_ml",
        fake_recommend_with_ml,
    )

    result = recommend_for_user(
        user_id=1,
        ratings=pd.DataFrame(),
        movies=pd.DataFrame(),
        limit=3,
    )

    assert [movie.movie_id for movie in result] == [
        1,
        3,
        2,
    ]