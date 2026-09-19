from src.hybrid.diversity import (
    _genre_set,
    _jaccard_similarity,
    rerank_for_diversity,
)
from src.hybrid.recommender import Recommendation


def test_genre_set_parses_pipe_separated_genres():
    assert _genre_set("Drama|Sci-Fi") == {
        "Drama",
        "Sci-Fi",
    }


def test_genre_set_empty_string_returns_empty_set():
    assert _genre_set("") == set()


def test_jaccard_similarity_identical_sets():
    a = {"Drama", "Sci-Fi"}
    b = {"Drama", "Sci-Fi"}

    assert _jaccard_similarity(a, b) == 1.0


def test_jaccard_similarity_partial_overlap():
    a = {"Drama", "Sci-Fi"}
    b = {"Drama", "Music"}

    assert _jaccard_similarity(a, b) == 1 / 3


def test_jaccard_similarity_disjoint_sets():
    a = {"Drama"}
    b = {"Comedy"}

    assert _jaccard_similarity(a, b) == 0.0


def test_jaccard_similarity_empty_sets():
    assert _jaccard_similarity(set(), set()) == 0.0


def test_rerank_for_diversity_keeps_best_movie_first():
    recommendations = [
        Recommendation(
            movie_id=1,
            title="Movie A",
            genres="Drama|Sci-Fi",
            score=1.00,
        ),
        Recommendation(
            movie_id=2,
            title="Movie B",
            genres="Drama|Sci-Fi",
            score=0.95,
        ),
        Recommendation(
            movie_id=3,
            title="Movie C",
            genres="Comedy",
            score=0.90,
        ),
    ]

    result = rerank_for_diversity(
        recommendations,
        limit=3,
        diversity_weight=0.20,
    )

    assert result[0].movie_id == 1


def test_rerank_for_diversity_can_promote_different_genre():
    recommendations = [
        Recommendation(
            movie_id=1,
            title="Movie A",
            genres="Drama|Sci-Fi",
            score=1.00,
        ),
        Recommendation(
            movie_id=2,
            title="Movie B",
            genres="Drama|Sci-Fi",
            score=0.95,
        ),
        Recommendation(
            movie_id=3,
            title="Movie C",
            genres="Comedy",
            score=0.90,
        ),
    ]

    result = rerank_for_diversity(
        recommendations,
        limit=3,
        diversity_weight=0.20,
    )

    assert [movie.movie_id for movie in result] == [
        1,
        3,
        2,
    ]


def test_rerank_for_diversity_with_zero_weight_preserves_order():
    recommendations = [
        Recommendation(
            movie_id=1,
            title="Movie A",
            genres="Drama",
            score=1.00,
        ),
        Recommendation(
            movie_id=2,
            title="Movie B",
            genres="Drama",
            score=0.90,
        ),
        Recommendation(
            movie_id=3,
            title="Movie C",
            genres="Comedy",
            score=0.80,
        ),
    ]

    result = rerank_for_diversity(
        recommendations,
        limit=3,
        diversity_weight=0.0,
    )

    assert [movie.movie_id for movie in result] == [
        1,
        2,
        3,
    ]


def test_rerank_for_diversity_respects_limit():
    recommendations = [
        Recommendation(
            movie_id=1,
            title="Movie A",
            genres="Drama",
            score=1.00,
        ),
        Recommendation(
            movie_id=2,
            title="Movie B",
            genres="Comedy",
            score=0.90,
        ),
        Recommendation(
            movie_id=3,
            title="Movie C",
            genres="Action",
            score=0.80,
        ),
    ]

    result = rerank_for_diversity(
        recommendations,
        limit=2,
    )

    assert len(result) == 2


def test_rerank_for_diversity_nonpositive_limit_returns_empty():
    recommendations = [
        Recommendation(
            movie_id=1,
            title="Movie A",
            genres="Drama",
            score=1.00,
        )
    ]

    assert rerank_for_diversity(
        recommendations,
        limit=0,
    ) == []

    assert rerank_for_diversity(
        recommendations,
        limit=-1,
    ) == []


def test_rerank_for_diversity_empty_input_returns_empty():
    assert rerank_for_diversity(
        [],
        limit=10,
    ) == []


def test_rerank_for_diversity_handles_missing_genres():
    recommendations = [
        Recommendation(
            movie_id=1,
            title="Movie A",
            genres="",
            score=1.00,
        ),
        Recommendation(
            movie_id=2,
            title="Movie B",
            genres="Drama",
            score=0.90,
        ),
    ]

    result = rerank_for_diversity(
        recommendations,
        limit=2,
    )

    assert [movie.movie_id for movie in result] == [
        1,
        2,
    ]