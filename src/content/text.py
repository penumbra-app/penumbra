"""Training-only TF-IDF and optional latent semantic text profiles.

Numerical dependencies are imported only by text methods. The core
content model remains usable with the Python standard library alone.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from src.content.genres import _unique_genres
from src.content.reliability import ProfileConfig, rating_weights
from src.content.schemas import MovieMetadata, UserRating


@dataclass(frozen=True)
class TextConfig:
    source: str = "combined"
    representation: str = "tfidf"
    ngram_max: int = 1
    min_df: int = 1
    max_features: int = 10000
    latent_dimensions: int = 32
    regularization_strength: float = 5.0
    weight: float = 0.5
    max_adjustment: float = 0.5

    def __post_init__(self) -> None:
        if self.source not in ("keywords", "plot", "combined"):
            raise ValueError("source must be keywords, plot, or combined")
        if self.representation not in ("tfidf", "lsa"):
            raise ValueError("representation must be tfidf or lsa")
        for name in ("ngram_max", "min_df", "max_features", "latent_dimensions"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ngram_max > 2:
            raise ValueError("ngram_max must be 1 or 2")
        for name in ("regularization_strength", "weight", "max_adjustment"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_adjustment > 5:
            raise ValueError("max_adjustment must not exceed 5 rating points")


@dataclass(frozen=True)
class TextSignal:
    raw_component: float = 0.0
    adjustment: float = 0.0
    evidence: float = 0.0
    weight: float = 0.0
    representation: str = "tfidf"


def movie_text(movie: MovieMetadata, source: str) -> str:
    """Do not silently substitute titles or genres for absent plot/keywords."""
    pieces = []
    if source in ("keywords", "combined"):
        pieces.extend(value.casefold() for value in _unique_genres(movie.keywords))
    if source in ("plot", "combined") and movie.plot:
        pieces.append(movie.plot.strip())
    return " . ".join(pieces)


class TextProfile:
    """Fit vocabulary, IDF, and optional SVD using only this user's history.

    Similarity-weighted, recency-weighted rating residuals supply the text
    adjustment. Similarity mass plus a prior shrinks sparse evidence. Candidate
    documents are transformed only: they cannot change vocabulary or IDF.
    """

    def __init__(
        self, ratings: Iterable[UserRating], movies: Mapping[int, MovieMetadata],
        config: TextConfig | None = None, profile_config: ProfileConfig | None = None,
    ) -> None:
        self.config = config or TextConfig()
        history = tuple(ratings)
        if len({r.user_id for r in history}) > 1:
            raise ValueError("TextProfile requires ratings from a single user")
        if len({r.movie_id for r in history}) != len(history):
            raise ValueError("Duplicate ratings for a user and movie are not supported")
        self.user_id = history[0].user_id if history else None
        self.rating_count = len(history)
        self.vectorizer = None
        self.reducer = None
        self._matrix = None
        self._weights = None
        self._residuals = None
        weights, _ = rating_weights(history, profile_config or ProfileConfig())
        if not history or self.config.weight == 0 or self.config.max_adjustment == 0:
            return

        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        baseline = sum(r.rating for r in history) / len(history)
        usable = [r for r in history if r.movie_id in movies
                  and movie_text(movies[r.movie_id], self.config.source).strip()]
        if len(usable) < self.config.min_df:
            return
        documents = [movie_text(movies[r.movie_id], self.config.source) for r in usable]
        vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english",
            ngram_range=(1, self.config.ngram_max), min_df=self.config.min_df,
            max_features=self.config.max_features, sublinear_tf=True, norm="l2",
        )
        try:
            matrix = vectorizer.fit_transform(documents)
        except ValueError as error:
            if "empty vocabulary" in str(error) or "After pruning, no terms remain" in str(error):
                return
            raise
        if self.config.representation == "lsa":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize

            dimensions = min(self.config.latent_dimensions, matrix.shape[0] - 1,
                             matrix.shape[1] - 1)
            # Too little text for a latent space: preserve the sparse TF-IDF signal.
            if dimensions >= 1 and matrix.shape[0] >= 3:
                self.reducer = TruncatedSVD(n_components=dimensions, random_state=42)
                matrix = normalize(self.reducer.fit_transform(matrix))
        self.vectorizer = vectorizer
        self._matrix = matrix
        self._weights = np.array([weights[r.movie_id] for r in usable])
        self._residuals = np.array([r.rating - baseline for r in usable])

    def evidence(self, movies: Iterable[MovieMetadata]):
        """Return residual numerators and similarity mass; bound temporary memory."""
        import numpy as np
        from scipy.sparse import issparse
        from sklearn.preprocessing import normalize

        candidates = tuple(movies)
        numerator, mass = np.zeros(len(candidates)), np.zeros(len(candidates))
        if self.vectorizer is None or not candidates:
            return numerator, mass
        for start in range(0, len(candidates), 256):
            documents = [movie_text(m, self.config.source) for m in candidates[start:start + 256]]
            vectors = self.vectorizer.transform(documents)
            if self.reducer is not None:
                vectors = normalize(self.reducer.transform(vectors))
            similarity = vectors @ self._matrix.T
            if issparse(similarity):
                similarity = similarity.toarray()
            # Negative LSA cosine is not evidence for a shared preference.
            similarity = np.clip(similarity, 0.0, 1.0)
            similarity[similarity < 1e-12] = 0
            weighted = similarity * self._weights
            stop = start + len(documents)
            numerator[start:stop] = weighted @ self._residuals
            mass[start:stop] = weighted.sum(axis=1)
        return numerator, mass

    def signals(self, movies: Iterable[MovieMetadata]) -> tuple[TextSignal, ...]:
        return signals_from_evidence(*self.evidence(movies), self.config)


def signals_from_evidence(numerator, mass, config: TextConfig) -> tuple[TextSignal, ...]:
    """Reuse fitted text evidence when comparing blend weights and priors."""
    from src.content.scoring import _bounded_adjustment

    signals = []
    for total, evidence in zip(numerator, mass, strict=True):
        enabled = config.weight > 0 and config.max_adjustment > 0
        raw = float(total / (evidence + config.regularization_strength)) if evidence > 0 else 0.0
        adjustment = _bounded_adjustment(raw, config.weight, config.max_adjustment)
        signals.append(TextSignal(
            raw, adjustment, float(evidence) if enabled else 0.0,
            config.weight, config.representation,
        ))
    return tuple(signals)
