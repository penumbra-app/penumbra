"""Experimental content representations and personalized text scorers.

Kept separate from the serving preset until validation and test evidence justify
promotion. Index fitting accepts movie metadata only, never user ratings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

from src.content.reliability import rating_weights
from src.content.text import movie_text


def fit_tfidf(documents, config):
    documents = [text for text in documents if text.strip()]
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        ngram_range=(1, config.ngram_max), min_df=config.min_df,
        max_features=config.max_features, sublinear_tf=True, norm="l2",
    )
    if not documents:
        return None
    try:
        vectorizer.fit(documents)
    except ValueError as error:
        if "empty vocabulary" in str(error) or "After pruning" in str(error):
            return None
        raise
    return vectorizer


def transform(vectorizer, documents):
    if vectorizer is None:
        return sparse.csr_matrix((len(documents), 1), dtype=np.float64)
    return vectorizer.transform(documents)


def separate_fields(plot, keywords, plot_weight):
    if not 0 <= plot_weight <= 1:
        raise ValueError("plot_weight must be in [0, 1]")
    # Squared block scales make dot products equal weighted field cosines.
    # Renormalization handles a movie missing one of the fields.
    return normalize(sparse.hstack((plot * np.sqrt(plot_weight),
                                    keywords * np.sqrt(1 - plot_weight)), format="csr"))


@dataclass
class TextIndex:
    matrix: object
    row_by_text: dict[str, int]
    source: str = "combined"

    def for_movies(self, movies):
        return self.matrix[[self.row_by_text[movie_text(m, self.source)] for m in movies]]


@dataclass
class FieldTextIndex:
    matrix: object
    row_by_fields: dict[tuple[str, str], int]

    def for_movies(self, movies):
        return self.matrix[[self.row_by_fields[(movie_text(m, "plot"), movie_text(m, "keywords"))]
                            for m in movies]]


def shared_index(catalog, documents, config, source="combined"):
    vectorizer = fit_tfidf([movie_text(m, source) for m in catalog], config)
    return TextIndex(transform(vectorizer, documents), {t: i for i, t in enumerate(documents)}, source)


def personal_matrices(history, catalog, candidates, config):
    history_movies = [catalog[r.movie_id] for r in history]
    vectorizer = fit_tfidf([movie_text(m, config.source) for m in history_movies], config)
    return (transform(vectorizer, [movie_text(m, config.source) for m in history_movies]),
            transform(vectorizer, [movie_text(m, config.source) for m in candidates]))


def history_targets(history, profile_config):
    if not history:
        return np.empty(0), np.empty(0)
    values = np.array([r.rating for r in history], dtype=float)
    weights, _ = rating_weights(tuple(history), profile_config)
    return values - values.mean(), np.array([weights[r.movie_id] for r in history])


def similarity_signal(history_vectors, candidate_vectors, residuals, weights, prior=5., top_k=None):
    if prior < 0 or (top_k is not None and (not isinstance(top_k, int) or top_k < 1)):
        raise ValueError("Invalid prior or top_k")
    scores = np.zeros(candidate_vectors.shape[0])
    mass = np.zeros_like(scores)
    if not len(residuals):
        return scores, mass
    for start in range(0, len(scores), 256):
        similarity = candidate_vectors[start:start + 256] @ history_vectors.T
        if sparse.issparse(similarity):
            similarity = similarity.toarray()
        similarity = np.clip(np.asarray(similarity, dtype=np.float64), 0., 1.)
        similarity[similarity < 1e-12] = 0
        if top_k is not None and top_k < len(residuals):
            # Stable ties follow the history's deterministic time/movie-ID order.
            remove = np.argsort(-similarity, axis=1, kind="stable")[:, top_k:]
            np.put_along_axis(similarity, remove, 0., axis=1)
        weighted = similarity * weights
        evidence = weighted.sum(axis=1)
        numerator = weighted @ residuals
        stop = start + len(evidence)
        scores[start:stop] = np.divide(numerator, evidence + prior,
                                       out=np.zeros_like(evidence), where=evidence > 0)
        mass[start:stop] = evidence
    return scores, mass


def ridge_signal(history_vectors, candidate_vectors, residuals, weights, alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive")
    if not len(residuals):
        return np.zeros(candidate_vectors.shape[0])
    # No intercept: absent content returns zero adjustment. Target and sample
    # weights match the existing text scorer; only the estimator changes.
    model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", tol=1e-8, max_iter=2000)
    model.fit(history_vectors, residuals, sample_weight=weights)
    return model.predict(candidate_vectors)


def combine(core_scores, text_signal, config):
    return np.clip(core_scores + np.clip(text_signal * config.weight,
                                        -config.max_adjustment, config.max_adjustment), 0., 5.)
