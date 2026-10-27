"""Experimental rating-anchored pairwise MF; independent of the default model."""
from pathlib import Path
from numbers import Integral, Real
import copy
import os
import tempfile

import joblib
import numpy as np

from .matrix_factorization import BiasedMatrixFactorization

DEFAULT_PATH = Path(__file__).resolve().parent / "artifacts/pairwise/selected_pairwise.joblib"


class PairwiseMatrixFactorization:
    """Warm-start MF, then jointly minimize rating error and pairwise logistic loss.

    Only observed unequal-rating pairs are sampled; unrated movies are never
    treated as negatives. predict returns bounded rating estimates, not final
    recommendations. Confidence retains the baseline's evidence heuristic.
    """

    def __init__(self, rank_weight=0.1, n_epochs=5, learning_rate=0.002,
                 regularization=0.02, baseline_epochs=40, n_factors=20,
                 random_state=42):
        for name, value in (("rank_weight", rank_weight), ("learning_rate", learning_rate),
                            ("regularization", regularization)):
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        for name, value in (("n_epochs", n_epochs), ("baseline_epochs", baseline_epochs), ("n_factors", n_factors)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        BiasedMatrixFactorization(random_state=random_state)
        self.rank_weight = rank_weight
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.baseline_epochs = baseline_epochs
        self.n_factors = n_factors
        self.random_state = random_state
        self.model_ = None

    def get_params(self):
        return {key: getattr(self, key) for key in (
            "rank_weight", "n_epochs", "learning_rate", "regularization",
            "baseline_epochs", "n_factors", "random_state")}

    def fit(self, ratings):
        if not {"userId", "movieId", "rating"} <= set(ratings.columns):
            raise ValueError("Training requires userId, movieId and rating")
        if ratings.duplicated(["userId", "movieId"]).any():
            raise ValueError("Duplicate user/movie ratings must be resolved")
        baseline = BiasedMatrixFactorization(n_epochs=self.baseline_epochs,
                                            n_factors=self.n_factors,
                                            random_state=self.random_state).fit(ratings)
        return self._fit_from_baseline(ratings, baseline)

    def _fit_from_baseline(self, ratings, baseline):
        """Experiment-only reuse: caller must fit baseline on exactly ratings."""
        model = copy.deepcopy(baseline)
        model.shrink_latent = False
        w = model.weights_
        rng = np.random.RandomState(self.random_state)
        pair_rng = np.random.RandomState(self.random_state ^ 0x5A5A5A5A)
        users = ratings.userId.map(w.user_to_idx).to_numpy(dtype=int)
        movies = ratings.movieId.map(w.movie_to_idx).to_numpy(dtype=int)
        actual = ratings.rating.to_numpy(dtype=float)
        # Each row points to alternative rating buckets for the same user.
        # Memory is O(observations), not O(all possible pairs).
        groups = {}
        for u in np.unique(users):
            indices = np.flatnonzero(users == u)
            groups[u] = {r: indices[actual[indices] == r] for r in np.unique(actual[indices])}
        alternatives = {(u, r): [v for key, v in buckets.items() if key != r]
                        for u, buckets in groups.items() for r in buckets}
        history = []
        lr, reg = self.learning_rate, self.regularization
        for epoch in range(self.n_epochs):
            losses, pairs = [], 0
            for row in rng.permutation(len(actual)):
                u, i, rating = users[row], movies[row], actual[row]
                # Rating anchor: squared-error SGD preserves a star-scale objective.
                p, q = w.user_factors[u].copy(), w.movie_factors[i].copy()
                error = rating - (w.global_mean + w.user_biases[u] + w.movie_biases[i] + p @ q)
                w.user_biases[u] += lr * (error - reg * w.user_biases[u])
                w.movie_biases[i] += lr * (error - reg * w.movie_biases[i])
                w.user_factors[u] += lr * (error * q - reg * p)
                w.movie_factors[i] += lr * (error * p - reg * q)
                buckets = alternatives[u, rating]
                if not buckets or self.rank_weight == 0:
                    continue
                bucket = buckets[pair_rng.randint(len(buckets))]
                other = bucket[pair_rng.randint(len(bucket))]
                high, low = (i, movies[other]) if rating > actual[other] else (movies[other], i)
                p = w.user_factors[u].copy()
                qi, qj = w.movie_factors[high].copy(), w.movie_factors[low].copy()
                margin = w.movie_biases[high] - w.movie_biases[low] + p @ (qi - qj)
                # sigmoid(-margin), written without an overflowing exponential.
                gradient = self.rank_weight * np.exp(-np.logaddexp(0., margin))
                w.user_factors[u] += lr * gradient * (qi - qj)
                w.movie_factors[high] += lr * gradient * p
                w.movie_factors[low] -= lr * gradient * p
                w.movie_biases[high] += lr * gradient
                w.movie_biases[low] -= lr * gradient
                losses.append(float(np.logaddexp(0., -margin)))
                pairs += 1
            if not all(np.isfinite(a).all() for a in (
                    w.user_biases, w.movie_biases, w.user_factors, w.movie_factors)):
                raise ValueError("Pairwise training diverged; reduce learning_rate")
            scores = (w.global_mean + w.user_biases[users] + w.movie_biases[movies]
                      + np.einsum('ij,ij->i', w.user_factors[users], w.movie_factors[movies]))
            if not np.isfinite(scores).all():
                raise ValueError("Pairwise predictions diverged; reduce learning_rate")
            history.append({"epoch": epoch + 1, "rating_rmse": float(np.sqrt(np.mean((scores - actual)**2))),
                            "sampled_pairs": pairs,
                            "online_pairwise_loss": float(np.mean(losses)) if losses else None})
        self.model_ = model
        self.training_history_ = history
        self.metadata_ = {"training_rows": len(ratings), "objective": "squared rating error + weighted pairwise logistic loss",
                          "pair_sampling": "one alternative rating bucket per observed row per epoch; training only"}
        return self

    def predict(self, user_ids, movie_ids):
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predicting")
        return self.model_.predict(user_ids, movie_ids)

    def save(self, path=DEFAULT_PATH):
        self._validate()
        path = Path(path)
        protected = Path(__file__).resolve().parent / "artifacts"
        if path.resolve() in {(protected / "selected_mf.joblib").resolve(),
                              (protected / "collaborative_mf.joblib").resolve()}:
            raise ValueError("Cannot overwrite a baseline artifact")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".joblib")
        os.close(handle)
        try:
            joblib.dump({"format_version": 1, "model_type": "pairwise_mf", "model": self}, temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _validate(self):
        type(self)(**self.get_params())
        if not isinstance(self.model_, BiasedMatrixFactorization) or not self.model_.is_fitted_:
            raise ValueError("Expected fitted pairwise model")
        BiasedMatrixFactorization(**self.model_.get_params())
        if self.model_.n_factors != self.n_factors or self.model_.shrink_latent:
            raise ValueError("Incompatible internal model settings")
        w = self.model_.weights_
        if w is None or not np.isfinite(w.global_mean):
            raise ValueError("Invalid model weights")
        for mapping, counts, biases, factors in (
            (w.user_to_idx, w.user_counts, w.user_biases, w.user_factors),
            (w.movie_to_idx, w.movie_counts, w.movie_biases, w.movie_factors)):
            if (set(mapping.values()) != set(range(len(mapping)))
                    or any(counts.get(k, 0) <= 0 for k in mapping)
                    or biases.shape != (len(mapping),)
                    or factors.shape != (len(mapping), self.n_factors)
                    or not np.isfinite(biases).all() or not np.isfinite(factors).all()):
                raise ValueError("Invalid model weights")

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        """Load only trusted joblib files (pickle serialization)."""
        envelope = joblib.load(path)
        if (not isinstance(envelope, dict) or envelope.get("format_version") != 1
                or envelope.get("model_type") != "pairwise_mf"
                or type(envelope.get("model")) is not cls):
            raise ValueError("Not a supported pairwise MF artifact")
        model = envelope["model"]
        model._validate()
        return model
