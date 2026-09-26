from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from numbers import Integral, Real

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "collaborative_mf.joblib"


def checked_ids(values, name):
    result = []
    for value in values:
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or value < 0 or value > np.iinfo(np.int64).max
                or not np.isfinite(value) or value != int(value)):
            raise ValueError(f"{name} must contain finite nonnegative integer IDs")
        result.append(int(value))
    return result



@dataclass
class ModelWeights:
    global_mean: float
    user_biases: np.ndarray
    movie_biases: np.ndarray
    user_factors: np.ndarray
    movie_factors: np.ndarray
    user_to_idx: dict[int, int]
    movie_to_idx: dict[int, int]
    user_counts: dict[int, int]
    movie_counts: dict[int, int]


class BiasedMatrixFactorization:
    # Collaborative Filtering using Latent Taste Vectors & Biases (FunkSVD).

    def __init__(
        self,
        n_factors: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        n_epochs: int = 20,
        prior_strength: float = 5.0,
        random_state: int = 42,
        shrink_latent: bool = True,
    ):
        for name, value in [("n_factors", n_factors), ("n_epochs", n_epochs)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in [("learning_rate", learning_rate), ("regularization", regularization), ("prior_strength", prior_strength)]:
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and numeric")
        if learning_rate <= 0 or regularization < 0:
            raise ValueError("learning_rate must be positive and regularization nonnegative")
        if not isinstance(shrink_latent, bool):
            raise ValueError("shrink_latent must be boolean")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral) or not 0 <= random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32)")
        if not np.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError("prior_strength must be finite and positive")
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.prior_strength = prior_strength
        self.random_state = random_state
        self.shrink_latent = shrink_latent

        self.weights_: ModelWeights | None = None
        self.is_fitted_ = False

    def fit(self, ratings: pd.DataFrame) -> BiasedMatrixFactorization:
        required_cols = {"userId", "movieId", "rating"}
        missing = required_cols - set(ratings.columns)
        if missing:
            raise ValueError(
                f"Ratings DataFrame is missing columns: {sorted(missing)}")

        # Reject invalid rows rather than silently changing the training sample.
        clean_ratings = ratings[["userId", "movieId", "rating"]].copy()
        clean_ratings["userId"] = checked_ids(clean_ratings["userId"], "userId")
        clean_ratings["movieId"] = checked_ids(clean_ratings["movieId"], "movieId")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
               or not np.isfinite(v) or not .5 <= v <= 5 for v in clean_ratings["rating"]):
            raise ValueError("rating must contain finite numeric values from 0.5 to 5")
        if clean_ratings.empty:
            raise ValueError("Ratings cannot be empty.")

        rng = np.random.RandomState(self.random_state)

        unique_users = clean_ratings["userId"].unique()
        unique_movies = clean_ratings["movieId"].unique()

        user_to_idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
        movie_to_idx = {int(mid): idx for idx, mid in enumerate(unique_movies)}

        n_users = len(user_to_idx)
        n_movies = len(movie_to_idx)
        user_counts = clean_ratings["userId"].value_counts().to_dict()
        movie_counts = clean_ratings["movieId"].value_counts().to_dict()
        user_indices = clean_ratings["userId"].map(
            user_to_idx).to_numpy(dtype=np.int32)
        movie_indices = clean_ratings["movieId"].map(
            movie_to_idx).to_numpy(dtype=np.int32)
        rating_values = clean_ratings["rating"].to_numpy(dtype=np.float64)

        global_mean = float(rating_values.mean())
        user_biases = np.zeros(n_users, dtype=np.float64)
        movie_biases = np.zeros(n_movies, dtype=np.float64)

        user_factors = rng.normal(0.0, 0.05, size=(n_users, self.n_factors))
        movie_factors = rng.normal(0.0, 0.05, size=(n_movies, self.n_factors))

        lr = self.learning_rate
        reg = self.regularization
        n_samples = len(rating_values)

        self.training_history_ = []
        for epoch in range(self.n_epochs):
            shuffle_order = rng.permutation(n_samples)

            for sample_idx in shuffle_order:
                u = user_indices[sample_idx]
                i = movie_indices[sample_idx]
                r = rating_values[sample_idx]

                # Current prediction
                pred = (
                    global_mean
                    + user_biases[u]
                    + movie_biases[i]
                    + np.dot(user_factors[u], movie_factors[i])
                )
                err = r - pred

                user_biases[u] += lr * (err - reg * user_biases[u])
                movie_biases[i] += lr * (err - reg * movie_biases[i])

                u_factors_prev = user_factors[u].copy()
                user_factors[u] += lr * \
                    (err * movie_factors[i] - reg * user_factors[u])
                movie_factors[i] += lr * \
                    (err * u_factors_prev - reg * movie_factors[i])

            if not all(np.isfinite(x).all() for x in (user_biases, movie_biases, user_factors, movie_factors)):
                raise ValueError("Training diverged; reduce learning_rate")
            fitted = (global_mean + user_biases[user_indices] + movie_biases[movie_indices]
                      + np.einsum("ij,ij->i", user_factors[user_indices], movie_factors[movie_indices]))
            if not np.isfinite(fitted).all():
                raise ValueError("Training predictions diverged; reduce learning_rate")
            self.training_history_.append({"epoch": epoch + 1,
                                           "rmse": float(np.sqrt(np.mean((rating_values - fitted) ** 2)))})

        self.weights_ = ModelWeights(
            global_mean=global_mean,
            user_biases=user_biases,
            movie_biases=movie_biases,
            user_factors=user_factors,
            movie_factors=movie_factors,
            user_to_idx=user_to_idx,
            movie_to_idx=movie_to_idx,
            user_counts={int(k): int(v) for k, v in user_counts.items()},
            movie_counts={int(k): int(v) for k, v in movie_counts.items()},
        )
        self.is_fitted_ = True
        self.metadata_ = {"training_rows": len(clean_ratings), "users": n_users, "movies": n_movies}
        return self

    def get_params(self):
        return {name: getattr(self, name) for name in (
            "n_factors", "learning_rate", "regularization", "n_epochs",
            "prior_strength", "random_state", "shrink_latent")}

    def predict(
        self,
        user_ids: list[int] | pd.Series,
        movie_ids: list[int] | pd.Series,
    ) -> pd.DataFrame:

        if not self.is_fitted_ or self.weights_ is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        if len(user_ids) != len(movie_ids):
            raise ValueError(
                "user_ids and movie_ids must have the same length")

        user_ids = checked_ids(user_ids, "user_ids")
        movie_ids = checked_ids(movie_ids, "movie_ids")
        w = self.weights_
        predictions = []

        for uid, mid in zip(user_ids, movie_ids):
            uid = int(uid)
            mid = int(mid)

            has_user = uid in w.user_to_idx
            has_movie = mid in w.movie_to_idx

            if has_user and has_movie:
                u_idx = w.user_to_idx[uid]
                m_idx = w.movie_to_idx[mid]

                score = (
                    w.global_mean
                    + w.user_biases[u_idx]
                    + w.movie_biases[m_idx]
                    + np.dot(w.user_factors[u_idx], w.movie_factors[m_idx])
                )

                u_count = w.user_counts.get(uid, 0)
                m_count = w.movie_counts.get(mid, 0)
                u_conf = u_count / (u_count + self.prior_strength)
                m_conf = m_count / (m_count + self.prior_strength)
                confidence = u_conf * m_conf
                # Weak evidence should not receive the full latent adjustment.
                # Keep learned biases as the fallback; training regularizes them.
                if getattr(self, "shrink_latent", False):
                    interaction = np.dot(w.user_factors[u_idx], w.movie_factors[m_idx])
                    score -= (1.0 - confidence) * interaction

            elif has_user:

                u_idx = w.user_to_idx[uid]
                score = w.global_mean + w.user_biases[u_idx]
                confidence = 0.1

            elif has_movie:

                m_idx = w.movie_to_idx[mid]
                score = w.global_mean + w.movie_biases[m_idx]
                m_count = w.movie_counts.get(mid, 0)
                confidence = 0.5 * (m_count / (m_count + self.prior_strength))

            else:

                score = w.global_mean
                confidence = 0.0

            clamped_score = max(0.5, min(5.0, float(score)))

            predictions.append(
                {
                    "user_id": uid,
                    "movie_id": mid,
                    "predicted_score": clamped_score,
                    "confidence": float(confidence),
                }
            )

        return pd.DataFrame(
            predictions,
            columns=["user_id", "movie_id", "predicted_score", "confidence"],
        )

    def save(self, filepath: str | Path = DEFAULT_MODEL_PATH) -> None:
        if not self.is_fitted_ or self.weights_ is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace only after serialization succeeds.
        import os
        import tempfile
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".joblib")
        os.close(handle)
        try:
            joblib.dump({"format_version": 1, "settings": self.get_params(),
                         "metadata": getattr(self, "metadata_", {}), "model": self}, temporary)
            os.replace(temporary, path)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()

    @classmethod
    def load(cls, filepath: str | Path = DEFAULT_MODEL_PATH) -> BiasedMatrixFactorization:
        """Load trusted local joblib files only (pickle can execute code)."""
        artifact = joblib.load(filepath)
        if isinstance(artifact, dict):
            if artifact.get("format_version") != 1:
                raise ValueError("Unsupported collaborative model format")
            model = artifact.get("model")
        else:
            model = artifact  # Compatibility with legacy raw model objects.
        if not isinstance(model, cls) or not model.is_fitted_ or model.weights_ is None:
            raise ValueError("Artifact does not contain a fitted collaborative model")
        if not hasattr(model, "shrink_latent"):
            model.shrink_latent = False
        cls(**model.get_params())  # Validate restored settings.
        if isinstance(artifact, dict) and artifact.get("settings") != model.get_params():
            raise ValueError("Artifact settings do not match model")
        w = model.weights_
        shapes = [(w.user_biases, (len(w.user_to_idx),)),
                  (w.movie_biases, (len(w.movie_to_idx),)),
                  (w.user_factors, (len(w.user_to_idx), model.n_factors)),
                  (w.movie_factors, (len(w.movie_to_idx), model.n_factors))]
        if not np.isfinite(w.global_mean) or any(x.shape != shape or not np.isfinite(x).all() for x, shape in shapes):
            raise ValueError("Invalid collaborative model weights")
        for mapping, counts in [(w.user_to_idx, w.user_counts), (w.movie_to_idx, w.movie_counts)]:
            if set(mapping.values()) != set(range(len(mapping))) or any(counts.get(k, 0) <= 0 for k in mapping):
                raise ValueError("Invalid collaborative model IDs or counts")
        return model
