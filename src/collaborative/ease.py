"""EASE with exact user-space solution and separately calibrated rating output.

Reference: Harald Steck, https://arxiv.org/abs/1905.03375.
"""
from pathlib import Path
from numbers import Real
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

from .matrix_factorization import checked_ids

DEFAULT_PATH = Path(__file__).resolve().parent / 'artifacts/ease/selected_ease.joblib'


def validate_ratings(ratings):
    if not {'userId', 'movieId', 'rating'} <= set(ratings) or ratings.empty:
        raise ValueError('Nonempty userId, movieId and rating columns required')
    checked_ids(ratings.userId, 'userId')
    checked_ids(ratings.movieId, 'movieId')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
           or not np.isfinite(v) or not .5 <= v <= 5 for v in ratings.rating):
        raise ValueError('Ratings must be finite numeric values in [0.5, 5]')
    if ratings.duplicated(['userId', 'movieId']).any():
        raise ValueError('Duplicate user/movie ratings must be resolved')


class EASE:
    """Binary-like EASE; cached exact scores for fitted users and movies.

    predict_ranking returns unconstrained ranking scores. After calibrate(),
    predict returns the common bounded rating/confidence contract. Refit to
    incorporate new histories; unknown IDs receive raw score/confidence zero.
    """

    def __init__(self, regularization=100., like_threshold=4., prior_strength=5.,
                 max_working_mb=512.):
        for name, value in (('regularization', regularization), ('prior_strength', prior_strength),
                            ('max_working_mb', max_working_mb), ('like_threshold', like_threshold)):
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not .5 <= like_threshold <= 5:
            raise ValueError('like_threshold must be in [0.5, 5]')
        self.regularization = regularization
        self.like_threshold = like_threshold
        self.prior_strength = prior_strength
        self.max_working_mb = max_working_mb
        self.scores_ = None
        self.calibration_ = None

    def get_params(self):
        return {k: getattr(self, k) for k in ('regularization', 'like_threshold', 'prior_strength', 'max_working_mb')}

    def fit(self, ratings):
        validate_ratings(ratings)
        users = sorted(int(u) for u in ratings.userId.unique())
        movies = sorted(int(m) for m in ratings.movieId.unique())
        n, m = len(users), len(movies)
        # Conservative allowance for dense solve inputs, output and work arrays.
        estimate = 8 * (4*n*n + 5*n*m)
        if estimate > self.max_working_mb * 1024**2:
            raise ValueError('EASE user-space solve exceeds configured memory budget')
        user_map = {u: i for i, u in enumerate(users)}
        movie_map = {v: i for i, v in enumerate(movies)}
        liked = ratings.loc[ratings.rating >= self.like_threshold]
        x = csr_matrix((np.ones(len(liked)), (liked.userId.map(user_map), liked.movieId.map(movie_map))), shape=(n, m))
        # Woodbury: P=(I-X.T @ Z)/lambda, Z=(lambda I+X X.T)^-1 X.
        # XB = X - (X P)/diag(P), and X P = Z. No item-square matrix.
        a = (x @ x.T).toarray()
        a.flat[::n+1] += self.regularization
        z = cho_solve(cho_factor(a, lower=True, check_finite=False), x.toarray(), check_finite=False)
        diagonal = (1. - np.asarray(x.multiply(z).sum(axis=0)).ravel()) / self.regularization
        if not np.isfinite(diagonal).all() or (diagonal <= 0).any():
            raise ValueError('Unstable EASE solution; increase regularization')
        scores = x.toarray() - z / diagonal
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite EASE scores')
        self.user_to_idx_ = user_map
        self.movie_to_idx_ = movie_map
        self.user_counts_ = np.asarray(x.sum(axis=1)).ravel()
        self.movie_counts_ = np.asarray(x.sum(axis=0)).ravel()
        self.scores_ = scores
        self.calibration_ = None
        self.training_pairs_ = set(zip(ratings.userId.astype(int), ratings.movieId.astype(int)))
        self.metadata_ = {'training_rows': len(ratings), 'liked_rows': len(liked),
                          'users': n, 'movies': m, 'estimated_working_bytes': estimate,
                          'stored_score_bytes': scores.nbytes}
        return self

    def predict_ranking(self, user_ids, movie_ids):
        if self.scores_ is None:
            raise RuntimeError('Fit EASE before predicting')
        if len(user_ids) != len(movie_ids):
            raise ValueError('user_ids and movie_ids must have matching lengths')
        users, movies = checked_ids(user_ids, 'user_ids'), checked_ids(movie_ids, 'movie_ids')
        output = []
        for uid, mid in zip(users, movies):
            u, i = self.user_to_idx_.get(uid), self.movie_to_idx_.get(mid)
            score, confidence = 0., 0.
            if u is not None and i is not None:
                score = float(self.scores_[u, i])
                uc, mc = self.user_counts_[u], self.movie_counts_[i]
                confidence = float(uc / (uc + self.prior_strength) * mc / (mc + self.prior_strength))
            output.append((uid, mid, score, confidence))
        return pd.DataFrame(output, columns=['user_id', 'movie_id', 'ranking_score', 'confidence'])

    def calibrate(self, ratings):
        """Nonnegative affine least squares on disjoint validation observations.

        The global monotonic mapping preserves order before clipping. It is a
        rating conversion, not probability calibration. Test data must not be used.
        """
        validate_ratings(ratings)
        if self.scores_ is None:
            raise RuntimeError('Fit EASE before calibration')
        if any(pair in self.training_pairs_ for pair in zip(ratings.userId, ratings.movieId)):
            raise ValueError('Calibration rows must be disjoint from training pairs')
        scores = self.predict_ranking(ratings.userId.tolist(), ratings.movieId.tolist()).ranking_score.to_numpy()
        actual = ratings.rating.to_numpy(dtype=float)
        centered = scores - scores.mean()
        denominator = float(centered @ centered)
        slope = max(0., float(centered @ (actual - actual.mean())) / denominator) if denominator > 1e-12 else 0.
        self.calibration_ = {'slope': slope, 'intercept': float(actual.mean() - slope*scores.mean()),
                             'rows': len(ratings)}
        return self

    def predict(self, user_ids, movie_ids):
        if self.calibration_ is None:
            raise RuntimeError('Calibrate EASE on validation ratings before rating predictions; use predict_ranking for raw scores')
        result = self.predict_ranking(user_ids, movie_ids).rename(columns={'ranking_score': 'predicted_score'})
        result['predicted_score'] = np.clip(self.calibration_['intercept'] + self.calibration_['slope'] * result.predicted_score, .5, 5.)
        return result

    def _validate(self):
        type(self)(**self.get_params())
        if self.scores_ is None:
            raise ValueError('Cannot save/load an unfitted EASE model')
        n, m = len(self.user_to_idx_), len(self.movie_to_idx_)
        if self.scores_.shape != (n, m) or not np.isfinite(self.scores_).all():
            raise ValueError('Invalid EASE scores')
        for mapping, counts in ((self.user_to_idx_, self.user_counts_), (self.movie_to_idx_, self.movie_counts_)):
            checked_ids(list(mapping), 'artifact IDs')
            if (set(mapping.values()) != set(range(len(mapping))) or counts.shape != (len(mapping),)
                    or not np.isfinite(counts).all() or (counts < 0).any()):
                raise ValueError('Invalid EASE IDs or counts')
        if self.calibration_ is not None and (not np.isfinite(self.calibration_['slope'])
                or self.calibration_['slope'] < 0 or not np.isfinite(self.calibration_['intercept'])):
            raise ValueError('Invalid EASE calibration')

    def save(self, path=DEFAULT_PATH):
        self._validate()
        path = Path(path)
        if path.name in {'selected_mf.joblib', 'collaborative_mf.joblib', 'selected_pairwise.joblib'}:
            raise ValueError('Cannot overwrite another model artifact')
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.joblib')
        os.close(fd)
        try:
            joblib.dump({'format_version': 1, 'model_type': 'ease', 'model': self}, temporary, compress=3)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        """Load trusted joblib files only (pickle serialization)."""
        data = joblib.load(path)
        if (not isinstance(data, dict) or data.get('format_version') != 1
                or data.get('model_type') != 'ease' or type(data.get('model')) is not cls):
            raise ValueError('Not a supported EASE artifact')
        data['model']._validate()
        return data['model']
