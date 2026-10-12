"""Validation-selected EASE benchmark. Run python -m src.collaborative.evaluation.ease_experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import scipy
import joblib

from src.collaborative.ease import EASE
from src.collaborative.matrix_factorization import BiasedMatrixFactorization
from .benchmark import shared_split, summarize_predictions
from .pairwise_experiment import paired_interval

ROOT = Path(__file__).resolve().parents[1]


def ranking_summary(rows, scores):
    summary, details = summarize_predictions(rows, scores)
    # Raw EASE scores are not star predictions: do not report their MAE/RMSE.
    summary.pop('mae')
    summary.pop('rmse')
    return summary, details


def run_experiment(ratings, regularizations=(1., 10., 100., 1000.),
                   baseline_seeds=(42, 43, 44), baseline_epochs=40):
    if not regularizations or not baseline_seeds or len(set(baseline_seeds)) != len(baseline_seeds):
        raise ValueError('Nonempty regularization candidates and distinct baseline seeds required')
    splits = shared_split(ratings)
    train, validation, test = [splits.loc[splits.split == part] for part in ('train', 'validation', 'test')]
    candidates, best, best_key = [], None, None
    for penalty in regularizations:
        start = time.perf_counter()
        model = EASE(regularization=penalty).fit(train)
        metrics, _ = ranking_summary(validation, model.predict_ranking(validation.userId.tolist(), validation.movieId.tolist()).ranking_score)
        if metrics['ndcg_at_10'] is None:
            raise ValueError('Validation requires users with liked held-out movies')
        candidates.append({'regularization': penalty, 'validation_ranking': metrics,
                           'fit_and_score_seconds': time.perf_counter() - start})
        key = (metrics['ndcg_at_10'], penalty)  # Exact ties favor stronger regularization.
        if best_key is None or key > best_key:
            best, best_key = model, key
        print(f"lambda={penalty}: validation NDCG={metrics['ndcg_at_10']:.6f}", flush=True)
    best.calibrate(validation)
    print(f'Selection locked: lambda={best.regularization}; rating mapping fitted on validation only', flush=True)
    raw, raw_users = ranking_summary(test, best.predict_ranking(test.userId.tolist(), test.movieId.tolist()).ranking_score)
    calibrated, calibrated_users = summarize_predictions(test, best.predict(test.userId.tolist(), test.movieId.tolist()).predicted_score)
    baselines, records = [], []
    for seed in baseline_seeds:
        print(f'Fitting unchanged MF baseline seed {seed}', flush=True)
        baseline = BiasedMatrixFactorization(n_epochs=baseline_epochs, random_state=seed).fit(train)
        val, _ = summarize_predictions(validation, baseline.predict(validation.userId.tolist(), validation.movieId.tolist()).predicted_score)
        metrics, details = summarize_predictions(test, baseline.predict(test.userId.tolist(), test.movieId.tolist()).predicted_score)
        baselines.append({'seed': seed, 'validation': val, 'test': metrics})
        records.extend(dict(seed=seed, model='baseline', **r) for r in details)
        records.extend(dict(seed=seed, model='selected', **r) for r in raw_users)
        records.extend(dict(seed=seed, model='ease_rating', **r) for r in calibrated_users)
    interval = paired_interval(records)
    validation_win = best_key[0] > np.mean([b['validation']['ndcg_at_10'] for b in baselines])
    ranking_supported = bool(validation_win and interval['ci95'] is not None and interval['ci95'][0] > 0
                             and all(raw['ndcg_at_10'] > b['test']['ndcg_at_10'] for b in baselines))
    report = {
        'reference': 'https://arxiv.org/abs/1905.03375',
        'protocol': 'shared chronological 60/20/20; EASE and MF fit 60%; no refit; EASE calibration on validation only',
        'input_representation': '1 for training rating >= 4; 0 otherwise (unobserved and lower ratings both zero)',
        'selection': 'highest validation raw NDCG@10; exact ties prefer larger regularization',
        'candidates': candidates, 'selected_settings': best.get_params(), 'calibration': best.calibration_,
        'ranking_candidates': 'held-out rated movies only; binary relevance >= 4; expected gain for score ties',
        'test': {'ease_raw_ranking': raw, 'ease_calibrated_rating': calibrated, 'mf_baselines': baselines},
        'paired_raw_ndcg_difference': interval,
        'ranking_improvement_supported': ranking_supported, 'promoted': False,
        'acceptance_rule': 'validation NDCG above mean MF; test NDCG above every MF seed; paired-user bootstrap lower bound above zero; no automatic promotion',
        'metadata': best.metadata_, 'baseline_settings': {'n_epochs': baseline_epochs, 'seeds': list(baseline_seeds)},
        'split_counts': {k: int(v) for k, v in splits.split.value_counts().items()},
        'split_sha256': hashlib.sha256(splits.to_csv(index=False, lineterminator='\n').encode()).hexdigest(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scipy': scipy.__version__, 'joblib': joblib.__version__},
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (ROOT/'ease.py', Path(__file__))},
        'caveat': 'Development holdout previously examined; raw ranking and rating conversion are distinct. Clipping can create ties. Confidence is an evidence heuristic. Deterministic EASE is compared with multiple stochastic MF seeds.',
    }
    return report, best, pd.DataFrame(records), test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', type=Path, default=ROOT.parents[1]/'data/ratings.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/ease')
    args = parser.parse_args()
    report, model, users, test = run_experiment(pd.read_csv(args.ratings))
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output/'selected_ease.joblib')
    loaded = EASE.load(args.output/'selected_ease.joblib')
    for method in ('predict', 'predict_ranking'):
        pd.testing.assert_frame_equal(getattr(model, method)(test.userId.tolist(), test.movieId.tolist()),
                                      getattr(loaded, method)(test.userId.tolist(), test.movieId.tolist()), check_exact=True)
    report['save_load_exact_match'] = True
    report['input_sha256'] = hashlib.sha256(args.ratings.read_bytes()).hexdigest()
    (args.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    users.to_csv(args.output/'per_user_metrics.csv', index=False)
    print(json.dumps(report['test'], indent=2))
    print('Ranking improvement supported:', report['ranking_improvement_supported'])


if __name__ == '__main__':
    main()
