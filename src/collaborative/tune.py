"""Small validation-only search; run with python -m src.collaborative.tune."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib
from src.collaborative import BiasedMatrixFactorization, MovieAverageBaseline
from src.collaborative.evaluation.evaluator import summarize, confidence_report

ROOT = Path(__file__).resolve().parent


def tuning_split(ratings):
    required = ["userId", "movieId", "rating", "timestamp"]
    if not set(required).issubset(ratings.columns) or ratings[required].isna().any().any():
        raise ValueError("Tuning requires nonmissing userId, movieId, rating and timestamp")
    if not pd.api.types.is_numeric_dtype(ratings["timestamp"]) or not np.isfinite(ratings["timestamp"].to_numpy(dtype=float)).all():
        raise ValueError("Timestamps must be finite")
    profiles, validations, tests = [], [], []
    for _, history in ratings.groupby("userId", sort=False):
        history = history.sort_values(["timestamp", "movieId"], kind="stable")
        if len(history) < 10:
            profiles.append(history)
            continue
        a, b = int(len(history) * .6), int(len(history) * .8)
        profiles.append(history.iloc[:a])
        validations.append(history.iloc[a:b])
        tests.append(history.iloc[b:])
    if not validations:
        raise ValueError("No users eligible for validation and testing")
    return tuple(pd.concat(parts, ignore_index=True) for parts in (profiles, validations, tests))


def candidate_settings():
    base = BiasedMatrixFactorization().get_params()
    # Bounded one-factor-at-a-time search, not a claim of exhaustive optimization.
    return [base] + [dict(base, **change) for change in [
        {"n_factors": 10}, {"n_factors": 40}, {"learning_rate": .01},
        {"regularization": .1}, {"n_epochs": 40}]]


def score(model, rows):
    predictions = model.predict(rows.userId.tolist(), rows.movieId.tolist())
    frame = rows.reset_index(drop=True).copy()
    frame["mf_score"] = predictions.predicted_score.to_numpy()
    frame["confidence"] = predictions.confidence.to_numpy()
    return summarize(frame.rating, frame.mf_score), frame


def select_configuration(profile, validation, candidates=None):
    experiments = []
    candidates = candidate_settings() if candidates is None else candidates
    for index, params in enumerate(candidates):
        model = BiasedMatrixFactorization(**params).fit(profile)
        for shrink in (True, False):
            model.shrink_latent = shrink
            metrics, predictions = score(model, validation)
            experiments.append({"settings": model.get_params(), "validation": metrics,
                                "confidence_buckets": confidence_report(predictions).to_dict("records"),
                                "training_history": model.training_history_})
        print(f"Candidate {index + 1}/{len(candidates)} validated", flush=True)
    if not experiments:
        raise ValueError("At least one candidate is required")
    # Stable ties retain the earlier candidate, starting with the current default.
    best = min(experiments, key=lambda x: (x["validation"]["RMSE"], x["validation"]["MAE"]))
    return dict(best["settings"]), experiments


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(ratings, output=ROOT / "artifacts"):
    profile, validation, test = tuning_split(ratings)
    selected, experiments = select_configuration(profile, validation)
    print("Selection locked using validation only:", selected, flush=True)
    # Reconstruct the existing per-user first-80% order for a fair comparison.
    from src.collaborative.evaluation.evaluator import collaborative_holdout
    training, _ = collaborative_holdout(ratings)
    models = {"Baseline": MovieAverageBaseline().fit(training),
              "Current MF": BiasedMatrixFactorization().fit(training),
              "Selected MF": BiasedMatrixFactorization(**selected).fit(training)}
    results, buckets = {}, {}
    for name, model in models.items():
        results[name], frame = score(model, test)
        buckets[name] = confidence_report(frame).to_dict("records")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = models["Selected MF"]
    model.metadata_.update({"selection": "validation RMSE, then MAE", "split": "chronological 60/20/20",
                           "training_scope": "first 80%; final 20% excluded"})
    model.save(output / "selected_mf.joblib")
    loaded = BiasedMatrixFactorization.load(output / "selected_mf.joblib")
    pd.testing.assert_frame_equal(model.predict(test.userId.tolist(), test.movieId.tolist()),
                                  loaded.predict(test.userId.tolist(), test.movieId.tolist()), check_exact=True)
    report = {"dataset_sha256": hashlib.sha256(ratings.to_csv(index=False).encode()).hexdigest(),
              "rows": {"profile": len(profile), "validation": len(validation), "test": len(test), "refit": len(training)},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "joblib": joblib.__version__},
              "selection": "Minimum validation RMSE, MAE tiebreak; no test-based selection",
              "caveat": "Development benchmark; test portion was examined in earlier project work",
              "selected_settings": selected, "experiments": experiments,
              "test_metrics": results, "test_confidence_buckets": buckets,
              "save_load_exact_match": True}
    (output / "tuning_results.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(pd.DataFrame(results).T.to_string(), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=ROOT.parents[1] / "data/ratings.csv")
    args = parser.parse_args()
    run(pd.read_csv(args.ratings))
