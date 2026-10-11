"""Evaluate the separate ranking-oriented model; never promote it automatically.

python -m src.collaborative.evaluation.pairwise_experiment
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import joblib

from src.collaborative.matrix_factorization import BiasedMatrixFactorization
from src.collaborative.pairwise_factorization import PairwiseMatrixFactorization
from .benchmark import shared_split, summarize_predictions


ROOT = Path(__file__).resolve().parents[1]


def measure(model, rows):
    predicted = model.predict(rows.userId.tolist(), rows.movieId.tolist())
    return summarize_predictions(rows, predicted.predicted_score)


def paired_interval(records, reference="baseline", samples=2000):
    """Bootstrap users, averaging repeated seeds within user before resampling."""
    frame = pd.DataFrame(records)
    pivot = frame.pivot(index=["seed", "user_id"], columns="model", values="ndcg_at_10")
    deltas = (pivot["selected"] - pivot[reference]).dropna().groupby("user_id").mean().to_numpy()
    if not len(deltas):
        return {"users": 0, "mean_delta": None, "ci95": None}
    rng = np.random.RandomState(2026)
    means = np.array([rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(samples)])
    return {"users": len(deltas), "mean_delta": float(deltas.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist()}


def run_experiment(ratings, seeds=(42, 43, 44), rank_weights=(0., .1, .5),
                   baseline_epochs=40, pairwise_epochs=5):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    if 0. not in rank_weights or not any(w > 0 for w in rank_weights) or len(set(rank_weights)) != len(rank_weights):
        raise ValueError("Include one zero-weight control and positive ranking candidates")
    splits = shared_split(ratings)
    train, validation, test = [splits.loc[splits.split == part] for part in ("train", "validation", "test")]
    fitted, experiments, history = {}, [], {}
    for seed in seeds:
        print(f"Seed {seed}: fitting unchanged MF baseline", flush=True)
        baseline = BiasedMatrixFactorization(n_epochs=baseline_epochs, random_state=seed).fit(train)
        fitted[seed] = {"baseline": baseline}
        metrics, _ = measure(baseline, validation)
        experiments.append({"seed": seed, "variant": "baseline", "rank_weight": None, "validation": metrics})
        for weight in rank_weights:
            model = PairwiseMatrixFactorization(rank_weight=weight, n_epochs=pairwise_epochs,
                                                baseline_epochs=baseline_epochs, random_state=seed)
            model._fit_from_baseline(train, baseline)
            fitted[seed][weight] = model
            metrics, _ = measure(model, validation)
            experiments.append({"seed": seed, "variant": "pairwise", "rank_weight": weight, "validation": metrics})
            history[f"{seed}/{weight}"] = model.training_history_
            print(f"  weight={weight}: validation NDCG={metrics['ndcg_at_10']}", flush=True)
    # Test labels have not been scored or used to choose weights.
    def mean_validation(weight):
        return np.mean([e["validation"]["ndcg_at_10"] for e in experiments
                        if e["variant"] == "pairwise" and e["rank_weight"] == weight])
    if any(e["validation"]["ndcg_at_10"] is None for e in experiments):
        raise ValueError("Validation requires at least one user with liked movies")
    selected_weight = max((w for w in rank_weights if w > 0), key=lambda w: (mean_validation(w), -w))
    print(f"Selection locked: rank_weight={selected_weight}", flush=True)
    tests, per_user = [], []
    for seed in seeds:
        for name, model in (("baseline", fitted[seed]["baseline"]),
                            ("rating_only_control", fitted[seed][0.]),
                            ("selected", fitted[seed][selected_weight])):
            metrics, details = measure(model, test)
            tests.append({"seed": seed, "model": name, **metrics})
            per_user.extend({"seed": seed, "model": name, **row} for row in details)
    intervals = {name: paired_interval(per_user, name) for name in ("baseline", "rating_only_control")}
    baseline_validation = np.mean([e["validation"]["ndcg_at_10"] for e in experiments if e["variant"] == "baseline"])
    validation_win = mean_validation(selected_weight) > max(baseline_validation, mean_validation(0.))
    consistent_win = all(
        next(r for r in tests if r["seed"] == seed and r["model"] == "selected")["ndcg_at_10"] is not None
        and all(next(r for r in tests if r["seed"] == seed and r["model"] == "selected")["ndcg_at_10"] >
                next(r for r in tests if r["seed"] == seed and r["model"] == reference)["ndcg_at_10"]
                for reference in ("baseline", "rating_only_control")) for seed in seeds)
    rating_guardrail = all(next(r for r in tests if r["seed"] == seed and r["model"] == "selected")["rmse"] <=
                           1.01 * next(r for r in tests if r["seed"] == seed and r["model"] == "baseline")["rmse"] for seed in seeds)
    supported = (validation_win and consistent_win and rating_guardrail
                 and all(v["ci95"] is not None and v["ci95"][0] > 0 for v in intervals.values())
                 and len(seeds) >= 3)
    report = {
        "protocol": "shared chronological 60/20/20; train 60%; validation-only selection; no refit",
        "hypothesis": "Pairwise logistic updates improve NDCG@10 beyond MF and matched rating-only extra training",
        "ranking_candidates": "held-out rated movies; binary relevance rating >= 4; expected gain for ties",
        "selection": "mean validation NDCG@10 across fixed seeds; ties prefer smaller positive weight",
        "selected_rank_weight": selected_weight,
        "selected_settings": fitted[seeds[0]][selected_weight].get_params(),
        "rank_weights": list(rank_weights), "seeds": list(seeds),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ROOT / "pairwise_factorization.py", Path(__file__))},
        "split_sha256": hashlib.sha256(splits.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "split_counts": {k: int(v) for k, v in splits.split.value_counts().items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "validation": experiments, "training_history": history, "test": tests,
        "paired_ndcg_differences": intervals,
        "acceptance_rule": "Beat baseline AND rating-only control in mean validation NDCG, every test seed and paired-user bootstrap CI lower bound; at least 3 seeds; RMSE <= 1.01 * baseline each seed",
        "checks": {"validation_win": bool(validation_win), "all_test_seeds_win": bool(consistent_win),
                   "rating_guardrail": bool(rating_guardrail)},
        "improvement_supported": bool(supported), "promoted": False,
        "caveat": "Previously examined development holdout; bootstrap describes user variation, not fresh-data generalization. No automatic replacement. Small fixed search.",
    }
    return report, pd.DataFrame(per_user), fitted[seeds[0]][selected_weight], test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=ROOT.parents[1] / "data/ratings.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/pairwise")
    args = parser.parse_args()
    report, users, model, test = run_experiment(pd.read_csv(args.ratings))
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = args.output / "selected_pairwise.joblib"
    model.save(artifact)
    restored = PairwiseMatrixFactorization.load(artifact)
    pd.testing.assert_frame_equal(model.predict(test.userId.tolist(), test.movieId.tolist()),
                                  restored.predict(test.userId.tolist(), test.movieId.tolist()), check_exact=True)
    report["save_load_exact_match"] = True
    report["input_sha256"] = hashlib.sha256(args.ratings.read_bytes()).hexdigest()
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    users.to_csv(args.output / "per_user_metrics.csv", index=False)
    print(pd.DataFrame(report["test"]).groupby("model")[["rmse", "mae", "ndcg_at_10", "pairwise_accuracy_macro"]].mean())
    print("Improvement supported:", report["improvement_supported"], "(baseline unchanged)")


if __name__ == "__main__":
    main()
