"""Development and single-use reserved offline evaluation of prepared 32M cohorts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.data_processing.movielens import movie_metadata_from_records, user_ratings_from_records
from src.evaluate_catalog import run_benchmark, write_json
from src.evaluate_content import candidate_from_dict, read_csv
from src.evaluation.catalog import CatalogCase, validate_ratings
from src.evaluation.confirmation import claim_once, identity, source_hashes
from src.prepare_ml32m import file_sha, split_history


def workflow_hashes():
    return {**source_hashes(), **{p: file_sha(p) for p in
            ("src/prepare_ml32m.py", "src/evaluate_ml32m.py")}}


def verify_dataset(data_dir, cohort=None):
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "MovieLens 32M":
        raise ValueError("Unsupported prepared dataset")
    required = {"metadata/movies.csv", "metadata/availability.csv",
                "development/ratings.csv", "reserved/ratings.csv", "README.txt"}
    if set(manifest["files"]) != required:
        raise ValueError("Unexpected dataset manifest files")
    for name, expected in manifest["files"].items():
        if cohort and name.endswith("/ratings.csv") and not name.startswith(cohort + "/"):
            continue  # Development does not open or hash reserved labels.
        if file_sha(root / name) != expected:
            raise ValueError(f"Prepared dataset changed: {name}")
    return manifest


def freeze(data_dir, preset, plan_path, minimum_effect=.01):
    if minimum_effect <= 0:
        raise ValueError("Minimum effect must be positive")
    root = Path(data_dir).resolve()
    manifest = verify_dataset(root)
    candidate_from_dict(json.loads(Path(preset).read_text())["candidate"])
    plan = {"schema_version": 1, "mode": "ml32m_reserved_offline", "data_dir": str(root),
            "manifest_sha256": file_sha(root / "manifest.json"),
            "preset": str(Path(preset).resolve()), "preset_sha256": file_sha(preset),
            "reserved_sha256": manifest["files"]["reserved/ratings.csv"],
            "source_sha256": workflow_hashes(), "primary_metric": "ndcg_at_10",
            "comparison": "selected content preset versus same structured core without text",
            "minimum_effect": minimum_effect, "split": manifest["split"],
            "status": "frozen before reserved metrics; stop tuning this comparison"}
    plan["plan_id"] = identity(plan)
    path = Path(plan_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        json.dump(plan, target, indent=2)
        target.write("\n")
    return plan


def verify_frozen(plan):
    if (plan.get("schema_version") != 1 or plan.get("mode") != "ml32m_reserved_offline"
            or identity(plan) != plan.get("plan_id")):
        raise ValueError("Reserved evaluation plan changed or has an unsupported schema")
    if workflow_hashes() != plan["source_sha256"]:
        raise ValueError("Source changed after freezing; use the frozen checkout")
    if file_sha(Path(plan["data_dir"]) / "manifest.json") != plan["manifest_sha256"]:
        raise ValueError("Dataset manifest changed after freezing")
    if file_sha(plan["preset"]) != plan["preset_sha256"]:
        raise ValueError("Preset changed after freezing")
    # Byte integrity only; no ratings are parsed before the single-use claim.
    verify_dataset(plan["data_dir"])


def cohort_cases(ratings):
    validate_ratings(ratings)
    grouped = defaultdict(list)
    for row in ratings:
        grouped[row.user_id].append(row)
    result = []
    for user, rows in sorted(grouped.items()):
        history, outcomes = split_history(rows)
        if len(rows) < 20 or len(history) < 5 or len(outcomes) < 2:
            raise ValueError("Prepared cohort contains an ineligible user")
        result.append(CatalogCase(user, max(r.timestamp for r in history), history, outcomes))
    return result


def evaluate(data_dir, preset_path, output, *, cohort="development", limit=None, plan=None):
    if cohort == "reserved" and plan is None:
        raise ValueError("Reserved evaluation requires a frozen plan and single-use claim")
    if cohort not in ("development", "reserved") or (limit is not None and limit <= 0):
        raise ValueError("Invalid cohort or smoke limit")
    root = Path(data_dir)
    manifest = verify_dataset(root, cohort)
    movies = movie_metadata_from_records(read_csv(root / "metadata/movies.csv"))
    availability = {int(r["movieId"]): int(r["timestamp"])
                    for r in read_csv(root / "metadata/availability.csv")}
    cases = cohort_cases(user_ratings_from_records(read_csv(root / cohort / "ratings.csv")))
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("Empty evaluation cohort")
    preset = candidate_from_dict(json.loads(Path(preset_path).read_text())["candidate"])
    protocol = {"mode": f"ml32m_{cohort}_offline", "plan_id": plan["plan_id"] if plan else None,
                "split": manifest["split"], "catalog_availability": manifest["availability"],
                "catalog_scope": manifest["catalog"], "metadata_temporality": manifest["metadata"],
                "cohort_eligibility": manifest["eligibility"], "limitations": manifest["limitations"],
                "smoke_limit": limit, "test_status": "reserved offline evaluation" if plan else "development only"}
    return run_benchmark(cases, movies, (), availability, preset, output, protocol,
                         minimum_effect=plan["minimum_effect"] if plan else .01,
                         input_hashes={"manifest": file_sha(root / "manifest.json"),
                                       "preset": file_sha(preset_path),
                                       **{k: v for k, v in manifest["files"].items()
                                          if k.startswith("metadata/") or k.startswith(cohort + "/")}})


def evaluate_reserved(plan_path, output, state_dir=".cache/evaluation/ml32m"):
    plan = json.loads(Path(plan_path).read_text())
    verify_frozen(plan)
    # Keyed to the actual reserved data, so freezing another preset cannot reopen it.
    claim = claim_once({"plan_id": plan["reserved_sha256"]}, state_dir)
    try:
        result = evaluate(plan["data_dir"], plan["preset"], output, cohort="reserved", plan=plan)
        write_json(claim, {"plan_id": plan["plan_id"], "status": "complete", "report": str(output)})
        return result
    except Exception:
        write_json(claim, {"plan_id": plan["plan_id"], "status": "failed after opening; audit before rerun"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("develop", "freeze"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", default="data/ml32m")
        sub.add_argument("--preset", default="reports/tmdb-full/selected_config.json")
        if name == "develop":
            sub.add_argument("--limit", type=int, help="Development smoke run only")
            sub.add_argument("--output", help="Defaults to separate smoke/full development directories")
        else:
            sub.add_argument("--plan", default="reports/ml32m/offline-plan.json")
            sub.add_argument("--minimum-effect", type=float, default=.01)
    sub = commands.add_parser("evaluate-reserved")
    sub.add_argument("--plan", default="reports/ml32m/offline-plan.json")
    sub.add_argument("--output", default="reports/ml32m/reserved")
    args = parser.parse_args()
    try:
        if args.command == "develop":
            output = args.output or ("reports/ml32m/development-smoke" if args.limit else "reports/ml32m/development")
            evaluate(args.data_dir, args.preset, output, limit=args.limit)
        elif args.command == "freeze":
            plan = freeze(args.data_dir, args.preset, args.plan, args.minimum_effect)
            print(f"Frozen {plan['plan_id']}. Reserved metrics have not been computed.")
        else:
            evaluate_reserved(args.plan, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Evaluation stopped: {exc}\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
