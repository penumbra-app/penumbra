"""Freeze future evaluation inputs and prevent accidental early/repeated peeking.

This is a local workflow guard, not an access-control or data-authenticity system.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.data_processing.movielens import user_ratings_from_records
from src.evaluate_content import read_csv
from src.evaluation.catalog import validate_ratings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = {Path("src/evaluate_catalog.py"), Path("src/evaluate_content.py")}
    for directory in ("src/content", "src/data_processing", "src/evaluation"):
        paths.update(Path(directory).glob("*.py"))
    return {str(p): sha(p) for p in sorted(paths)}


def identity(plan):
    content = {k: v for k, v in plan.items() if k != "plan_id"}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def register_plan(movies, history, tags, preset, output, days=30, minimum_effect=.01, now=None):
    if days < 1 or minimum_effect <= 0:
        raise ValueError("Require a positive future window and minimum worthwhile effect")
    now = int(time.time()) if now is None else int(now)
    ratings = user_ratings_from_records(read_csv(Path(history)))
    validate_ratings(ratings)
    if any(r.timestamp > now for r in ratings):
        raise ValueError("Frozen history contains future events")
    files = {role: {"path": str(path), "sha256": sha(path)}
             for role, path in (("movies", movies), ("history", history), ("tags", tags), ("preset", preset))}
    plan = {"schema_version": 1, "status": "awaiting future outcomes; no untouched historical test exists",
            "registered_at": now, "history_cutoff": now, "outcome_end": now + days*86400,
            "minimum_history": 5, "primary_metric": "ndcg_at_10", "ks": [10, 20],
            "positive_rating_threshold": 4., "minimum_worthwhile_effect": minimum_effect,
            "baseline": "same structured core without text", "challenger": "frozen enriched TF-IDF preset",
            "comparison_count": 1, "bootstrap_resamples": 10000, "seed": 42,
            "catalog_availability": "full frozen catalog available at registration",
            "cohort": "all frozen-history users with >=5 ratings, including those with no future observations",
            "limits": "observational known-positive metrics, not online causality; provenance depends on supplied data",
            "files": files, "source_sha256": source_hashes()}
    plan["plan_id"] = identity(plan)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")
    return plan


def verify_plan(plan, now=None):
    now = int(time.time()) if now is None else int(now)
    if plan.get("schema_version") != 1 or identity(plan) != plan.get("plan_id"):
        raise ValueError("Confirmation plan was modified or has an unsupported schema")
    if now < plan["outcome_end"]:
        raise ValueError("Confirmation window is still open; do not inspect outcome metrics early")
    for value in plan["files"].values():
        if sha(value["path"]) != value["sha256"]:
            raise ValueError("A frozen dataset or preset changed; confirmation is invalid")
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("Model/evaluation source changed after registration; use the frozen checkout")


def claim_once(plan, state_dir):
    """Claim before outcome parsing; interrupted attempts also remain recorded."""
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    path = state / f"{plan['plan_id']}.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump({"plan_id": plan["plan_id"], "status": "opened", "opened_at": int(time.time())}, f)
    return path
