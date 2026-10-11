"""Catalog-level metrics, cutoff filtering, and user-level uncertainty."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from src.content.schemas import UserRating
from src.evaluate_content import temporal_split


@dataclass(frozen=True)
class CatalogCase:
    user_id: int
    cutoff: int
    history: tuple[UserRating, ...]
    outcomes: tuple[UserRating, ...]


def validate_ratings(ratings):
    seen = set()
    for r in ratings:
        if r.timestamp is None:
            raise ValueError("All evaluation ratings require timestamps")
        key = (r.user_id, r.movie_id)
        if key in seen:
            raise ValueError("Duplicate user/movie ratings are not supported")
        seen.add(key)


def first_observed(ratings):
    """Conservative availability proxy; only the earliest interaction time is used."""
    result = {}
    for r in ratings:
        if r.timestamp is None:
            raise ValueError("Catalog availability requires timestamps")
        result[r.movie_id] = min(result.get(r.movie_id, r.timestamp), r.timestamp)
    return result


def development_cases(ratings):
    validate_ratings(ratings)
    grouped = defaultdict(list)
    for r in ratings:
        grouped[r.user_id].append(r)
    cases, skipped = [], 0
    for user_id, rows in sorted(grouped.items()):
        history, validation, _ = temporal_split(rows)
        if len(history) < 5 or not validation:
            skipped += 1
            continue
        cases.append(CatalogCase(user_id, max(r.timestamp for r in history), history, validation))
    return cases, skipped


def future_cases(history, outcomes, cutoff, end, eligible_users):
    validate_ratings(history)
    validate_ratings(outcomes)
    if any(r.timestamp > cutoff for r in history):
        raise ValueError("Training history exceeds frozen cutoff")
    if any(not cutoff < r.timestamp <= end for r in outcomes):
        raise ValueError("Outcomes must fall strictly after cutoff and within the frozen window")
    known = {(r.user_id, r.movie_id) for r in history}
    if any((r.user_id, r.movie_id) in known for r in outcomes):
        raise ValueError("Outcome repeats a history user/movie pair")
    histories, targets = defaultdict(list), defaultdict(list)
    for r in history:
        histories[r.user_id].append(r)
    for r in outcomes:
        targets[r.user_id].append(r)
    cases = [CatalogCase(u, cutoff, tuple(sorted(histories[u], key=lambda r: (r.timestamp, r.movie_id))),
                         tuple(targets[u])) for u in eligible_users]
    unknown_users = len(set(targets) - set(eligible_users))
    return cases, unknown_users


def candidate_ids(case, availability, movie_ids):
    seen = {r.movie_id for r in case.history}
    return tuple(sorted(m for m in movie_ids if m not in seen
                        and availability.get(m, float("inf")) <= case.cutoff))


def full_catalog_metrics(ids, scores, outcomes, ks=(10, 20), threshold=4.):
    """Observed-positive recall/NDCG; unknown items are unjudged, not dislikes.

    All models use score descending then movie ID ascending to break ties.
    A user with no eligible positives has undefined recall/NDCG and is excluded
    from those macros, with the excluded count reported explicitly.
    """
    ids, scores = np.asarray(ids, dtype=np.int64), np.asarray(scores, dtype=float)
    if len(ids) != len(scores) or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("A finite score is required for every candidate")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate candidates")
    if any(k <= 0 for k in ks) or not 0 <= threshold <= 5:
        raise ValueError("Invalid metric configuration")
    known = {r.movie_id: r.rating for r in outcomes}
    if len(known) != len(outcomes):
        raise ValueError("Duplicate outcomes")
    all_positives = {m for m, rating in known.items() if rating >= threshold}
    available = set(ids.tolist())
    positives = all_positives & available
    ranked = ids[np.lexsort((ids, -scores))]
    metrics = {"candidates": len(ids), "outcomes": len(outcomes),
               "eligible_outcomes": len(set(known) & available),
               "positives_total": len(all_positives), "positives_eligible": len(positives)}
    for k in ks:
        top = ranked[:k].tolist()
        hits = sum(m in positives for m in top)
        gains = np.array([float(m in positives) for m in top])
        ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(positives))))
        metrics[f"recall_at_{k}"] = hits / len(positives) if positives else None
        metrics[f"ndcg_at_{k}"] = float(np.sum(gains / np.log2(np.arange(len(top)) + 2)) / ideal) if ideal else None
        metrics[f"recall_all_observed_positives_at_{k}"] = hits / len(all_positives) if all_positives else None
        metrics[f"judged_fraction_at_{k}"] = sum(m in known for m in top) / len(top) if top else None
    return metrics, ranked[:max(ks)].tolist()


def aggregate(rows, ks=(10, 20)):
    metrics = {}
    for k in ks:
        for key in (f"recall_at_{k}", f"ndcg_at_{k}", f"recall_all_observed_positives_at_{k}",
                    f"judged_fraction_at_{k}"):
            values = [r[key] for r in rows if r[key] is not None]
            metrics[key] = float(np.mean(values)) if values else None
    positive_total = sum(r["positives_total"] for r in rows)
    eligible = sum(r["positives_eligible"] for r in rows)
    return {**metrics, "users": len(rows),
            "users_with_eligible_positives": sum(r["positives_eligible"] > 0 for r in rows),
            "users_without_eligible_positives": sum(r["positives_eligible"] == 0 for r in rows),
            "users_without_outcomes": sum(r["outcomes"] == 0 for r in rows),
            "positive_availability_fraction": eligible / positive_total if positive_total else None,
            "positives_total": positive_total, "positives_eligible": eligible,
            "candidate_count_median": float(np.median([r["candidates"] for r in rows])) if rows else None}


def paired_inference(baseline, challenger, metric, repeats=10000, seed=42, minimum_effect=.01, comparisons=1):
    if baseline.keys() != challenger.keys():
        raise ValueError("Models must be evaluated on exactly the same users")
    if repeats < 100 or minimum_effect <= 0 or comparisons < 1:
        raise ValueError("Invalid inference settings")
    pairs = []
    for user in sorted(baseline):
        a, b = baseline[user][metric], challenger[user][metric]
        if (a is None) != (b is None):
            raise ValueError("Metric eligibility must not depend on model predictions")
        if a is not None:
            pairs.append(b - a)
    if len(pairs) < 2:
        return {"users": len(pairs), "status": "insufficient eligible users"}
    deltas = np.asarray(pairs)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, repeats, 100):
        sample = rng.choice(deltas, (min(100, repeats-start), len(deltas)), replace=True)
        means.extend(sample.mean(axis=1))
    alpha = .05 / comparisons
    sd = float(deltas.std(ddof=1))
    z = NormalDist().inv_cdf(1-alpha/2) + NormalDist().inv_cdf(.8)
    return {"users": len(pairs), "mean_delta": float(deltas.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "familywise_ci95": np.quantile(means, [alpha/2, 1-alpha/2]).tolist(),
            "resamples": repeats, "seed": seed, "comparison_count": comparisons,
            "paired_user_sd": sd, "planning_mde_80pct_power": z*sd/np.sqrt(len(pairs)),
            "planning_minimum_effect": minimum_effect,
            "planning_users_for_effect": int(np.ceil((z*sd/minimum_effect)**2)) if sd else None,
            "planning_caveat": "normal approximation, independent users, same future variance; zero observed variance cannot establish power"}
