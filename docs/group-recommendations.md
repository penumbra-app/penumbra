# Penumbra Group Recommendations V1

## Status and scope

Group V1 is implemented in Hybrid ownership. It aggregates the existing frozen Week 7 five-signal model without training a group model. Mean is the default. This completes the scoped Group V1 implementation, not every separate Week 8 roadmap item. No Week 9 work is included.

## Public API

```python
from src.hybrid.group_recommender import recommend_group

recommendations = recommend_group(
    [23, 24], ratings, movies, bundle=bundle, limit=10, strategy="mean",
)
```

Groups require at least two distinct positive integer IDs. Member order does not affect results. Duplicate, boolean, string, noninteger or nonpositive IDs are rejected. Unknown positive IDs use existing cold-start scoring. `limit` must be a nonnegative integer; zero returns an empty list. Invalid strategies, missing/duplicate table columns, invalid table IDs, nonfinite/out-of-range ratings and invalid timestamps raise `ValueError`. Ratings must include `userId`, `movieId`, `rating` (0–5), and nonnegative whole-second `timestamp`. Nonempty ID columns must have integer dtype. Empty tables with the required columns are accepted. Bundle catalog metadata is authoritative; the supplied movie table needs `movieId`.

The immutable `GroupRecommendation` contains `movie_id`, `title`, `genres`, `group_score`, `member_scores`, `min_satisfaction`, and `disagreement`. `member_scores` is an immutable tuple of `(user_id, normalized_preference)` pairs in ascending user-ID order; use `dict(row.member_scores)` for lookup. The `.score` property exposes the same aggregation score to the existing diversity reranker. It never reports a diversity-adjusted score as the group score.

The field name `min_satisfaction` denotes the minimum normalized member preference required by the output contract. It is not observed satisfaction, a star rating, or a probability.

## Candidate policy and individual scoring

1. Intersect the supplied movie IDs with the frozen serving catalog.
2. Deduplicate IDs and exclude every movie watched by any group member.
3. Sort IDs and score every member against that exact shared pool. If it is empty, return `[]`.
4. Obtain all raw Week 7 decision scores using `score_week7_candidates`, with no per-user top-K truncation or per-user diversity.

The original Collaborative, Personal, Content, Quality and Popularity signals, scaler, feature order and ranker remain unchanged. The score-only helper was extracted from `recommend_week7`; individual recommendation callers keep their API and exact ordering/scores. The helper avoids constructing full-catalog display objects for each group member. No Content or Collaborative algorithm was edited.

## Percentile normalization

For each member independently, rank raw scores across the entire shared pool in ascending order, using average ranks for ties. For a nonconstant pool of size `n > 1`:

```text
normalized_preference = (average_rank - 1) / (n - 1)
```

Values lie in `[0, 1]`, strictly increasing when raw scores differ; tied raw scores get identical preferences. Tied extremes need not reach exactly 0 or 1. A single candidate and a constant-score pool both receive neutral `0.5` for every candidate. Empty input produces an empty preference array. Nonfinite raw scores are rejected. Percentiles are relative preferences within this particular pool, never calibrated satisfaction or predicted stars.

## Aggregation and fairness tradeoffs

| Strategy argument | Group score | Tradeoff |
|---|---|---|
| `mean` (default) | Mean member preference | Rewards overall preference but can tolerate one low member score. |
| `least_misery` | Minimum member preference | Protects the least-preferred member score; may favor cautious compromises. |
| `fairness_aware` | `0.7 * mean + 0.3 * minimum` | Provisional compromise between those goals; 70/30 is not a validated optimum. |

`disagreement` is the range `max(member_preferences) - min(member_preferences)` for a movie. It is reported separately, not used as an additional hidden penalty. Equal member weights are used; history length and uncertainty are not additional weights. Cold users' relative ordering may be driven largely by population signals, so normalization does not create evidence of personal preferences.

## Diversity and deterministic ordering

All candidates are aggregated before any truncation. The initial group ranking uses descending aggregation score and ascending movie ID on ties. The diversity pool is the first `max(limit * 5, limit)` movies of that group ranking, matching the existing individual candidate-pool convention.

The existing greedy genre-diversity reranker then applies its unchanged default weight `0.15`. An optional movie-ID tie-break callback is supplied only by group callers, resolving equal adjusted scores by ascending movie ID. Individual callers omit it and retain their existing behavior. The final list is in greedy diversity order and may not have descending group scores. Output group scores remain the actual aggregation values.

## Cache and performance

Group V1 uses the existing `RankingCache` and its unchanged content-based invalidation keys. It adds no group cache or fitted-model cache. Keys include the target rating history, allowed pool, frozen catalog/population, model state and preset contents. Rating edits invalidate the changed member's scores; adding a watched movie changes the shared pool and can invalidate every member's entry. Other unaffected member entries can be reused for rating-only edits. Choosing a different strategy or limit reuses the same raw member scores.

The cache remains bounded by 16 entries and 200,000 candidate pairs, process-local and protected for concurrent bookkeeping. Large groups or varied candidate pools can cause eviction and recomputation. Concurrent misses may duplicate computation; callers must not mutate shared input/model objects while requests execute. Warm gains require a long-lived process. Fingerprinting, normalization and aggregation still run on warm requests.

Local benchmark samples and strategy proxy results are saved in `reports/group_v1/evaluation.json`. Cold means an empty application cache, not a machine with empty filesystem caches. Loading CSVs, models and Python imports is outside the reported call latency. These measurements do not establish production throughput or tail latency. The report includes separate existing, three-member, sparse-history, cold-start and repeated-request cases, plus an individual warm-latency check.

Local medians over three repeats (mean strategy; all member scoring included):

| Case | Empty cache | Warm repeated |
|---|---:|---:|
| two_existing | 19.225s | 0.704s |
| three_existing | 27.808s | 1.059s |
| with_sparse | 14.919s | 0.655s |
| with_cold_start | 12.177s | 0.629s |

Individual warm-call median after group use: 0.288s. Complete individual rankings and diversified outputs remained exactly equal to the archived implementation.

## Evaluation methodology

Run from the repository root:

```text
.venv\Scripts\python.exe -m src.scripts.evaluate_group_v1 --runs 3 --output reports/group_v1/recheck.json
```

The evaluator uses only the frozen bundle's reference/profile histories plus one deterministic synthetic sparse profile. It does not read final-test labels, retrain, tune weights, or select a winning strategy. This is a reproducible serving/proxy audit, not held-out relevance evaluation. It records empty-cache and repeated warm mean-strategy calls, then compares all three strategies with warm member-score caches. Scoring is identical across strategies; cold scoring is measured once per repetition rather than repeating the same expensive work for each aggregation formula.

Metrics on the final diversified slate:

- Mean normalized member preference across members and movies.
- Mean of each movie's minimum member preference; also the worst member's slate-average preference.
- Mean per-movie disagreement (member preference range).
- Each member's mean preference and fraction of slate movies at or above their within-pool 75th percentile; fraction of members represented by at least one such movie. The threshold is a descriptive convention, not a validated acceptance threshold.
- Mean pairwise Jaccard genre distance and unique genre count.
- Runtime samples and medians for cold and warm calls.

No real group-choice/acceptance data is available in this evaluation. These model-derived proxies cannot prove real-world group satisfaction. Higher minimum scores for least misery, or higher means for mean aggregation, are partly consequences of their objectives and are not evidence that either is universally better. Post-aggregation diversity may trade a small amount of preference score for genre variety. There is no strategy winner or automatic model promotion.

Real-artifact checks compare complete individual score rankings and final diversity slates against the archived pre-group serving source for existing, sparse and cold users. Rating-only changes are checked for one miss plus one unchanged-member hit; new watched movies must disappear immediately. Regression tests cover synthetic math, the real Content serving adapter, caching, invalid inputs, ties, top-K pitfalls and individual preservation.

## Known limitations and remaining work

- Percentiles discard raw score magnitudes and calibration; near ties can be expanded into meaningful rank differences. Results depend on the shared candidate set.
- The 70/30 heuristic, diversity weight on percentile scores and representation threshold are provisional. No group-choice evidence currently validates them.
- Cold/sparse histories contain little personal evidence; population preferences can dominate.
- Group scores are simple aggregations, not learned group dynamics or negotiated choices.
- Candidate churn, groups exceeding cache capacity, and first requests still require full per-member inference. There is no persisted/shared group cache, production load benchmark or group service/UI endpoint.
- Optional runtime/streaming filters, voting, group-choice data collection, acceptance evaluation, confidence-aware weighting and richer group explanations remain future work. Broader Week 8 roadmap completion is not asserted.

## Relevant files

- `src/hybrid/group_recommender.py`: validation, shared pool, percentiles, strategies and immutable output.
- `src/hybrid/week7.py`: shared full-score helper and unchanged individual recommendation semantics.
- `src/hybrid/serving_cache.py`: reused unchanged cache implementation.
- `src/hybrid/diversity.py`: optional group tie-break callback; original default behavior.
- `tests/hybrid/test_group_recommender.py`: mathematical and serving integration regressions.
- `src/scripts/evaluate_group_v1.py`: reproducible proxy evaluation and local benchmarks.
- `reports/group_v1/`: raw samples, strategy metrics, baseline snapshot and preservation evidence.
- `docs/week8-serving-latency.md`: the earlier single-user optimization report, preserved separately.

## Final verification

Full repository suite: 450 passed, 83 subtests passed, 36 dependency deprecation warnings in 26.36s. Includes 51 Group V1 regression cases. Original individual rankings and diversity results remain identical. No commits or pushes.


## Week 8 final engineering freeze

See [Week 8 finalization](week8-finalization.md), [rating update contract](rating-updates.md), and the conflict/performance evidence under `reports/week8/freeze/`. Earlier measurements above are historical local observations, not guaranteed current results. Current phase instrumentation includes candidate/fallback counts and separate fingerprint/lookup/storage timing. Frozen presets are rechecked on requests; Group V1 supports optional `timings={}`. This engineering freeze retains the documented production and group-choice limitations.
