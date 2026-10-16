# Week 8 serving latency

Frozen Week 7 serving only. No retraining, final-test tuning, group recommendations, or Week 9 work.

## Measured results

Three sequential samples per case and mode; medians below. Times cover `recommend_for_user` including its unchanged diversity reranker. CSV loading, bundle loading, imports and profiler overhead are excluded. Baseline and optimized calls were interleaved in one process; cold means empty application cache, not empty OS caches. These are local measurements, not production percentiles.

| Serving case | History rows | Before | After, empty cache | After, warm cache | Cold reduction | Warm speedup |
|---|---:|---:|---:|---:|---:|---:|
| existing (user 23) | 121 | 7.651s | 4.673s | 0.150s | 38.9% | 51.1x |
| sparse_one_rating (user 23) | 1 | 8.118s | 5.077s | 0.308s | 37.5% | 26.3x |
| cold_start (user 611) | 0 | 4.329s | 2.689s | 0.305s | 37.9% | 14.2x |

Separate bundle load: 0.099s; CSV load: 0.140s.

## Bottlenecks and changes

- Repeated unchanged requests previously rebuilt Content state, predicted all candidates, rescored genres, and reran Collaborative prediction and the frozen ranker. A bounded, process-local LRU now reuses the complete exact sorted `(movie_id, score)` sequence. The requested candidate-pool prefix is unchanged and diversity still runs on every outer request.
- Genre scoring repeatedly extracted the same pandas columns for each candidate and genre. Those identical numeric arrays are now extracted once per request; formulae, reductions, row order, and weights stay unchanged.
- Serving previously constructed metadata and recommendation objects for every candidate before slicing. It now constructs them only for the requested sorted prefix. It still scores the entire candidate catalog.
- Temporary print timers were replaced with opt-in `perf_counter` phase measurements through `timings={}` on `recommend_week7` and `build_feature_rows`. `use_cache=False` provides an uncached diagnostic path.

Median cold-request phase times (overlapping feature totals are omitted):

| Case | Cache key/lookup | Content build | Genre scoring | Content prediction | Collaborative | Metadata |
|---|---:|---:|---:|---:|---:|---:|
| existing | 0.149s | 0.453s | 0.567s | 3.288s | 0.124s | 0.003s |
| sparse_one_rating | 0.261s | 0.799s | 0.844s | 2.495s | 0.374s | 0.017s |
| cold_start | 0.235s | 0.733s | 0.001s | 1.067s | 0.288s | 0.013s |

Detailed cumulative profiles for all three cases are saved beside the raw JSON report. Profiling runs were separate from latency samples. Content prediction/conversion remain the main first-request costs; their teammate-owned implementations were not edited.

## Cache correctness and lifecycle

- Keys fingerprint the target history (values, timestamps, order, index, schema), allowed candidate IDs, frozen catalog and population snapshot, scaler/ranker/MF state, and selected preset file contents/path. In-place rating edits, additions, deletions, timestamp edits and model/catalog/population/preset changes cannot reuse the prior key. Only the target user history is relevant: changing another user in the caller ratings frame does not change the frozen population supplied by the bundle.
- Hashing is recomputed on every request rather than trusting object identity or modification times. It has a measurable warm-request cost. Unsupported custom inputs bypass caching. Failed inference is never stored.
- At most 16 entries and 200,000 candidate pairs are retained, with LRU eviction; overlarge entries bypass retention. Scores are immutable tuples and returned recommendation lists are fresh. `SERVING_CACHE.clear()` releases entries; `SERVING_CACHE.info()` reports hits, misses, eviction counts and bounds.
- Bookkeeping is protected by a lock. Concurrent misses may compute the same result twice. Callers must not mutate shared input tables or model objects while a request is executing. Caches are per process, not distributed or persisted.

## Verification

- Exact full-ranking ID/title/genre/float-score equality with the original implementation for existing (121 ratings), one-rating and zero-rating users; exact final top-10 diversity equality in every measured sample.
- Real-history rating edits/new cold-user history and newly watched movies forced cache misses, matched the original implementation, then produced warm hits. A newly watched recommendation disappeared immediately.
- Regression tests cover rating/timestamp/add/delete/reorder mutations, model/preset/catalog/population/candidate changes, changing limits, ties, missing metadata, duplicate catalog rows, empty results, opt-out, bounded eviction, mutable returned lists, failures, concurrent bookkeeping and golden pre-Week-8 genre outputs.
- Frozen artifacts, teammate Content/Collaborative source files and pre-existing tracked changes were checked against pre-change hashes/diffs. See `preservation.json`.

## Files and reproduction

- Runtime: `src/hybrid/week7.py`, `src/hybrid/serving_cache.py`, `src/hybrid/genre_recommender.py`.
- Tests: `tests/hybrid/test_week8.py`, `tests/hybrid/fixtures/week8_genre_golden.json`.
- Benchmark: `src/scripts/benchmark_week8.py`; evidence: `reports/week8/` (raw samples, profiles, original source snapshots, preservation hashes and runtime diff).
- Run `.venv\Scripts\python.exe -m src.scripts.benchmark_week8 --runs 3 --output reports/week8/recheck.json` from the repository root. Existing reports are never overwritten. The archived baseline source snapshots are diagnostic references only.

## Remaining limits

- Warm gains require a long-lived process. A new CLI process starts empty; first requests and changed histories still build/predict Content and score all candidates.
- Fingerprinting scans bundle tables and model state each request. Large deployments may need an explicit immutable versioned serving API to avoid that cost safely.
- Cache entries contain derived user-specific scores in process memory until eviction/clear/process exit; no cache data is written to disk. There is no TTL or cross-worker sharing.
- Three repeats on one local machine do not establish p95/p99 latency, multiworker throughput or load behavior. OS caches, CPU load and Python startup were not controlled.
- Full repository suite: 399 passed, 83 subtests passed, 36 dependency deprecation warnings in 28.01s. This includes 24 Week 8 regression cases. Existing unit tests may fit synthetic models in temporary directories; serving artifacts were not retrained or changed.
- No commits or pushes were made.


## Week 8 final engineering freeze

See [Week 8 finalization](week8-finalization.md), [rating update contract](rating-updates.md), and the conflict/performance evidence under `reports/week8/freeze/`. Earlier measurements above are historical local observations, not guaranteed current results. Current phase instrumentation includes candidate/fallback counts and separate fingerprint/lookup/storage timing. Frozen presets are rechecked on requests; Group V1 supports optional `timings={}`. This engineering freeze retains the documented production and group-choice limitations.
