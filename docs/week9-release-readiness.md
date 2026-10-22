# Penumbra Week 9 release and integration contract

## Artifact and environment

Use operator-controlled **private versioned release artifacts**, release `penumbra-week9-v1`. The frozen bundle is 9,788,194 bytes (~9.34 MiB); it contains 60,255 profile ratings with user IDs/timestamps, enriched metadata, learned user/item factors and ranker/scaler. These are potentially sensitive/licensed data: review consent, retention, redistribution and access controls before any distribution. No upload, publication, Git staging or LFS migration is performed. Its current Git tracking is audited in `reports/week9/git-before.txt` and final status. Historical auxiliary models/reports may remain untracked.

Bundle `src/models/week7_serving.joblib`, schema 1, SHA-256 `3ed639f5a58fa1029523ed925cb3a31050e502f40f4af989e550db632e1b3cb3`. Selected configuration SHA-256 `7b6bbfaad010c70d51cff5b33de10e5ca87311c17169f3307c40943837233ead`. Exact feature order: collaborative, personal, content, quality, popularity. Selected unbalanced TF-IDF pairwise logistic and scaler remain frozen. Manifest: `docs/week9-artifact-manifest.json`. Retain original ranker provenance/manifests and reference dataset checksums alongside the private release. No separate ratings CSV is needed for inference if the backend provides current histories; catalog/population/MF are embedded. Validation/CLI examples also require the local data files.

CPython 3.13.14 and `requirements-serving.txt` pin the **tested** dependency versions without modifying general requirements or the existing environment. This strict release loader rejects different versions; other Python versions/platforms are not certified. Portable paths normalize the bundle's historical Windows separators under an explicit root, without changing process cwd. Current tests run on Windows; Unix instructions are supplied but Unix binary compatibility is not measured. These are exact version constraints, not a platform wheel/hash lock: archive vetted wheels per target platform before deployment.

Fresh environment from repository root (do not replace the developer environment):

```sh
python3.13 -m venv .venv-serving
# Unix
.venv-serving/bin/python -m pip install -r requirements-serving.txt
# Windows PowerShell equivalent
.venv-serving/Scripts/python.exe -m pip install -r requirements-serving.txt
```

Week 9 created a separate isolated environment without modifying the existing virtual environment. Offline installation initially failed because no wheels were cached; package-index installation with all ten exact pins then succeeded. Clean `pip check` passed. Real individual and all-strategy group smoke checks from another working directory produced byte-identical full-score digests and output JSON in clean/current environments. This verifies the tested Windows platform, not other binary builds or operating systems. Archive vetted platform wheels and authenticated dependency provenance before deployment. Tests additionally use development dependencies, including pytest 9.1.1; serving pins do not lock teammate experiments.

## Trusted loading and public calls

Obtain the manifest through an authenticated, operator-controlled channel and mount resources read-only. A SHA verifies integrity against that reference; it is **not authenticity**. Never accept user-provided joblib files or manifests. Verification checks schema/features, exact interpreter/dependencies, config/artifact hashes before joblib load and checks the embedded preset agrees. There is no silent fallback to an older model. Do not regenerate a missing frozen artifact: preparation scripts fit MF and cannot prove exact original reproduction.

```python
from pathlib import Path
from src.hybrid.release import load_release, RatingHistory, ServingSession, ServingMetrics

root = Path('/operator/configured/release/root')  # use your release directory
bundle = load_release(root, trusted_manifest=root/'docs/week9-artifact-manifest.json')
metrics = ServingMetrics()  # omit metrics argument to disable collection
history = RatingHistory(current_ratings)  # DataFrame userId/movieId/rating/timestamp
service = ServingSession(bundle, history, metrics=metrics)
individual = service.individual(23, limit=10)
group = service.group([23, 24], limit=10)  # mean default
fair = service.group([23, 24, 25], strategy='fairness_aware')
revision = service.update(expected_revision=0, user_id=23, movie_id=123,
                          operation='add', rating=4.0, timestamp=1800000000)
health = metrics.snapshot()
```

The illustrative add requires an unseen/nonexisting rating; edit/remove require an existing unique user/movie event. IDs must be positive integers; ratings finite 0–5; timestamps nonnegative integers. See rating-updates.md for exact edit/remove rules. Caller handles ValueError (bad request, revision conflict, schema/config/hash mismatch), FileNotFoundError/OSError (missing/unreadable resource), package/import errors (incompatible installation) and unexpected inference errors. Never expose file paths/internal exception details to end users. Zero/empty results are supported, not artifact fallback. Frozen metadata or watched intersection can yield fewer than K results.

Individual immutable records contain movie_id/title/genres/score. Raw Hybrid decision margins are uncalibrated preference scores, not stars/probabilities; displayed diversity order need not descend by raw score. Group immutable records add group_score/member_scores/min_satisfaction/disagreement: member scores are tied percentile ranks over the current full shared candidate pool (neutral .5 for a singleton/constant); disagreement is max minus min, not a confidence interval. Mean/least_misery/fairness_aware aggregate those ranks; diversity preserves actual aggregate scores. Results are deterministic for the same immutable resources and history snapshot. No real group-choice accuracy is guaranteed.

## Lifecycle and consistency

Startup authenticates/verifies/releases resources once per worker, then initializes histories and optional bounded counters. Keep bundle/catalog/config immutable for the worker lifetime. Requests atomically copy a RatingHistory snapshot under its lock and release the lock before inference. Updates return a new revision; optimistic expected_revision prevents lost updates inside that one process. A request already running can return the previous revision's output; next request sees the change. Frame values/history keys invalidate cached scores; no global flush is needed per rating edit. Cache is bounded to 16 entries/200,000 candidates, process-local, not persisted. Concurrent misses can duplicate computation. Do not mutate raw caller frames concurrently when using lower-level APIs directly.

Backend owns authentication, persistence, transaction/idempotency/event semantics, cross-worker update/version propagation, request-to-revision tracing and user erasure. The in-memory facade is not a database. Deploy a bounded authoritative history store; do not instantiate an unbounded per-user service registry. Persist updates before publishing new versions to workers under an agreed backend transaction contract. Snapshot storage itself scales with the supplied ratings frame. Shutdown drains active requests then discards history/cache; restart cold workers from verified persisted state. Global cache clearing is process-wide; coordinate worker restart for model refresh.

Personal/Content/watched eligibility update immediately; MF user factors/biases and population Quality/Popularity remain frozen. New users keep MF item/global fallback. Collaborative owner must supply/evaluate a supported fold-in or versioned refresh/deletion contract; Hybrid must not invent one. Content owner owns cold inference/metadata fixes. Hybrid owns adapters, candidate policy, score/normalization/diversity contracts, cache, verification and tests. Backend/frontend own API/auth/storage/error translation/UI/rollout.

## Observability and failure policy

Optional ServingMetrics stores only bounded aggregate per-kind counts, total latency, errors and group member totals, plus cache snapshot. No IDs/history payloads are retained. Existing optional lower-level timing dictionaries expose candidate counts, fallback counts, fingerprinting, Content, aggregation and diversity phases; collect per request or export aggregates, never accumulate unbounded dictionaries. Backend should export request latency histograms, invalid-input and artifact-load error counters, update operation counts, candidates/fallback counts, cache hits/misses/evictions, group-size distribution and serving errors. Counters are per process and reset at restart; deltas of shared cache counters cannot attribute a concurrent request precisely. No monitoring service was introduced.

Provisional starting alerts: any checksum/schema/dependency mismatch blocks readiness; any nonfinite output is an error; sustained >1% unexpected serving errors or >5% unexpected fallback predictions triggers investigation; >20% eviction/miss increases versus workload baseline triggers capacity review. Warm p95 above twice the measured staging baseline warrants investigation. These are proposed starting points, not measured SLAs. Cold requests and large groups need separate budgets. Apply bounded worker queues, admission limits/timeouts and per-instance memory budgets after staging measurement. Do not hide timeouts by switching recommendation models.

## Rollout and rollback procedures (not executed)

1. Local: verify trusted release hashes, full tests, exact reference outputs, data/privacy review and target-platform environment.
2. Staging: immutable artifact/config/source versions, backend persistence/auth/version propagation, warm selected common histories, readiness/smoke checks and bounded representative load; verify cold/memory budgets.
3. Limited rollout behind a reversible backend feature flag; retain prior verified source/environment/artifact together.
4. Monitor cohort latency/errors/cache/fallbacks/update consistency and review group feedback without retuning frozen test outcomes.
5. Expand only after operational owners accept evidence and rollback rehearsal.

Rollback routes traffic to the prior **verified implementation + compatible environment + verified artifact/config + history schema**, not an arbitrary older joblib. Drain workers, discard their process-local caches, initialize prior workers from authoritative persisted ratings and verify smoke/golden outputs before restoring traffic. Do not roll back durable rating writes or assume newer cache keys work in an older implementation. Week 9 baseline source copies and HEAD identify this task's starting implementation; they are evidence, not a complete signed production release. Backend must retain/deploy a complete prior release package. No deployments, commits or pushes were performed here.
