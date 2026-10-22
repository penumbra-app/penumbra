# Penumbra Week 9 finalization

**Engineering complete for the scoped Hybrid/Group Python serving release; conditional backend-integration readiness. Not deployed or production-certified.** Deployment gates are explicit below, not missing claims disguised as completion.

## Objectives and completed work

Audited Git/evaluation/provenance/ownership, preserved all historical evidence and artifacts, added trusted release verification and portable resource roots, pinned the serving dependency closure, verified an isolated installation, added atomic rating snapshots with optimistic revisions and optional bounded aggregate metrics, added regression tests, measured cold/warm/update/eviction/concurrent/RSS behavior, compared complete frozen outputs, and documented release/rollback/handoff.

## Final architecture and frozen configuration

Startup verifies operator-trusted manifest/resources/dependencies and loads the unchanged Week 7 bundle. Individual: atomic rating snapshot → shared eligible full-catalog five-feature score/cache pipeline → first 5*K candidates → unchanged diversity. Group: shared unseen catalog → same complete per-member scores → average-tied percentile ranks → mean/default, least-misery or provisional .7 mean+.3 minimum → top 5*K → group diversity. Feature order collaborative/personal/content/quality/popularity; selected **unbalanced TF-IDF logistic**, original scaler, MF and population remain frozen. No training, final-test re-evaluation, algorithm promotion or teammate edits.

## Evaluation and reproducibility

See [evaluation audit](week9-evaluation.md): historical test metrics are preserved, compared baselines are qualified, per-user chronology and MF/Content/scaler boundaries are audited, first-60% profile reconstruction matches exactly, and three prespecified validation-only candidate cohorts reproduce exact scores. This is frozen serving reproducibility, not exact historical training reconstruction. Global/as-of leakage limitations, selection reuse and exposure-biased candidate metrics remain explicit.

## Dependency and artifact contract

See [release readiness](week9-release-readiness.md), requirements-serving.txt and week9-artifact-manifest.json. Python 3.13.14 plus all ten serving packages are pinned, including transitive Narwhals 2.24.0. Existing environment/general requirements unchanged. Offline isolated install initially lacked cached wheels; authenticated package-index installation succeeded in a separate environment. An initially floating Narwhals resolver choice was corrected to the existing version before verification. Both environments loaded the verified bundle from another working directory and produced byte-identical serialized full-score digests and individual/group outputs for all three strategies. Clean pip check passed. Windows verified; Unix instructions provided, platform portability not certified. Pins are not wheel hashes/signed supply-chain provenance. Original artifact remains byte-identical; private versioned distribution is selected, but nothing was published. Embedded profile ratings/factors/metadata require privacy/licensing review.

## Individual/group integration and rating lifecycle

ServingSession delegates unchanged individual/group algorithms. RatingHistory privately owns snapshots and serializes add/edit/remove with optional expected_revision. An in-flight request retains its old snapshot; later requests observe the update. Process-local consistency only. Backend owns durable/authenticated updates, cross-worker revisions, event IDs/conflicts/erasure and bounded authoritative storage. MF factors/population remain frozen; no unsupported fold-in. Raw Hybrid scores are not calibrated stars; group percentiles depend on shared candidates and are not acceptance probabilities.

## Performance and reliability

See [fresh performance evidence](week9-performance.md) and reports/week9. Expensive cold Content inference persists; warm caches are substantially faster. Week 9 does not claim another algorithmic speedup. Matched before/after checks preserve scores; three-member/five-member/conflict groups, updates, eviction, 200-request individual load, bounded group concurrency and OS process memory are measured. Initial existing cold samples overlapped tests and are explicitly replaced by serial measurements. Startup and tiny samples are not SLAs/p99 estimates.

Optional bounded aggregate counters plus existing phase/candidate/fallback diagnostics cover library observability; deployment exports/alerts remain backend work. Missing resources, checksum/schema/config/dependency errors fail closed. Controlled synthetic tests reject corrupt/untrusted content before deserialization; tests do not load arbitrary uploaded artifacts. Existing legacy corruption tests use locally constructed fixtures only. Concurrent cache bookkeeping and mid-request updates are covered. No unbounded serving cache, new infrastructure or raw history logging.

## Verification

Final full suite: **492 passed, 36 warnings, 83 subtests passed; 0 failures, 0 skips**. Warnings are existing NumPy 2.5 array-shape deprecations in joblib, not silently suppressed. New targeted release suite: 12 passed. Final relevant Hybrid suite: 166 passed, 14 existing warnings, 0 failures/skips in 14.72s. Final full suite completed in 28.80s. Exact commands/logs are in reports/week9. Existing Week 7/8/group/cache/rating regressions are included. Real individual CLI and full individual/group smoke checks passed. Three individuals' complete score sequences and diversity match archived references; nine groups × three strategies compare all returned records/scores exactly. Clean/current environments produce identical smoke JSON. Git whitespace and preservation checks pass; all prior files except explicitly intended README/week7 loader changes remain byte-identical.

## Known limitations and deferred production work

1. Cold Content prediction costs, duplicated concurrent misses, 16-entry cache thrash and value-fingerprint overhead require deployment capacity/admission budgets; no distributed cache, multiworker stress study or production SLA.
2. Rating persistence/auth/idempotency/version propagation and deletion obligations are backend responsibilities. Snapshot copies consume memory proportional to supplied history; do not create an unbounded service registry.
3. Collaborative owner must define legitimate MF fold-in/refresh/deletion behavior and independent validation. Frozen MF does not react to live edits.
4. Content owner must review as-of metadata/provenance and any algorithm performance improvements. No owner algorithm was modified.
5. Real group-choice labels, full-catalog quality/acceptance calibration and independent fairness validation are absent; mean and provisional 70/30 stay frozen.
6. Authenticated private artifact distribution, profile/metadata privacy/licensing review, signed manifests/vetted platform wheels, Unix verification and staging rollback rehearsal remain deployment gates.
7. Original training inputs/dependency provenance are incomplete; hashes recorded now cannot retroactively prove exact original training reproduction or strict global temporal isolation.

## Ownership, release and freeze

Hybrid: scoring/adapters/candidate policy/group math/diversity/cache/release checks/tests. Content: metadata/profile algorithms and approved performance fixes. Collaborative: MF lifecycle. Backend/frontend: auth/storage/version propagation/API/error translation/UI/rollout/monitoring. Release stages and rollback procedure are documented, not executed. Restore prior compatible source/environment/artifact/config together, drain workers and discard incompatible caches; never undo durable rating events to roll back a model.

All scoped completion criteria have evidence or documented environment/provenance limitations. Backend integration can proceed behind a reversible feature flag in local/staging, subject to the listed gates before deployment. No new recommendation-model phase started.

## Git and preservation

Starting branch hybrid-group-reranking was aligned with its locally recorded origin ref at HEAD 37ada06; no six unpushed commits remained. No network fetch was used to assert live remote state. Prior untracked inputs/reports/auxiliary models were preserved. The serving bundle is now tracked in the starting repository, unlike the earlier Week 8 audit. No staging/reset/clean/merge/rebase/commit/push occurred. reports/week9/preservation-final.json verifies unchanged HEAD, frozen models/config/historical reports/owner files and every unrelated pre-existing file.

## Exact changed files

Modified: `README.md`, `src/hybrid/week7.py`.

Created: `src/hybrid/release.py`, `requirements-serving.txt`, `src/scripts/benchmark_week9.py`, `src/scripts/verify_week9.py`, `src/scripts/smoke_week9.py`, `tests/hybrid/test_week9_release.py`, `docs/week9-artifact-manifest.json`, `docs/week9-evaluation.md`, `docs/week9-performance.md`, `docs/week9-release-readiness.md`, `docs/week9-finalization.md`.

New evidence: `reports/week9/` (baseline copies, checksums, benchmark/verification/smoke JSON, test/CLI logs, commands, Git diff/status and preservation/freeze manifests). No historical report files were changed.
