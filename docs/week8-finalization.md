# Penumbra Week 8 finalization

## Purpose and status

**Week 8 engineering freeze complete for the scoped Python Hybrid/Group pipeline**, subject to the explicit deployment limitations below. This is not a claim of production deployment readiness or empirically validated group satisfaction. No Week 9 implementation has begun.

## Completed work

Preserved serving optimization and Group V1; added correlation-defined conflict evaluation, independent validation observations, explicit rating lifecycle helper, cold/update/eviction/startup/cache memory/concurrency diagnostics, candidate/fallback counters, malformed-history guards, frozen-preset checks and optional pre-deserialization integrity verification. Normal serving has no temporary timing prints.

## Architecture and group algorithm

Individual: request snapshot → eligible frozen catalog → same cached full five-signal scores → original top (5 * limit) pool → unchanged individual diversity → immutable recommendations. Group: shared unseen intersection → each member scores that full pool through the same helper/cache → tied average percentile ranks → mean/minimum/provisional .7 mean+.3 minimum → group top (5 * limit) pool → existing diversity with group-only ID tie-break. No member top-K truncation precedes aggregation. Percentiles are relative preferences, not predicted stars or acceptance probabilities. Mean remains default.

## Performance

| Case | Cold median [min,max] | Warm median [min,max] | Rating change median | Evicted median |
|---|---:|---:|---:|---:|
| existing | 3.941s [3.857, 4.025] | 0.160s [0.142, 0.179] | 4.103s | 3.822s |
| sparse | 2.224s [2.200, 2.248] | 0.146s [0.130, 0.162] | 2.293s | 2.368s |
| cold | 1.068s [0.934, 1.202] | 0.117s [0.103, 0.130] | 2.278s | 1.155s |
| similar_pair | 7.858s [7.746, 7.970] | 0.349s [0.339, 0.360] | 4.248s | 7.094s |
| one_outlier | 12.822s [12.424, 13.220] | 0.515s [0.508, 0.522] | 4.371s | 13.176s |
| conflicting_pair | 8.664s [8.358, 8.969] | 0.339s [0.295, 0.382] | 3.637s | 8.900s |
| larger_five | 20.763s [19.586, 21.940] | 0.716s [0.654, 0.778] | 3.734s | 19.324s |

Two repetitions per mode; variability, startup/import/load distinction, matched before/after warm checks, exact preservation and phase samples are in [serving diagnostics](../reports/week8/freeze/serving-diagnostics.md). Historical Week 8 measurements remain intact in their original reports; workloads/environment differ, so no unsupported cross-report speedup claim is made.

## Evaluation

Nine categories × three strategies × K=5/10/20. Construction uses profile-only co-rated Pearson correlation (>=8 common movies) and reports support. Similar/moderate/conflicting, outlier, sparse/cold, multiple-cold and five-person groups are covered. Frozen 70–80% individual validation outcomes supplement the proxies without selecting groups or strategies. No frozen final-test evaluation was rerun. Fairness changes some slates but can worsen an outlier minimum after diversity; there is no demonstrated winner. Group acceptance accuracy is unknown without real group-choice labels. See [conflict report](../reports/week8/freeze/conflict-evaluation.md) for measured tradeoffs, variability limits and actual per-member examples.

## Rating updates

Personal/Content and watched eligibility react immediately to supplied history; frozen MF factors/biases and frozen Quality/Popularity do not learn from those edits. Unknown users keep MF item/global fallback until a legitimate owner-approved refresh. No MF fold-in API exists. The pure add/edit/remove helper and exact regression tests separate request updates from artifact refresh. See [rating contract](rating-updates.md).

## Testing

Final full suite: **480 passed, 36 warnings, 83 subtests passed in 24.55s**. Final failures: 0; skipped: 0. Includes 30 new finalization regression cases.

Executed full command: `python -m pytest -q -p no:cacheprovider`. Hybrid-stage checks also passed 149 tests before the final extra cases. The complete final run includes Group, cache, Week 7, rating-update and Content/Collaborative integrations. Raw output: [tests-final.txt](../reports/week8/freeze/tests-final.txt).

The suite covers exact individual feature/score/order preservation, group math/scale invariance and conflicts, all update operations/noninterference, duplicate events, empty/single/constant/tie cases, missing/malformed predictions, cache invalidation/model/config/catalog changes, LRU bounds/eviction, concurrent requests, corruption/checksum handling and validation-only split sentinels. CLI smoke evidence is saved separately. Dependency warnings are not test failures. The initial tests.txt retains one resolved test-only Pearson roundoff assertion (-0.9999999999999999 versus -1); tests-final.txt uses appropriate correlation tolerance. Frozen recommendation score/order comparisons remain exact.

## Production-readiness review and limitations

- Stable Python schemas and clear failures are tested; no web/REST/UI/social/voting/streaming system was added. Legacy and opt-in Week 7 entry points remain supported.
- Joblib/pickle can execute code: load only trusted, controlled local artifacts. Optional expected SHA-256 is checked before deserialization, but authenticity requires a separately trusted/signed release manifest and access controls. Do not accept uploaded model files. Missing/corrupt/incompatible artifacts fail; no automatic model fallback hides failures.
- Mount immutable bundle/config/catalog snapshots for active workers. Preset checks guard stale loaded configurations; callers must not concurrently mutate model arrays or shared frames. Checksums are not a lock or authorization system.
- Cache bounds, concurrency and Python allocation diagnostics pass under that contract. Miss duplication, LRU thrashing for large groups, fingerprint overhead and expensive cold Content inference remain. No distributed cache, process-RSS ceiling or production tail/load SLA has been established.
- Request history persistence, authorization, idempotency/event IDs, transaction conflicts, data-erasure policy, structured service logging, latency/error dashboards, admission control, deployment health checks and rollback are release/deployment requirements. Optional timings/counters provide library-level observability, not a monitoring service.
- MF refresh, truly held-out group choices, historically timestamped metadata, acceptance calibration and representative load/user studies remain limited. Validation observations are sparse/exposure-biased. These limits are recorded, not concealed by high percentile proxies.

## Frozen decisions

Keep the unbalanced Week 7 TF-IDF five-feature logistic artifact, scaler, selected Content configuration and profile-only MF bundle. Keep exact existing individual scores/order and diversity behavior. Preserve feature order, frozen population/catalog policy, Group V1 percentile formula/neutral .5 edge cases, mean default, three strategy names, provisional 70/30 weighting, .15 diversity and a 5 * limit group pool, immutable output schemas, bounded existing RankingCache and profile/validation/test boundaries. Diagnostics/rating helper are additive; model/config changes require versioned review, never silent promotion.

## Completion criteria

- [x] Group V1 integrated; three strategies tested.
- [x] Profile-defined conflict groups evaluated with methodological limits.
- [x] Add/edit/remove lifecycle, noninterference and cache invalidation tested.
- [x] MF no-fold-in limitation and concrete owner handoff documented.
- [x] Full-catalog candidate/fallback/empty/tie diagnostics pass.
- [x] Bounded cache, forced eviction, mutation invalidation and concurrency tested.
- [x] Reproducible cold/warm/update/eviction/startup/memory evidence saved.
- [x] Exact frozen individual full rankings and diversity preserved.
- [x] Deterministic group behavior, raw-scale invariance and no early truncation verified.
- [x] All relevant and full regression suites pass.
- [x] Production-readiness limitations reviewed and documented.
- [x] Updated documentation and Week 9 handoff complete.

Deferred deployment requirements are the listed production limitations, not undisclosed missing Week 8 feature work. No real group satisfaction claim, MF refresh implementation or production SLA is marked complete.

## Week 9 handoff (documentation only)

1. Pin the tested environment and trusted artifact manifest; review/check in only intended files after human review, with no automatic commit here. Retain all historical artifacts and recorded final-test results; do not rerun or tune against that frozen partition.
2. Re-run the documented reproducible benchmarks on fixed hardware with controlled workload and adequate repeats; add RSS/peak allocation, realistic multiworker concurrency, tail latency, admission limits and cold/eviction budgets before setting SLAs.
3. Define immutable artifact/version rollout, integrity/authenticity checks, worker warmup/health checks, cache lifecycle, atomic rollback and structured monitoring. Review authorization/persistence/event-version/erasure requirements for rating changes.
4. Obtain the Collaborative owner’s fold-in/refresh/deletion contract and evaluate any new artifact separately on validation before changing frozen behavior. Coordinate Content metadata/provenance updates with its owner.
5. Acquire consented real group-choice evidence; validate fairness/normalization/diversity and confidence handling independently before changing mean or 70/30. Keep frontend/REST/voting/social/streaming product work separately scoped.

## Evidence and Git

Reproduce with `python -m src.scripts.evaluate_week8_conflicts --output <new-path>` and `python -m src.scripts.diagnose_week8_serving --runs 2 --output <new-path>` from the repository root; diagnostics expect the archived baseline under the report parent. Scripts refuse report overwrite. Run `python -m pytest -q -p no:cacheprovider` for regression verification. See reports/week8/freeze for raw JSON, baseline, tests, smoke checks, manifest and preservation audit. No commits or pushes.
