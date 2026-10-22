# Penumbra Week 9 performance

Measured locally on Windows/CPython 3.13.14 with the exact pinned serving environment. Frozen profile histories (60,255 ratings), full 9,742-film catalog, K=10, mean group default; profiles define the same similar/conflicting/outlier cohorts as Week 8. Synthetic sparse/cold users extend that profile population. No final-test inference/metrics or fitting occurs. This workload differs from CLI full histories and older Week 8 runs: no cross-report speedup claim is supported.

## Serving latency

Three cold cache-clear runs and 30 warm requests per workload. One forced-eviction measurement per case is diagnostic, not a latency distribution. Existing cold samples initially overlapped full tests; their original cold 8.218s median and warm samples remain in benchmark.json but these are **not the reported clean measurements**: existing-cold-serial.json supplies both cold and warm serial reruns below. Other test/install processes finished before the later cases. Developer workstation background activity is uncontrolled; these are local engineering measurements, not service guarantees.

| Workload | Cold median [min,max] | Warm median [min,max] | Evicted |
|---|---:|---:|---:|
| existing (1 member) | 6.179s [3.917, 8.499] | 0.265s [0.235, 0.364] | 8.262s |
| sparse (1 member) | 2.110s [1.979, 2.279] | 0.133s [0.091, 0.226] | 2.179s |
| cold (1 member) | 0.883s [0.874, 0.964] | 0.094s [0.088, 0.135] | 0.980s |
| similar_pair (2 members) | 6.341s [6.178, 7.283] | 0.257s [0.192, 0.409] | 6.628s |
| one_outlier (3 members) | 12.066s [11.927, 15.402] | 0.371s [0.306, 0.554] | 14.798s |
| conflicting_pair (2 members) | 8.936s [7.516, 13.946] | 0.558s [0.506, 0.770] | 18.151s |
| larger_five (5 members) | 40.103s [36.235, 40.524] | 1.350s [1.281, 1.731] | 49.724s |

Cold min/max and warm min/max are reported; warm n=30 does not support credible tails. p90/p95/p99 are deliberately null at this sample size. The harness emits p90 only for >=100, p95 for >=200, p99 for >=1,000 samples. Even then these are empirical workload quantiles, not production SLAs. No p99 is reported in Week 9.

## Phase diagnostics

Cold phase medians summed across group members; warm fingerprint/aggregation/diversity below are the **last sample**, not distributions. Existing initial cold phases include test contention and must be treated as exploratory. Timings are nested: do not add parent/child durations. Group normalization and feature-total/ranking/metadata/request-prepare timings are available in JSON. The per-member phase data measures both Content model construction and prediction; it does not alter teammate code.

| Case | Content build | Content predict | Population | Personal | MF | Warm fingerprint | Warm aggregation/sort | Warm diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| existing | 0.710s | 4.687s | 0.074s | 1.987s | 0.361s | 0.078s | 0.0000s | 0.0012s |
| sparse | 0.370s | 1.144s | 0.027s | 0.328s | 0.108s | 0.114s | 0.0000s | 0.0016s |
| cold | 0.304s | 0.314s | 0.021s | 0.000s | 0.074s | 0.094s | 0.0000s | 0.0012s |
| similar_pair | 0.721s | 3.903s | 0.063s | 1.128s | 0.201s | 0.152s | 0.0064s | 0.0012s |
| one_outlier | 1.183s | 7.907s | 0.132s | 1.896s | 0.377s | 0.277s | 0.0115s | 0.0019s |
| conflicting_pair | 0.867s | 5.907s | 0.094s | 1.428s | 0.211s | 0.420s | 0.0173s | 0.0026s |
| larger_five | 3.467s | 23.992s | 0.366s | 8.068s | 1.625s | 1.125s | 0.0170s | 0.0026s |

Content prediction dominates cold work. Group cold cost grows with per-member complete-catalog scoring. Warm fingerprinting hashes model/catalog/population/history by value to preserve mutation invalidation; replacing it with identity-only keys could return stale results. No speculative cache or algorithm optimization was introduced in Week 9. Deployment mitigation: immutable preloaded workers, bounded queues/timeouts, selective worker-local warmup of common histories, capacity planning for cache eviction and cold groups, and Content-owner profiling for future approved improvements. Cache misses may duplicate concurrent inference; no single-flight/distributed-cache claim.

Matched alternating warm score-prefix measurements (30 pairs each; same profile/catalog, diversity excluded) compare the archived start-of-Week-9 function with current code. Exact complete scores match. These are preservation/overhead checks, not an optimization claim:

| Case | Before median | After median |
|---|---:|---:|
| existing | 0.250s | 0.258s |
| sparse | 0.278s | 0.268s |
| cold | 0.130s | 0.126s |

## Startup, updates, load and memory

Fresh process startup/import/load samples: [{"process_wall": 9.447879900006228, "imports": 8.704301000005216, "artifact": 0.12910310000006575}, {"process_wall": 3.1761812000040663, "imports": 2.6389156999939587, "artifact": 0.03233999999065418}, {"process_wall": 2.9795247000001837, "imports": 2.4516945999930613, "artifact": 0.0507458000065526}]. Parent verified release initialization took 0.482s; it includes checksum/dependency checks and deserialization. CSV loading took 0.311s. Process wall includes launch + imports + artifact loading; do not count them as independent additive requests. Initial startup overlapped tests and is exploratory, not a clean startup SLA.

Add/edit/remove were executed for every workload. Each pure helper duration and subsequent inference duration is in benchmark.json; all individual updated full scores match an explicit uncached recomputation, and every updated group equals a fresh-cache recomputation. Remove may legitimately restore a prior key, and edits invalidate the affected member while other member keys remain reusable. Helpers do not persist or update MF. Forced LRU eviction returns the original exact output. Warm last-phase cache-hit flags and bounded occupancy are recorded.

Controlled warm individual load: **200 requests, 4 threads, 48.112s elapsed, 4.16 requests/s**. Per-request median 0.972s, p90 1.101s, p95 1.148s, min 0.509s, max 1.294s. Includes RatingHistory snapshot and final diversity. All complete outputs match. This is a closed-loop local queue-free worker test, not offered-load saturation or multi-process behavior. Cache counters: {"entries": 1, "candidates": 9670, "hits": 200, "misses": 1, "evictions": 0, "max_entries": 16, "max_candidates": 200000}; seed cold miss is included. Hit rate 99.50%, evictions per measured request 0.000. No production eviction rate is inferred from deliberate forced insertions.

Group concurrency: 8 warm requests / 4 threads in 4.468s; full outputs exactly match. Sample too small for tails. Regression tests also verify an update mid-request preserves the old request snapshot and publishes the new history only to subsequent requests.

Peak process RSS/working set: **252.6 MiB** (Windows process working set); starting process RSS 159.4 MiB. Peak includes real serving, retained benchmark/reference outputs and a separate synthetic fill of production cache bounds; it is not an isolated cache footprint or a validated worker ceiling. Synthetic bounded-cache occupancy: {"entries": 16, "candidates": 200000, "hits": 0, "misses": 0, "evictions": 4, "max_entries": 16, "max_candidates": 200000}. Windows working-set peak is OS-reported process memory, unlike Week 8's Python-only tracemalloc allocation measurement. Neither proves multiworker/container memory capacity.

## Reproduce and limitations

From repository root in the pinned environment:

```sh
python -m src.scripts.benchmark_week9 --output reports/week9/benchmark-rerun.json --cold-runs 3 --warm-runs 30 --load-requests 200
python -m src.scripts.verify_week9 reports/week9/verification-rerun
```

Reports refuse final-file overwrite; retain baseline sources under reports/week9/baseline. Run benchmark/test/install jobs serially on fixed hardware for release measurement. Current performance permits backend integration experiments with explicit cold budgets; admission control, representative mixed loads, target-platform staging, true production tails, privacy review and cold Content costs remain deployment gates.

## Rating update samples

One sample per operation/workload; each cell is pure in-memory helper time / subsequent full inference time. These are diagnostics, not tail estimates; persistence is excluded. All fresh recomputation assertions pass.

| Workload | Add | Edit | Remove |
|---|---:|---:|---:|
| existing | 2.54 ms / 4.585s | 5.07 ms / 3.722s | 4.70 ms / 0.136s |
| sparse | 2.43 ms / 2.364s | 2.90 ms / 2.388s | 2.28 ms / 0.114s |
| cold | 2.76 ms / 1.534s | 3.19 ms / 1.809s | 2.55 ms / 0.103s |
| similar_pair | 2.14 ms / 6.758s | 4.28 ms / 3.493s | 2.82 ms / 6.482s |
| one_outlier | 4.68 ms / 11.956s | 3.04 ms / 3.614s | 3.07 ms / 11.665s |
| conflicting_pair | 4.74 ms / 18.064s | 5.69 ms / 7.131s | 28.23 ms / 15.151s |
| larger_five | 6.62 ms / 41.134s | 6.20 ms / 7.236s | 7.19 ms / 44.117s |
