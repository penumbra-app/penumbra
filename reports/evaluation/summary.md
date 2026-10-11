# Evaluation environment verification

The full-catalog development benchmark completed for 607 users in 266 seconds.
586 users had eligible positive outcomes; 21 were explicitly excluded from
positive-recall/NDCG macros. The median candidate pool was 4,949 movies, with
9,719 unique movies eligible across users. History items and movies not yet
observed by each prediction cutoff were excluded.

| Development metric | Structured core | Selected enriched TF-IDF |
| --- | ---: | ---: |
| NDCG@10 (primary) | 0.039654 | 0.049208 |
| Recall@10 | 3.027% | 3.982% |
| NDCG@20 | 0.044534 | 0.054291 |
| Recall@20 | 4.751% | 5.907% |

Selected-minus-core NDCG@10: +0.009554, paired-user 95% interval
[+0.004091, +0.015381], using 10,000 resamples. These are development results on
an inspected dataset, not confirmation. The earlier rated-only graded NDCG
numbers are not comparable to this binary full-catalog protocol.

85.25% of observed positives were available under the conservative cutoff proxy.
Only 4.23% of the selected model's top-10 recommendations had any held-out rating.
Unobserved recommendations remain unjudged, not confirmed dislikes. This limits
what the offline result establishes about real preference satisfaction.

The measured paired-user variance gives a planning minimum detectable NDCG@10
effect of approximately 0.00816 at 80% power. Detecting a 0.01 effect would require
approximately 390 independent eligible users under the same-variance normal
approximation; this does not guarantee power in a new cohort.

## Complete-request benchmark

Measured 9 fresh-process cold requests, 30 warm requests, and a 9-request burst
with two workers across short/median/long histories (12/43/1,618 ratings).
The public serving API ranked the full current 9,742-movie snapshot, excluded
history movies and serialized its top ten recommendations. Responses matched
across cold, warm and concurrent execution.

| History | Warm p50 | Warm p95 |
| --- | ---: | ---: |
| Short | 0.448 s | 0.484 s |
| Median | 0.577 s | 0.612 s |
| Long | 5.236 s | 5.386 s |

Mixed-history cold-process p95: 6.283 seconds. Two-worker burst p95 including
queueing: 16.688 seconds, with throughput 0.480 requests/second. Cold process
maximum high-water RSS was 237.94 MiB; the warmed benchmark process peaked at
284.52 MiB. This is a small local microbenchmark, not a production SLO estimate.
Long histories and queued concurrent requests remain performance bottlenecks.

## Verification and remaining requirements

All 155 tests pass, including in a clean temporary directory with no private
datasets or credentials. Tests cover catalog availability, history exclusion,
ranking metric hand calculations, model/serving score parity, outcome leakage,
user alignment for inference, and the complete confirmation guard lifecycle.
The GitHub Actions workflow was added; no remote workflow has been run yet.
Input/source hashes, per-user metric aggregation, and the outcome-free team
candidate export were verified.

The default future plan demonstrates registration against the frozen MovieLens
snapshot. MovieLens identities are anonymized: a real prospective confirmation
needs an accessible cohort, its own history, and a newly registered plan before
fresh outcomes are observed. Future confirmation refuses early/repeated runs and
changed snapshots. It cannot create fresh data or establish provenance by itself.

See [the workflow guide](../../docs/evaluation-environment.md),
[development report](development/results.json),
[latency report](latency/results.json), and [verification](verification.json).
