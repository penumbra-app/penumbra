# Decision from the controlled experiment

Retain the enriched preset in `reports/tmdb-full/selected_config.json` for the
existing pairwise-accuracy objective. Among 17 validation settings, the original
per-user TF-IDF model remains the winner (59.9719% validation pairwise accuracy).
This is the enriched preset, not the older sparse-tag default in
`src/content/selected_config.json`. No serving preset was changed.

Qwen has the highest test pairwise point estimate (57.6042% versus 57.4692%), but
its gain is inconclusive: +0.1350 percentage points, paired-user 95% interval
[-0.5829, +0.8666]. It also loses on validation (59.2140%). MiniLM loses on
validation (58.9213%), so Qwen represents the semantic family in the test table.
Neither encoder is fine-tuned, and both use the same similarity scorer. This
does not establish that semantic embeddings cannot help under a different scorer.

Shared-catalog TF-IDF is the most useful speed/top-10-ordering tradeoff. Test NDCG
increases from 0.773654 to 0.777279: delta +0.003625, 95% interval
[+0.001098, +0.006222]. This is a secondary exploratory finding, not the
predeclared selection objective. Pairwise improvement is inconclusive, validation
pairwise is lower, and RMSE worsens from 1.019924 to 1.021399. Cached text scoring
for the 20,156 test candidates takes 0.545 seconds versus 6.984 seconds for the
baseline; shared index construction takes another 0.527 seconds. These are
single-run text-stage timings, not end-to-end serving latency.

Separate fields select 25% plot / 75% keywords on validation but do not establish
an accuracy gain. Ridge selects alpha=10; top-k selects k=50. Both use per-user
TF-IDF because it wins representation selection. Ridge does not help here.
Top-50 slightly improves RMSE by 0.000247 (95% interval for new minus old:
[-0.000367, -0.000130]), but pairwise accuracy is slightly lower and inconclusive.
That tiny secondary error improvement does not justify switching the primary
ranking model.

Frozen preprocessing took 51.6 seconds for MiniLM and 2,150 seconds (35.8 minutes)
for Qwen, including model loading/download. It ran locally on the Mac GPU and is
cached for future experiments. Predictions require no API credential.

The input snapshot and user split are identical, and baseline metrics reproduce
the previous report. All 142 tests pass. User bootstrapping uses 10,000 resamples
and reports 99% Bonferroni intervals for the five primary family comparisons;
none establishes a positive pairwise gain. The previously inspected test set,
retrospective metadata and transductive catalog representation limit the claim.
Use a new holdout to confirm the promising NDCG/speed tradeoff before changing
the selection objective or promoting a model.
