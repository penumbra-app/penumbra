# MovieLens 32M content experiment — 2026-10-08

The frozen TF-IDF content model improved the primary NDCG@10 metric on the reserved cohort. The gain was statistically distinguishable from zero, but smaller than the predeclared worthwhile difference of +0.01 absolute NDCG. Recall gains were not statistically distinguishable from zero.

Keep the existing TF-IDF model over its structured-only baseline on this evidence. This experiment does not establish that it is optimal among embeddings, ridge, top-k history matching or other variants; only this fixed two-model comparison was run.

## Reserved result

| Metric | Structured core | Core + TF-IDF | Relative change | Absolute gain: 95% paired bootstrap CI |
| --- | ---: | ---: | ---: | --- |
| NDCG@10 (primary) | 0.03093 | 0.03527 | +14.0% | +0.00434 [+0.00065, +0.00798] |
| Recall@10 | 0.02115 | 0.02232 | +5.5% | +0.00117 [-0.00178, +0.00392] |
| NDCG@20 | 0.03198 | 0.03583 | +12.1% | +0.00386 [+0.00054, +0.00708] |
| Recall@20 | 0.03116 | 0.03270 | +4.9% | +0.00154 [-0.00240, +0.00556] |

NDCG@10 increased by +0.00434, or 14.0% relative, with a 95% interval of [+0.00065, +0.00798]. The entire interval is below the +0.01 target: this is evidence of a modest ranking gain, not evidence that the target was met. NDCG@20 also has a positive unadjusted interval, but all secondary metrics are exploratory. Both recall intervals cross zero.

All 1,000 reserved users were scored; 987 had eligible held-out positives and contributed to NDCG/recall means. The other 13 are reported rather than silently treated as zero. The median candidate set contained 9,375 unseen movies. All 15,347 observed positive outcomes were eligible.

Top-10 recommended-catalog coverage increased from 26.12% to 32.36%. Only 3.73% of the TF-IDF top-10 recommendations had a held-out rating, so these known-positive metrics do not measure the true relevance of most recommendations.

## Development result

| Metric | Structured core | Core + TF-IDF | Relative change |
| --- | ---: | ---: | ---: |
| NDCG@10 (primary) | 0.03023 | 0.03857 | +27.6% |
| Recall@10 | 0.01950 | 0.02388 | +22.4% |
| NDCG@20 | 0.03254 | 0.03947 | +21.3% |
| Recall@20 | 0.03057 | 0.03618 | +18.4% |

Development used 1,000 users, with 991 eligible for positive-outcome metrics. NDCG@10 gained +0.00834 with 95% CI [+0.00484, +0.01183]. The smaller reserved effect is the result to use for the final claim. No parameters were changed after inspecting development results.

## Setup and checks

- The official ZIP was copied from Downloads to ignored `data/imports/ml-32m.zip`; all four published CSV checksums passed.
- All 32,000,204 ratings were scanned for sampling, overlap checks and item-availability timestamps. The model was not trained on all 32 million ratings.
- The catalog contains 9,448 movies matched to existing metadata; 9,364 have plots. Of 294 omitted old-catalog movies, 275 lacked a matching external identity and 19 had conflicting/ambiguous identifiers.
- Two disjoint cohorts of 1,000 users were chosen by a fixed hash rule. Each user first appears in the source after 2018-09-24; exact matches to old movie/rating/timestamp events were excluded.
- The first approximately 80% of each user’s catalog ratings formed history, with the final 20% held out. Timestamp groups remained intact.
- Development: 126,870 history ratings and 31,496 outcomes. Reserved: 130,532 history ratings and 32,478 outcomes.
- Both scorers use the same structured features, user histories, candidate sets and tie rule. The challenger adds the existing per-user-history TF-IDF plot/keyword adjustment. No community tags or collaborative features were used.
- The comparison, code, inputs, preset, primary metric and +0.01 target were frozen before either cohort was evaluated. Inference uses 10,000 user-paired bootstrap resamples, seed 42.
- All 166 automated tests passed. Post-run verification reproduced aggregate means and deltas from private per-user reports, confirmed zero cohort-user overlap, checked chronological splits, and verified that frozen files/source were unchanged.
- Development took 17.28 minutes; reserved evaluation took 17.31 minutes. They ran concurrently. These are batch evaluation times, not serving-latency measurements.
- ZIP, prepared ratings/metadata, per-user reports and the local frozen-code archive are ignored by Git. No API calls, commits or pushes were made.

## Interpretation limits

This is an offline test on the previously enriched catalog intersection, not the entire 32M movie catalog or a prospective online experiment. Retrospective metadata and observed-only feedback remain limitations. User IDs are anonymized across MovieLens releases; newer recorded activity and exact-event screening reduce overlap but do not prove disjoint person identities across releases.

Earlier NDCG values such as 0.7737 used rated-only candidates and graded relevance. Even the earlier full-catalog score of 0.04921 used different users, history splits, tags and candidate availability. Do not interpret lower absolute 32M scores as a measured regression; the controlled within-cohort comparison is the valid comparison here.

The reserved cohort has now been evaluated and is no longer untouched. Future model selection should use development data, with another predeclared uninspected holdout for a new final claim.

## Files

- [Reserved results](reserved/results.json)
- [Development results](development/results.json)
- [Frozen comparison](offline-plan.json)
- [Cohort provenance](cohort-summary.json)
- [Verification](verification.json)
- [Run and software versions](run.json)
- [Preparation and evaluation commands](../../docs/movielens-32m.md)
