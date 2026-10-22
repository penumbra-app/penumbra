# Penumbra Week 9 evaluation audit

Week 9 audits the frozen decision; it does not select or train a model. Historical evidence is retained byte-for-byte. No final-test evaluation is repeated.

## Recorded evidence and comparable baselines

`reports/week7/tfidf_final.json` records the matched original-Content versus selected-TF-IDF **five-feature** pairwise-logistic comparison: 600 users, 633,032 unequal-rating pairs. The request's “Old ML baseline” label refers here to original Content in this matched five-signal experiment, not automatically to the older three-feature production model.

| Historical final-test metric | Original Content baseline | Selected TF-IDF Hybrid |
|---|---:|---:|
| Macro pairwise accuracy | .635886 | .639273 |
| Pair-weighted accuracy | .684689 | .688359 |
| NDCG@10 | .908293 | .909509 |

`reports/week7/tfidf_comparison.json` and `finalization_validation.json` document validation. Matched balanced/unbalanced validation has 563 users and 165,790 pairs; unbalanced remains selected. Balanced validation point estimates do not authorize promotion or another test run. Week 5/6 Collaborative-only, Content-only and ablation reports use historical 60/20/20 splits and/or different catalog, Content, candidate and feature protocols. Their scores are contextual evidence, not a matched leaderboard against this selected model. The older `week7_user_balanced.joblib` includes the current validation interval in training and is incompatible with this validation comparison.

## Inspected protocol

- `ml_reranker.chronological_split` sorts each user's rows by timestamp then movie ID, stable; profile ends at floor(.60*N), Hybrid training at floor(.70*N). Its generic default is 60/20/20: Week 7 callers explicitly pass train_fraction=.10. `evaluate_week7_tfidf` splits the remainder at floor(.80*N)-floor(.70*N). Users with fewer than ten events are excluded from the historical protocol. Users can occur in all partitions; events are separated, not users.
- Hybrid dataset construction and MF evaluation fit MF exclusively on concatenated first-60% profiles, random seed 42. Content constructs target profiles from profile rows only, and uses fixed selected metadata/configuration. Quality/Popularity references exclude the target user from the profile population. Frozen serving preserves that population and MF, even after rating updates.
- Training forms positive/negative feature differences for every unequal-rating pair, skipping equal outcomes. The selected unbalanced model gives each signed example equal weight, so prolific users dominate training. Optional equal-user weighting exists but is not selected. StandardScaler and logistic regression fit training differences only; serving uses that frozen scaler, not request-fitted scaling. Logistic random_state=42.
- Historical evaluation candidates are rated outcomes in the requested held-out slice, further intersected with available MF/popularity/personal/quality evidence. They are **not** the entire unseen serving catalog. Pairwise correct/incorrect/tied-prediction credit is 1/0/.5. Macro averages contributing users; weighted divides total pair credit by total pairs. NDCG@10 uses linear star relevance with logarithmic discount and stable input-index ties, not binary relevance or the separate newer `ranking_metrics` helper. Historical evaluation scores precede serving diversity; it does not measure diverse full-catalog slates.
- Individual serving sorts raw scores descending then ID, excludes supplied watched IDs, intersects supplied catalog with frozen catalog, and applies the unchanged .15 diversity penalty to the first 5*K pool. Group scores the full shared unseen pool per member before percentile normalization/aggregation, then diversifies 5*K. Mean remains default; .7 mean+.3 minimum remains provisional.

## Leakage and reproducibility limits

Upstream original-Content selection reports use 60/20/20 validation, overlapping the later Hybrid 60-70% training and 70-80% validation intervals. The selected TF-IDF handoff preset records validation metrics but lacks a complete split manifest. Do not describe Hybrid validation as nested, selection-independent evidence; these provenance limits do not authorize another final-test run.

Per-user order is not a global time fence: other users' profiles may occur later than a target outcome. Metadata/tags are not audited as-of snapshots. Catalog coverage and observed held-out exposures limit claims; unknown items and sparse/cold extensions lack matched historical quality metrics. No actual group-choice labels establish group acceptance or fairness. Week 8 conflict proxies are useful diagnostics, not independently validated satisfaction.

Bundle SHA, selected-config SHA, feature order, Python/package pins and current source/data checksums are recorded in the release manifest and preservation inventory. The bundle records original ranker/catalog/ratings hashes and profile-only MF provenance. Original training manifests lack complete dependency/input hashes; current checksums cannot retroactively prove those original inputs or exact retraining reproducibility. Restore the frozen artifact rather than regenerate it.

Week 9's benchmark rebuilds only first-60% profiles and compares the full frame exactly with bundled references. It repeats frozen scoring on prespecified users 23/69/84 using only 70–80% validation candidate IDs, checks full scores exactly, and records cohort sizes. This is deterministic serving reproducibility, not a re-estimate of the historical aggregate metrics. No fitting occurs. Complete individual scores and diversity outputs also compare with the source snapshot archived at Week 9 start, and the existing Week 8 golden/feature-equivalence tests remain in the full suite. See `reports/week9/benchmark.json` and the full test log for executed evidence.
