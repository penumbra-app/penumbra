# Penumbra Week 7 finalization

This work preserves the existing uncommitted changes, teammate-owned Content/MF
implementations, final reports, and selected artifacts. No commits or pushes.
The pre-existing edits to `src/hybrid/content_adapter.py`,
`src/hybrid/ml_reranker.py`, and `src/scripts/week6_ablations.py` were retained;
this task did not edit those three files.

## Audit before changes

| Requirement | Evidence at audit | Status at audit | Action |
|---|---|---|---|
| User balance | `ml_reranker.equal_user_pair_weights`, weighted `train_ranker`; TF-IDF training unweighted | PARTIAL | Separate validation-only balanced candidate |
| Macro/weighted pairwise and NDCG@10 | `evaluate_week7_tfidf.evaluate` | COMPLETE | Preserve final report |
| Frozen TF-IDF | `content_adapter.build_content_model_from_frames`; offline only | PARTIAL | Explicit serving bundle |
| Five signals | `AblationCandidate`; production used `MovieFeatures` | PARTIAL | Five-signal serving route |
| Split isolation | Week 7 60/10/10/20; older scripts 60/20/20 | COMPLETE for selected Week 7 run | Regression boundary test |
| MF downstream target leakage | `build_ablation_training_dataset`: MF fits first 60% only; evaluation reproduces this | COMPLETE for inspected Week 7 paths | Test exact fit rows |
| Offline/production consistency | No shared artifact contract | PARTIAL | Fixed order, scaler/model checks, equivalence test |
| Cold start/missing signals | Old serving rejected users without history | PARTIAL | Explicit neutral fallback semantics |
| Regression coverage | 59 existing hybrid/evaluation tests passed | PARTIAL | New serving/balance tests |
| Group recommendations | `docs/group-recommendations.md`: no implementation | MISSING | Needs fairness/aggregation design decision |

## Code and operation

- `src/hybrid/week7.py`: five-signal serving with exact order
  collaborative/personal/content/quality/popularity, fitted scaler, deterministic
  movie-ID tie-break, ID-based MF join, and finite score checks. Content predictions
  use the teammate's public batch API. User history is the only TF-IDF fitting
  data; the bundled catalog keeps metadata consistent with offline training.
- `src/hybrid/recommender.py`: keyword-only `week7_bundle` selects the new path,
  retains the larger candidate pool and existing diversity stage. Existing calls
  continue to use the old path.
- `main.py`: `--week7` selects the prepared bundle explicitly. Missing or invalid
  bundles fail clearly; there is no automatic fallback to a different ranker.
- `prepare_week7_serving.py`: packages the frozen selected TF-IDF ranker,
  catalog, preset hash, profile-only population snapshot, and profile-only MF.
  Refuses overwrites. Does not replace any existing model.
- `train_week7_tfidf.py`: optional `--user-balanced`, equal user weights in
  **both** scaler and logistic fit, new manifest statistics and runtime.
  Signed reverse pairs are intentional class balancing, preserved from training.
  No pair subsampling, seed changes, label changes, or hyperparameter search.
- `evaluate_week7_tfidf.py`: supports named content variants and opt-in per-user
  metric retention; corrects obsolete split documentation, checks partition names,
  refuses report overwrites. Default feature/scoring calculations are preserved.
- `verify_week7.py`: validation-only matched balance comparison, paired user
  bootstrap, and real full-catalog serving smoke checks. No final-test evaluation.
- `tests/hybrid/test_week7.py`: balance, split/MF fit boundary, feature-order/width,
  missing MF, cold/sparse/existing users, metadata gaps, no duplicates, watched
  exclusion, deterministic ranking, offline/serving equivalence, paired bootstrap.
- Older hybrid configuration loaders in `evaluate_ranking.py`,
  `train_hybrid_reranker.py`, `evaluate_week6_ablations.py`,
  `evaluate_week6_gbt.py`, `evaluate_week6_gbt_ablations.py`,
  `train_week6_gbt.py`, and `week6_gbt_ablations.py`: ignore the retired
  `max_cast_members` field when constructing the updated Content `ProfileConfig`.
  This fixes a constructor failure; it does not edit the saved configurations.

Serving fallback definitions: missing/nonfinite MF = 2.5, nonfinite content = 2.5,
no personal history = 0, absent population quality/popularity = 0. Malformed
prediction schemas or duplicate MF results raise. These cold-start extensions
are smoke-tested, not covered by the reported warm-user ranking benchmark.
The existing genre scorer omits movies without population evidence, so their
personal signal also defaults to zero; Content and MF still contribute when available.

## Evaluation integrity

The selected rerankers fit 60–70% outcomes; MF and Content profiles use first 60%.
Validation is 70–80%; final test is 80–100%. Users recur across partitions by
design; outcomes are isolated, not disjoint user cohorts. MF can legitimately
fit target users' **profile** ratings. It does not fit their downstream labels.
Quality/popularity reference features exclude the target user's ratings.

This protocol is per-user chronological, not a single global wall-clock cutoff.
Other users' profile timestamps can be later than a target user's outcomes. The
enriched metadata is a static catalog, not an audited historical snapshot. Do not
describe these results as a strict point-in-time production backtest. No scoring
changes were made to the final benchmark to retrofit a different protocol.

Pairwise metrics exclude equal-rating pairs and give prediction ties half credit.
Macro accuracy averages eligible users; weighted accuracy divides total credit
by total unordered pairs. NDCG@10 uses raw ratings as linear relevance gains
(`src/evaluation/metrics.py:ndcg_at_k`), with a deterministic candidate-order
tie-break, and is averaged over users who have unequal-rating pairs.

The metrics rank held-out **rated** movies with population evidence and unequal
ratings. They do not measure full-catalog retrieval, exposure bias, diversity,
novel movies, group satisfaction, or production latency. Saved final reports
contain aggregate metrics only; final per-user bootstrap intervals cannot be
reconstructed without rerunning final test, which this work does not do.

## Preserved final TF-IDF test results

From `reports/week7/tfidf_final.json`; 600 users / 633,032 unequal-rating pairs.

| Metric | Baseline | TF-IDF |
|---|---:|---:|
| Pairwise macro | 0.6358855602 | 0.6392733058 |
| Pairwise weighted | 0.6846889257 | 0.6883585664 |
| NDCG@10 | 0.9082933983 | 0.9095092282 |

These are separate from the validation-only balance comparison. No final test
rerun, further model selection on test, or statistical-significance claim.

## Upstream responsibilities and release limits

No confirmed downstream-target leakage defect was found in the inspected Week 7
MF path. No teammate-owned model change is required for that boundary.

- Collaborative: `BiasedMatrixFactorization.fit/predict` has no public fold-in
  interface for updating an already trained user's factors with new history.
  The bundle deliberately uses the evaluated profile-only MF. Before live refresh,
  agree on a public refresh/fold-in contract and evaluate it on validation; verify
  no hidden downstream outcomes enter the fit. Do not silently swap in
  `selected_mf.joblib`, which has a different provenance/configuration.
- Content/data: `load_selected_model` consumes supplied metadata and
  `TextProfile` fits only history. Strict historical claims additionally need
  as-of metadata/tag snapshots with target-user tag exclusion. Coordinate that
  data protocol with the Content owner and rerun a separate validation backtest.
- Group: no dedicated implementation exists. Choose aggregation/fairness and
  group-level evaluation before claiming group production readiness.
- Artifact provenance: the original TF-IDF manifests lack input checksums and
  dependency versions. The new bundle records hashes of the current local files;
  these cannot retroactively prove the original training inputs. Retain the
  original manifest/report together and add complete provenance to future runs.
- Serving uses a fixed catalog/population snapshot. Refresh and cache lifecycle,
  request latency budget, full-catalog ranking validation, and deployment are
  Week 8 work. The new path is opt-in; the balanced candidate is not promoted.

## Independent PowerShell checks

```powershell
.venv\Scripts\python.exe -m pytest tests src/tests src/collaborative src/evaluation -q -p no:cacheprovider
.venv\Scripts\python.exe main.py --week7
.venv\Scripts\python.exe -m src.scripts.verify_week7 --artifacts-only --output reports/week7/artifacts_independent.json
.venv\Scripts\python.exe -m src.scripts.verify_week7 --smoke-only --output reports/week7/smoke_independent.json
```

To reproduce the validation comparison in fresh output paths (never reuse a
path containing an existing report/artifact):

```powershell
.venv\Scripts\python.exe -m src.scripts.train_week7_tfidf --variant tfidf --user-balanced --output-dir src/models/week7_balanced_independent
.venv\Scripts\python.exe -m src.scripts.verify_week7 --balanced-model src/models/week7_balanced_independent/tfidf_plus_personal.joblib --output reports/week7/validation_independent.json
```

If restoring without the local serving bundle, prepare it first:

```powershell
.venv\Scripts\python.exe -m src.scripts.prepare_week7_serving
```

Local artifacts and reports are generated for review only; none were committed.

## Verification evidence

- Before changes: 59 hybrid/evaluation tests passed.
- Final suite: **375 passed, 83 subtests passed**, 36 NumPy/joblib deprecation
  warnings; no test failures. Command is listed above.
- `python -m compileall -q main.py src/hybrid src/scripts` passed.
- `git diff --check` passed.
- `python main.py --week7` succeeded and returned ten recommendations for user 23.
- Real catalog: 9,742 movies; bundle MF fit on 60,255 profile ratings. Frozen
  baseline/TF-IDF artifacts and the new balanced artifact load successfully.
- `reports/week7/finalization_artifacts.json`: actual baseline, TF-IDF, balanced,
  and legacy artifacts load with their expected widths; selected TF-IDF is active
  with a fitted 10,000-term vocabulary. Preset hash matches the serving bundle.
- `reports/week7/finalization_smoke.json` and `finalization_smoke_batch.json`:
  existing/single-rating/cold users each returned ten unique unseen movies,
  deterministic across repeated calls; missing MF uses neutral 2.5.
  Top-10 IDs match between single-item and batch Content calls.
- Observed batch request times: existing 71.36 s, sparse 9.64 s, cold 4.97 s.
  These are uncached local runs with background training; they are diagnostics,
  not a controlled performance comparison. Warm-user latency needs work.

Balanced training: 564 users, 361,830 signed examples (180,915 unordered
unequal-rating pairs). Examples/user: min 2, median 37, mean 641.54, max 36,642;
25th/75th percentiles 8/209.5. All contributing users have equal total weight.
The manifest records a wall duration of 17,314.43 s; no comparable historical
unbalanced duration exists, so runtime impact cannot be established from that
number. The feature-generation procedure is unchanged; weighting retains all
examples. See `src/models/week7_balanced_validation/tfidf_plus_personal.json`
for the full per-user count distribution and reproducible configuration.

The older `week7_user_balanced.joblib` used 60/20/20 splitting and a different
Content configuration. Its training outcomes include the current 70–80%
validation segment, so it is not a valid candidate for this validation
comparison. The new balanced TF-IDF model uses the matched 60/10/10/20 split.

## Matched balance validation results

Source: `reports/week7/finalization_validation.json`. Both models use selected
TF-IDF, identical features, profile-only MF, catalog, labels and candidate set.
563 users and 165,790 unordered unequal-rating pairs in the 70–80% validation
partition. The unbalanced metrics exactly reproduce the TF-IDF values in the
previous `reports/week7/tfidf_comparison.json`.

| Validation metric | Unbalanced TF-IDF | Balanced TF-IDF | Difference |
|---|---:|---:|---:|
| Pairwise macro | 0.6549483025 | 0.6572958227 | +0.0023475201 |
| Pairwise weighted | 0.6895289221 | 0.6917003438 | +0.0021714217 |
| NDCG@10 | 0.9350496482 | 0.9359806660 | +0.0009310178 |

Paired percentile bootstrap: 2,000 resamples of users, seed 42. Weighted accuracy
is recomputed from total credit/pairs within each resampled user cohort.

| Difference | 95% user-bootstrap interval |
|---|---|
| Pairwise macro | [-0.00332894, +0.00833702] |
| Pairwise weighted | [+0.00044469, +0.00411586] |
| NDCG@10 | [-0.00003798, +0.00195221] |

All point estimates improved. Only the weighted-accuracy interval excludes zero;
macro and NDCG uncertainty includes no improvement. These are conditional on the
fixed fitted models and evaluation protocol, and do not include retraining
uncertainty. No claim that all ranking gains are statistically established.
Evaluation wall time was 1,542.96 s. Per-user metrics and full training statistics
are retained in the new report. The selected serving bundle continues to use the
frozen **unbalanced** TF-IDF artifact; the balanced candidate is available in its
separate directory and was not silently promoted or evaluated on final test.

## Final Week 7 completion status and Week 8 readiness

| Requirement | Final status | Evidence |
|---|---|---|
| Per-user training balance | COMPLETE | `train_week7_tfidf --user-balanced`; equal weight in scaler/model; new artifact and matched validation report |
| Three evaluation metrics | COMPLETE | `evaluate_week7_tfidf.evaluate`; preserved test report and new validation report |
| Selected TF-IDF integration | COMPLETE | `prepare_week7_serving`, `load_bundle`; preset hash/vocabulary checks |
| All five hybrid signals | COMPLETE for explicit Week 7 path | `week7.build_feature_rows`, `recommend_week7`; `main.py --week7` succeeds |
| Train/validation/test isolation | COMPLETE within declared per-user protocol | 60/10/10/20; MF-fit regression test; final test not rerun |
| MF downstream target leakage safeguards | COMPLETE for inspected Week 7 paths | MF fit only on profiles; exact fit-row test; Content sees only history |
| Offline/serving feature equivalence | COMPLETE for supported warm inputs | TF-IDF pairwise feature-equivalence regression test; identical feature order and scaler |
| Cold/sparse/missing-signal handling | COMPLETE for graceful behavior | Real-catalog smoke reports and missing metadata/MF regression tests; quality for these extensions remains unmeasured |
| Regression coverage | COMPLETE | 375 tests and 83 subtests passed; compilation and whitespace checks passed |
| Strict historical data provenance | PARTIAL | No global cutoff or audited as-of metadata; original training manifests lack checksums |
| Current runtime artifact compatibility | COMPLETE | `finalization_artifacts.json`; all required widths load and preset hash matches |
| Group recommendation implementation | NOT STARTED | `docs/group-recommendations.md`; requires agreed aggregation/fairness design |
| Live refresh/performance/full-catalog quality | PARTIAL | Fixed snapshot, opt-in path; warm smoke requests about 71–80 seconds; no full-catalog ranking benchmark |

**Week 8 recommendation: CONDITIONAL GO.** The five-signal pipeline is ready for
Week 8 integration and release engineering within the documented evaluation
protocol. Before enabling live serving, address warm-user latency and cache/
refresh lifecycle, validate full-catalog ranking and fallback quality, confirm
the intended default model/path, and preserve artifact/input provenance. If the
release includes groups or strict point-in-time claims, those remain additional
blockers. Content/MF internals were not changed; upstream responsibilities and
verification steps are listed above. No further changes or tests remain running.
