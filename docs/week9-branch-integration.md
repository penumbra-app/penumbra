# Penumbra Week 9 branch integration — 2026-10-26

## Scope

Integrated `origin/content-model` (9884593) and `origin/collaborative-model`
(eea9635) into `main`, starting from current remote main 33c9ff7. The next
integration step merges this main into `hybrid-group-reranking`; it does not
promote experimental Content/Collaborative models into the frozen Week 7 ranker.

## Conflict resolutions

- Retained the Content owner's exact current algorithms and public API. Removed
  an obsolete `max_cast_members` argument introduced by Git's automatic merge
  by restoring the exact owner version of `src/content/profiles.py`.
- Retained shared metadata/evaluation adapters and combined metric functions
  already verified on the Hybrid branch, rather than dropping either API.
- Updated legacy cast tests to the already verified current-API tests from
  Hybrid. Both the owner's new tests and main's other tests remain collected.
- Reused the tested `as_posix()` manifest-key fix in `src/prepare_ml32m.py` so
  Windows dataset producers and cross-platform validators agree.
- Reused Hybrid's pinned serving dependencies and full-pytest CI workflow.
- Preserved local CSVs as ignored files after Content removed them from tracking.
  Ratings SHA-256 remains
  `aa289ca83157595d0df6aea1be6a4ded676ddc4385472e8313a8ed9805352646`.
  Pre-existing untracked reports, private handoff inputs and archives were not staged.

## Main validation

`python -m pytest -q -p no:cacheprovider`: **305 passed, 83 subtests passed,
22 existing NumPy/joblib warnings, 0 failures/skips**, in 9.95 seconds.

`python main.py`: succeeded. `git diff --check`: passed for compatibility fixes.
No real models were retrained and no final-test evaluation was run.
Main has a smaller test population than Hybrid because Hybrid-owned serving,
group and Week 7–9 regressions live on the Hybrid branch.

The upstream MovieLens README contains a literal seven-equals heading underline;
Git's staged whitespace check flags it as a conflict marker when newly added.
It is dataset documentation, not an unresolved merge. Git's unmerged-file list
is empty; source conflicts were resolved explicitly.
