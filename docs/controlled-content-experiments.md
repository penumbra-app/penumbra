# Controlled content experiments

The experiment is separate from the serving model and never changes its preset.
It compares the enriched validation-selected TF-IDF model against a shared
catalog vocabulary, separately normalized plot/keyword fields, frozen MiniLM and
Qwen embeddings, per-user ridge, and top-k history matching. See the generated
[`summary.md`](../reports/content-controlled/summary.md) and
[`results.json`](../reports/content-controlled/results.json) for measured results.

## Reproduce

Restore the private datasets as described in [the team handoff](hybrid-handoff.md).
From the repository root:

```bash
.venv/bin/python -m pip install -r requirements-embeddings.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m src.encode_content
.venv/bin/python -m src.experiment_content
```

`src.encode_content` downloads public model weights and computes embeddings
locally; it does not upload movie text or use TMDB credentials. The first run is
substantially slower than subsequent runs. Model weights, document-derived
vectors, per-user metrics, and prediction arrays are ignored by Git. Aggregate
reports and frozen validation choices are suitable for the public repository.
Hash validation prevents reusing embeddings with changed movie text or tags.
The encoder report records immutable model revisions, pooling, device,
truncation length, cache checksum and inference time. Cached embeddings can be
used without network. Model revisions are pinned in `src/encode_content.py`, so
a fresh download also uses the reported revisions rather than a changing branch.

Models are [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
Use their default pooling and normalized movie/movie vectors without a query
instruction. MiniLM is capped at 256 tokens and Qwen at 512 for this experiment;
longer descriptions can be truncated. No model is fine-tuned on these users.

## Controls

- Identical per-user chronological 60/20/20 splits, grouped equal timestamps;
  the first 60% is frozen for both validation and test.
- Identical structured core, recency weights, personal-mean rating residuals,
  text adjustment prior 5, weight 1, and cap 0.5. Similarities below zero do not
  contribute evidence. Scores are clamped once after the core/text combination.
- Identical candidate/history text, including only the permitted historical
  tags from other users. Shared vocabulary and IDF fit on the static TMDB catalog
  only; personalized permitted tags are transformed without refitting IDF.
- Shared vocabulary uses unigrams, min_df=1, max_features=10,000, sublinear TF,
  English stop words and L2 normalization, matching the selected baseline.
- Separate fields use their own vocabulary/IDF, with plot weight in
  `[0, 0.25, 0.5, 0.75, 1]`; block square-root scaling and final normalization
  produce weighted cosine similarity and handle missing fields.
- The best representation is selected by validation pairwise accuracy, RMSE
  tie-break. It can remain the original per-user representation. Ridge tests
  alpha `[0.1, 1, 10, 100]`; top-k tests `[5, 10, 20, 50]`.
- Ridge learns only each user's residuals with recency sample weights and no
  intercept. It replaces the bounded text adjustment; it does not add user IDs,
  collaborative features, or a new hybrid reranker.
- A selection artifact is written before test scoring. The original baseline's
  four test metrics must reproduce the saved enriched report within 1e-10.
- 10,000 paired user bootstrap resamples, seed 42; 95% intervals and Bonferroni
  99% intervals for five primary family comparisons. Secondary metrics and
  history-size slices are exploratory, not independent confirmation.

The full catalog text is transductive and retrospective; it includes movies
that would not have existed at every user's historical cutoff. These results
do not establish performance with historically available catalog metadata.
The test set has been inspected before; use a new holdout for confirmation.
These experiments measure ranking among held-out rated movies, not full-catalog
retrieval or online engagement. Frozen semantic similarity does not necessarily
predict preference, and the tested scorer may not suit every embedding geometry.

The runner deliberately keeps the serving implementation and its confidence
semantics unchanged. Promote a new configuration only after reviewing its
accuracy, runtime tradeoffs and a fresh holdout.
