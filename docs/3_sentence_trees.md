# Sentence Trees

Per-sentence phylogenetic trees are inferred from audio embeddings, then
combined into a species-level supertree via ASTRAL.

## Pipeline overview

The pipeline has two stages:

1. **Tree inference** (one of):
   - _Discrete trees_: embeddings are discretized into characters, then
     maximum-likelihood trees are inferred per sentence via IQ-TREE.
   - _Distance trees_: pairwise distance matrices are computed from embeddings,
     then distance-based trees are inferred per sentence via FastME. Currently
     unused in the main analysis.
2. **Post-inference** (run in sequence, or all at once via
   `sentence_postprocess`):
   - _Tree statistics_: per-sentence tree quality metrics (branch support,
     clock-likeness, stemminess). Used downstream by `beast_generate_xml` to
     select the top-p% trees.
   - _ASTRAL supertree_: combines per-sentence gene trees into a single species
     tree.
   - _Summary_: scores each run's supertree against reference trees (RF, quartet
     similarity).

### Run IDs

Each tree inference run gets a random UUID (e.g. `ba9f2d2a`). Outputs are stored
under `data/trees/per_sentence/<dtype>/<run_id>/`, with a `cfg.json` sidecar
that records all CLI arguments and the git commit SHA. Every invocation produces
a fresh UUID; there is no content-based deduplication. Downstream scripts
(`sentence_stats`, `sentence_astral`, `beast_generate_xml`, etc.) resolve a
`run_id` by globbing the `per_sentence/` directory tree.

## Discrete trees

- **Command:**
  `pixi run sentence_discrete --dtype <name> --dataset fleurs-r --model_id <model>`
- **Requires:** IQ-TREE (or RAxML/pratchet), `extract_embeddings` or `lid` (for
  a trained checkpoint)
- **Inputs:** FLEURS-R audio files (or cached embeddings via
  `--embeddings-cache`)
- **Outputs:** `data/trees/per_sentence/<dtype>[+vote][+pca...]/<run_id>/`
  (FASTA files, inferred trees, `cfg.json`)

Extracts embeddings from FLEURS-R audio per sentence, discretizes them into
binary or multi-state characters, writes FASTA alignments, and runs
maximum-likelihood tree inference. Pass `--method` to select the tree inference
backend (default: `iqtree2`). The required `--dtype` flag sets the output
subdirectory name under `per_sentence/`.

Discretization methods (`--discretization`):

- `ste` (straight-through estimator from a trained LID checkpoint, preferred)
- `step` (Heaviside)
- `quantile`
- `kmeans`

```bash
pixi run sentence_discrete --dtype discrete3 --dataset fleurs-r --model_id NeMo_ambernet
```

## Distance trees

- **Command:**
  `pixi run sentence_distance --dtype <name> --dataset fleurs-r --model_id <model>`
- **Requires:** FastME, `extract_embeddings` or `lid`
- **Inputs:** FLEURS-R audio files (or cached embeddings via
  `--embeddings-cache`)
- **Outputs:** `data/trees/per_sentence/<dtype>+<method>[+pca...]/<run_id>/`
  (distance matrices, inferred trees, `cfg.json`)

Extracts embeddings per sentence and computes pairwise distance matrices between
languages, then infers distance-based trees. The required `--dtype` flag sets
the output subdirectory prefix under `per_sentence/`. Currently unused in the
main analysis (discrete trees are used instead).

Distance metrics (`--metric`):

- `euclidean` (default)
- `cosine`
- `angular`
- `manhattan`
- `sqeuclidean`
- `neuclidean` (normalized Euclidean)

Tree inference methods (`--method`):

- `fastme` (default)
- `nj` (neighbor-joining)
- `upgma`

```bash
pixi run sentence_distance --dtype pdist --dataset fleurs-r --model_id NeMo_ambernet --ebs 1
```

## Postprocessing

- **Command:** `pixi run sentence_postprocess <dirname>`
- **Requires:** all of the below (ASTRAL, R stats, summary)
- **Inputs:** per-sentence trees in `data/trees/per_sentence/<dirname>/`
- **Outputs:** everything produced by stats, ASTRAL, and summary

Orchestrates the full post-inference pipeline by sequentially running tree
statistics, ASTRAL supertree estimation, and summary evaluation. The three steps
can also be run individually:

### Tree statistics

- **Command:** `pixi run sentence_stats <dirname>`
- **Requires:** R (`dplyr`, `ape`), `sentence_discrete` or `sentence_distance`
- **Inputs:** IQ-TREE `.treefile` outputs in
  `data/trees/per_sentence/<dirname>/`
- **Outputs:** `_stats.csv` per run directory (one row per sentence tree)

Computes four quality statistics for each per-sentence tree:

- mean branch support
- root-to-tip clock-likeness (CoV)
- stemminess
- number of tips

Used by `beast_generate_xml` to select the top-p% trees for BEAST input.

### ASTRAL supertree

- **Command:** `pixi run sentence_astral <dirname>`
- **Requires:** ASTRAL-IV (or wASTRAL), `sentence_discrete` or
  `sentence_distance`
- **Inputs:** per-sentence gene trees (`_trees.nex`) in
  `data/trees/per_sentence/<dirname>/`
- **Outputs:** `_trees_astral4.txt` (or `_trees_wastral.txt`) alongside each
  input

Runs ASTRAL-IV species-tree estimation on the per-sentence gene trees. Use
`--method wastral` for the weighted variant. Pass `--include`/`--exclude` to
filter by data split before inference.

```bash
pixi run sentence_astral pdist
```

### Summary

- **Command:** `pixi run sentence_summary <dirname>`
- **Requires:** `sentence_astral`, reference trees in
  `data/trees/references/processed/`
- **Inputs:** ASTRAL summary trees, `cfg.json` per run
- **Outputs:** `data/trees/per_sentence/<dirname>/summary.csv`

Collects metadata from all runs, loads each ASTRAL summary tree, and computes
Robinson-Foulds and quartet-similarity metrics against reference trees. Writes a
ranked summary CSV.

```bash
pixi run sentence_summary pdist
```
