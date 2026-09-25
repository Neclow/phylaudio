# External Data (Zenodo)

Files too large for git that must be downloaded separately to reproduce figures
and analyses. They are archived as a single tarball
(`phylaudio_zenodo_v2.tar.gz`, ~28 GB) at
[doi:10.5281/zenodo.19187043](https://doi.org/10.5281/zenodo.19187043). To
download, verify and unpack it into the repository root:

```bash
pixi run download_zenodo
```

All paths below are relative to the repository root.

## XLS-R embeddings

`data/embeddings/fleurs-r/67c9af47-6177-4d06-bcc5-7c64b43e4b06/`

- `embeddings.pt`: XLS-R embeddings of all 350,180 FLEURS-R recordings (102
  languages).
- `labels.pt`: language label of each embedding.
- `meta.parquet`: per-recording metadata (language label, data split, sentence
  identifier).
- `taxa.json`: mapping from label indices to FLEURS language codes.
- `cfg.json`: configuration of the embedding extraction run.

## BEAST2 speech posterior (binary covarion)

`data/trees/beast/ba9f2d2a-27f3-4100-a1c0-43f8fe1c39fc/0.05_brsupport_dev_test/`

- `input_v2.xml`: BEAST2 input for the main analysis.
- `prior_v2.xml`: the same analysis, sampling from the prior only.
- `input_v2_20{1-4}.*`: logs, trees and final states of the four independent MC³
  runs.
- `chain{1-3}input_v2_20{1-3}.*`: outputs of the heated MC³ chains.
- `prior_v2_1.*`: outputs of the prior-only run.
- `metadata_with_inventory.csv`: per-language metadata, including phoneme
  inventory sizes.
- `combined_v2/`: the four runs combined after burn-in, the resampled posterior
  and its MCC tree.
- `ns_v2/`: nested sampling runs for the model comparison.
- `nmf/`: sparse NMF input and outputs, including the cross-entropy sweep over K
  = 2–30.
- `brms_phoible/`: Bayesian ridge regressions of NMF cluster proportions on
  PHOIBLE features.
- `phyloregression/`: phylogenetic regressions of evolutionary rates.
- `gp_cache/`: cached Gaussian process fits for the rate maps.
- `input_v2b.xml`, `input_v2b_1.*`: the same analysis with standard single-chain
  MCMC instead of MC³; stopped after a few samples and not used in the paper.

## BEAST2 cognate tree (IE-CoR)

`data/trees/beast/iecor/`

- Full directory: posterior trees, logs, priors, phylogenetic regressions,
  Gaussian process cache, metadata and XML inputs.

## Per-sentence IQ-TREE outputs

`data/trees/per_sentence/discrete3+vote/`

- `{run_id}/`: one folder per model run (64 runs), each with ~2008 per-sentence
  IQ-TREE outputs (`.fa`, `.treefile`, `.contree`, `.iqtree`, `.log`,
  `.splits.nex`, `.bionj`, `.mldist`, `.model.gz`, `.parstree`). Excludes
  `.ufboot` and `.ckp.gz`.
- `{run_id}/_stats.csv`: per-sentence tree statistics, including mean bootstrap
  support.
- `summary.csv`: one row per model run.

## LID evaluation checkpoints

`data/eval/phylaudio2/`

- One folder per trained LID classifier (80 in total).
- `summary.csv`: test accuracy and F1 score of each classifier.

## Metadata

`data/metadata/fleurs-r/`

- All files except `opensmile.csv`.

## Not in the deposit

- `data/metadata/fleurs-r/language_polygons.geojson`: language-area polygons
  from Glottography (asher2007world). Fetch with `pixi run download_geojson`.
