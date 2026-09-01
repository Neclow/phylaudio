# External Data (Zenodo)

Files too large for git that must be downloaded separately to reproduce figures
and analyses.

## XLS-R embeddings

Destination: `data/embeddings/fleurs-r/67c9af47-6177-4d06-bcc5-7c64b43e4b06/`

| File            | Used by            |
| --------------- | ------------------ |
| `embeddings.pt` | fig1_pca, fig1_sqa |
| `labels.pt`     | fig1_pca, fig1_sqa |
| `cfg.json`      | fig1_pca_plotly    |
| `taxa.json`     | supplementary_v2   |
| `meta.parquet`  | supplementary_v2   |

## BEAST2 speech posterior (contraband BM)

Destination:
`data/trees/beast/ba9f2d2a-27f3-4100-a1c0-43f8fe1c39fc/0.05_brsupport_dev_test/`

### MCMC chains (input_v2)

| Pattern                        | Used by              |
| ------------------------------ | -------------------- |
| `input_v2_20{1-4}.*`           | combined_v2          |
| `chain{1-3}input_v2_20{1-3}.*` | combined_v2          |
| `input_v2b*`                   | combined_v2          |
| `input_v2.xml`, `prior_v2.xml` | reproducibility      |
| `prior_v2_1.*`                 | fig2_root_age        |
| `metadata_with_inventory.csv`  | fig2_rates, fig3_geo |

### Subdirectories

| Directory          | Used by                                   |
| ------------------ | ----------------------------------------- |
| `combined_v2/`     | fig2_densitree, fig2_rates, fig2_root_age |
| `ns_v2/`           | nested sampling model comparison          |
| `nmf/`             | fig1_nmf, fig1_delta                      |
| `brms_phoible/`    | fig1_nmf                                  |
| `phyloregression/` | fig3_geo                                  |
| `gp_cache/`        | fig3_geo                                  |

## BEAST2 cognate tree (IE-CoR)

Destination: `data/trees/beast/iecor/`

Full directory including posterior trees, logs, priors, phyloregression,
gp_cache, metadata, and XML inputs.

## Per-sentence IQ-TREE outputs

Destination: `data/trees/per_sentence/discrete3+vote/`

64 model runs, each with ~2008 per-sentence IQ-TREE outputs (`.fa`, `.treefile`,
`.contree`, `.iqtree`, `.log`, `.splits.nex`, `.bionj`, `.mldist`, `.model.gz`,
`.parstree`). Excludes `.ufboot` and `.ckp.gz`.

| File                            | Used by               |
| ------------------------------- | --------------------- |
| `summary.csv`                   | fig1_acc_vs_brsupport |
| `{run_id}/_stats.csv` (per run) | fig1_acc_vs_brsupport |

## LID evaluation checkpoints

Destination: `data/eval/phylaudio2/`

80 trained LID model checkpoints plus `summary.csv`.

## Metadata

Destination: `data/metadata/fleurs-r/`

All files except `opensmile.csv`.

## Downloadable externally (not Zenodo)

These can be fetched via provided scripts:

| File                                               | Script                         | Source                      |
| -------------------------------------------------- | ------------------------------ | --------------------------- |
| `data/metadata/fleurs-r/language_polygons.geojson` | `pipeline/download_geojson.sh` | Glottography/asher2007world |
