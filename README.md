# Phylaudio

## Installation

### Dependencies

> [!WARNING] Embedding extraction and LID training were run on a CUDA 123.2 GPU;
> higher CUDA versions are likely to work but are untested. To use a different
> CUDA version, edit the `cuda` key under `[system-requirements]` in
> `pixi.toml`. On CPU-only machines, running
> `CONDA_OVERRIDE_CUDA=13.2 pixi install` will pretend CUDA 13.2 is available
> and install the GPU-pinned dependencies; in that case, pass `--device cpu` to
> any script that accepts a device flag.

```bash
pixi install # Or CONDA_OVERRIDE_CUDA=13.2 pixi install
pixi run post_install
```

### Data

Download pipeline inputs:

```bash
pixi run download_models           # download pre-trained audio models
pixi run download_fleurs           # download FLEURS-R audio dataset
pixi run download_glottolog        # extract lineages from FLEURS-R
pixi run download_reference_trees  # extract and process reference trees
pixi run download_geojson          # download language polygon data (Glottography)
```

Download external data from Zenodo (see `ZENODO.md` for full manifest):

```bash
# From the repo root:
tar -xzf phylaudio_zenodo.tar.gz
```

This unpacks BEAST2 posteriors, XLS-R embeddings, and regression outputs into
`data/`.

## Language identification

By default, training metrics are written to a local CSV file via Lightning's
`CSVLogger`. To use [Weights & Biases](https://wandb.ai) instead, set up an
account (see the [Quickstart](https://docs.wandb.ai/quickstart/)) and pass
`--project <name>`.

```bash
# CSV logger (default, no account required)
pixi run lid --dataset fleurs-r --model_id NeMo_ambernet

# wandb logger
pixi run lid --dataset fleurs-r --model_id NeMo_ambernet --project phylaudio
```

## Sentence-wise trees

### Distance trees

```bash
pixi run sentence_distance --dataset fleurs-r --model_id NeMo_ambernet --ebs 1
```

### Discrete trees

```bash
pixi run sentence_discrete --dataset fleurs-r --model_id NeMo_ambernet
```

### ASTRAL species tree estimation

```bash
pixi run sentence_astral pdist
pixi run sentence_summary pdist
```

## BEAST2

### Standard run

```bash
pixi run beast2 -beagle_SSE -threads 8 -seed 889 data/trees/beast/speech/0.01_brsupport/input.xml
```

### Prior only

```bash
pixi run beast2 -sampleFromPrior -beagle_SSE -threads 8 -seed 889 data/trees/beast/speech/0.01_brsupport/prior.xml
```

### Combining runs

```bash
scripts/beast_combine_logs.sh data/trees/beast/speech/0.01_brsupport input_v12
```

### Tree summaries (CCD)

```bash
pixi run treeannotator -topology CCD0 data/trees/beast/speech/0.01_brsupport/input_combined_resampled.trees input_combined_resampled.ccd0
```

### Phylogenetic networks

```bash
pixi run network_analysis data/trees/beast/speech/0.01_brsupport/input.xml
```

## Phylogenetic regression

Install the regression environment:

```bash
pixi install -e regression
```

### Required files

Before running regression or plotting, the following files must be present:

| File                                                                    | Source                                                    |
| ----------------------------------------------------------------------- | --------------------------------------------------------- |
| `data/trees/beast/speech/0.01_brsupport/input_combined_resampled.mcc`   | Zenodo (speech MCC tree)                                  |
| `data/trees/beast/speech/0.01_brsupport/input_combined_resampled.log`   | Zenodo (speech BEAST log)                                 |
| `data/trees/beast/speech/0.01_brsupport/input_combined_resampled.trees` | Zenodo (speech posterior trees)                           |
| `data/trees/beast/speech/0.01_brsupport/prior_1.log`                    | Zenodo (speech prior log)                                 |
| `data/trees/references/raw/iecor.nex`                                   | `pixi run download_reference_trees` (IECoR MCC tree)      |
| `data/trees/beast/iecor/raw.trees`                                      | `pixi run download_reference_trees` (IECoR posterior)     |
| `data/trees/beast/iecor/raw.log`                                        | `pixi run download_reference_trees` (IECoR posterior log) |
| `data/trees/beast/iecor/prior/raw.log`                                  | `pixi run download_reference_trees` (IECoR prior log)     |
| `data/trees/beast/iecor/prunedtomodern.trees`                           | `pixi run download_reference_trees` (auto-pruned)         |

### Prepare regression data

Generates metadata CSVs (with and without phoneme inventory) for both speech and
cognate trees. Reads MCC trees from `data/trees/beast/`:

```bash
pixi run -e regression prepare_regression_data
```

This writes 4 files to `data/phyloregression/`.

### Linear regression (brms)

```bash
pixi run -e regression beast_phylolm -- --model_type linear_geo --tree input_v12_combined_resampled --variant with_inventory
pixi run -e regression beast_phylolm -- --model_type linear_geo --tree heggarty2024_raw --variant with_inventory
```

### GP regression (cmdstanr)

```bash
pixi run -e regression beast_phylolm -- --model_type gp_geo --tree input_v12_combined_resampled --variant with_inventory
pixi run -e regression beast_phylolm -- --model_type gp_geo --tree heggarty2024_raw --variant with_inventory
```

Results are written to `data/phyloregression/<variant>/`.

## Plots

Install visualization dependencies:

```bash
pixi install -e viz
```

### Figure dependency table

Each figure depends on one or more pipeline scripts that must be run first to
produce the intermediate data. The plot scripts in `plots/` read that data and
generate publication-ready figures.

| Figure                                | Plot script                        | Pipeline dependency        | Data                                          |
| ------------------------------------- | ---------------------------------- | -------------------------- | --------------------------------------------- |
| Fig 1a                                | —                                  | —                          | —                                             |
| Fig 1b (NMF STRUCTURE)                | `plots/fig1_nmf.py`                | `pipeline/nmf_structure.R` | `{run}/nmf/Q_K*.csv`                          |
| Fig 1d (delta)                        | `plots/fig1_delta.py`              | `pipeline/nmf_structure.R` | `{run}/nmf/Q_K*.csv`, `{run}/_delta.csv`      |
| Supp Fig 2 (NMF K selection)          | `plots/fig1_nmf.py`                | `pipeline/nmf_structure.R` | `{run}/nmf/cross_entropy.csv`                 |
| Supp Fig 3 (PHOIBLE regression)       | `plots/fig1_nmf.py`                | `pipeline/nmf_brms.R`      | `{run}/brms_phoible/`                         |
| Ext (PCA)                             | `plots/fig1_pca.py`                |                            | `{run}/nmf/Q_K*.csv`                          |
| Ext (audio quality)                   | `plots/fig1_sqa.py`                |                            |                                               |
| Fig 2a (root age)                     | `plots/fig2_heights.py`            | `run_beast.sh`             | `{run}/input_v1_101.log`                      |
| Fig 2b (speech rates)                 | `plots/fig2_rates.py`              | `run_beast.sh`             | `{run}/input_v1_101.trees`                    |
| Supp Fig 4 (cognate root age + rates) | `fig2_heights.py`, `fig2_rates.py` | `run_beast.sh`             | `iecor/raw.log`, `iecor/prunedtomodern.trees` |
| Fig 3 (geo regression)                | `plots/fig3_geo.py`                | `pipeline/beast_phylolm.R` | `data/phyloregression/`                       |
| Ext (rates & maps)                    | `plots/ext_rates_and_maps.py`      | `pipeline/beast_phylolm.R` | `data/phyloregression/`                       |

### Publication figures

```bash
# Figure 1
pixi run -e viz fig1_acc_vs_brsupport  # Panel A: LID accuracy vs. bootstrap support
pixi run -e viz fig1_nmf               # Panel B: sNMF structure plot
pixi run -e viz fig1_delta             # Panel D: per-language delta scores
pixi run -e viz fig1_pca               # Extended: PCA of XLS-R embeddings
pixi run -e viz fig1_sqa               # Extended: silhouette vs. SI-SDR + correlation

# Figures 2–3
pixi run -e viz fig2_heights           # Figure 2 panel A: root age distribution
pixi run -e viz fig2_rates             # Figure 2 panel B + Supp Fig 4: speech & cognate rates
pixi run -e viz fig3_geo               # Figure 3: regression panels

# Extended
pixi run -e viz ext_rates_and_maps     # rate scatter, GP maps, root age, rate-over-time
```

### Compute paper stats

```bash
pixi run python -m src.tasks.phylo.compute_paper_stats
```
