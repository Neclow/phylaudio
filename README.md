# Phylaudio

## Installation

### Dependencies

```bash
pixi install
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

You will need to setup a user and project in
[Weights & Biases](https://wandb.ai). See the
[Quickstart](https://docs.wandb.ai/quickstart/) for more information.

```bash
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

### Publication figures

```bash
# Figure 1
pixi run -e viz fig1_acc_vs_brsupport  # Panel A: LID accuracy vs. bootstrap support
pixi run -e viz fig1_nmf               # Panel B: NMF structure plot
pixi run -e viz fig1_delta             # Panel D: per-language delta scores
pixi run -e viz fig1_pca               # Extended: PCA of XLS-R embeddings
pixi run -e viz fig1_sqa               # Extended: silhouette vs. SI-SDR + correlation

# Figures 2–3
pixi run -e viz fig2_rates             # Figure 2 panel B: speech rate over time
pixi run -e viz fig2_rates_cognate     # Cognate rate over time
pixi run -e viz fig3_geo              # Figure 3: regression panels

# Extended
pixi run -e viz ext_rates_and_maps     # rate scatter, GP maps, root age, rate-over-time
```

### Compute paper stats

```bash
pixi run python -m src.tasks.phylo.compute_paper_stats
```
