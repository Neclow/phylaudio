# Phylaudio

## Installation

### Dependencies

```bash
pixi install
pixi run post_install
```

### Data

```bash
pixi run download_models # download all pre-trained models
pixi run download_fleurs # download FLEURS-R
pixi run download_glottolog # extract lineages from FLEURS-R
pixi run download_reference_trees # extract and process reference trees
```

## Language identification

You will need to setup a user and project in
[Weights & Biases](https://wandb.ai). See the
[Quickstart](https://docs.wandb.ai/quickstart/) for more information.

```bash
pixi run lid --dataset fleurs-r --model_id NeMo_ambernet --project phylaudio
```

## Sentence-wise distance trees

```bash
pixi run sentence_distance --dataset fleurs-r --model_id NeMo_ambernet --ebs 1
```

## Sentence-wise discrete trees

```bash
pixi run sentence_discrete --dataset fleurs-r --model_id NeMo_ambernet
```

## Generating single language-tree estimates from sentence-wise runs (using ASTRAL-IV)

```bash
pixi run sentence_astral pdist
```

## Comparing ASTRAL-IV trees

```bash
pixi run sentence_summary pdist
```

## BEAST

### Standard run

```bash
pixi run beast2 -beagle_SSE -threads 8 -seed 889 data/trees/beast/iecor/fleurs_v5.xml
```

### Prior only

```bash
pixi run beast2 -sampleFromPrior -beagle_SSE -threads 8 -seed 889 data/trees/beast/iecor/prior/fleurs_v5_prior.xml
```

### Getting tree summaries with CCD

```bash
pixi run treeannotator -topology CCD0 ./data/trees/beast/eab44e7f-54cc-4469-87d1-282cc81e02c2/0.25/long_v3_44.trees long_v3_44.CCD.nex
```

## Generating phylogenetic networks

```bash
pixi run network_analysis data/trees/beast/eab44e7f-54cc-4469-87d1-282cc81e02c2/0.25/input.xml
```

## Phylogenetic regression

Install the regression environment:

```bash
pixi install -e regression
```

### Required files

Before running regression or plotting, the following files must be present in `data/trees/beast/`:

| File | Source |
|---|---|
| `input_v12_combined_resampled.mcc` | Already in repo (speech MCC tree) |
| `input_v12_combined_resampled.log` | Already in repo (speech BEAST log) |
| `input_v12_combined_resampled.trees` | Already in repo (speech posterior trees) |
| `IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_mcc.tree` | `pixi run download_iecor_trees` |
| `IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined.log` | `pixi run download_iecor_trees` |
| `IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined.trees` | `pixi run download_iecor_trees` |
| `priors/prior_v12_1.log` | Already in repo (speech prior log) |
| `priors/IECoR_..._combined_PRIOR.log` | `pixi run download_iecor_trees` |

Download the IECoR files:

```bash
pixi run download_iecor_trees
```

### Prepare regression data

Generates metadata CSVs (with and without phoneme inventory) for both speech and cognate trees. Reads MCC trees from `data/trees/beast/`:

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

To install visualization dependencies, run:

```bash
pixi install -e viz
```

### Visualizing audio embeddings

```bash
pixi install -e viz_embeddings_pca
```

Currently, you can visualize embeddings from XLS-R (fine-tuned on VoxLingua 107)
at: <https://neclow.github.io/phylaudio/>

### Download auxiliary data

```bash
pixi run download_geojson        # download language polygon data (Glottography)
```

### Prune IECoR trees (for rate-over-time plots)

```bash
pixi run -e viz prune_iecor_trees  # prune historical taxa from IECoR posterior trees (slow)
```

This produces `IECoR_..._combined_prunedtomodern.trees` in `data/trees/beast/`, required by the rate-over-time plots.

### Publication figures

```bash
pixi run -e viz viz_figure2_rates        # Figure 2 panel B (rate over time with CI bands)
pixi run -e viz viz_figure3_geo          # Figure 3 (regression panels)
pixi run -e viz viz_plot_rates_and_maps  # rate scatter, GP maps, root age, rate-over-time
```
