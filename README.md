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
pixi run run download_glottolog # extract lineages from FLEURS-R
pixi run run download_reference_trees # extract and process reference trees
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
