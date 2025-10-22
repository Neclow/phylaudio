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

You will need to setup a user and project in [Weights & Biases](https://wandb.ai). See the [Quickstart](https://docs.wandb.ai/quickstart/) for more information.

```bash
pixi run language_identification --dataset fleurs-r --model_id NeMo_ambernet --project phylaudio
```

## Sentence-wise distance trees

```bash
pixi run distance_phylo --dataset fleurs-r --model_id NeMo_ambernet --ebs 1
```

## Sentence-wise discrete trees

```bash
pixi run discrete_phylo --dataset fleurs-r --model_id NeMo_ambernet
```

## Generating single language-tree estimates from sentence-wise runs

```bash
pixi run sentence_astral pdist
```
