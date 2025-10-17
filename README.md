# Phylaudio

## Installation

```bash
pixi install
pixi run post_install
```

## Language identification

You will need to setup a user and project in [Weights & Biases](https://wandb.ai). See the [Quickstart](https://docs.wandb.ai/quickstart/) for more information.

```bash
pixi run language_identification --dataset fleurs-r --model_id NeMo_ambernet --project phylaudio
```

## Sentence-wise distance trees

```bash
pixi run distance_phylo --dataset fleurs-r --model_id NeMo_ambernet
```

## Sentence-wise discrete trees

```bash
pixi run discrete_phylo --dataset fleurs-r --model_id NeMo_ambernet
```
