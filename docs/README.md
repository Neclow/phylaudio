# Pipeline

All commands use [pixi](https://pixi.sh) task definitions from `pixi.toml`.

1. [Data Download](1_download.md) — models, FLEURS-R, reference trees, Zenodo
2. [Language Identification](2_lid.md) — LID training and evaluation
3. [Sentence Trees](3_sentence_trees.md) — distance, discrete, and ASTRAL
   supertrees
4. [BEAST2](4_beast.md) — Bayesian phylogenetic inference
5. [Post-BEAST Analysis](5_post_beast.md) — regression, NMF
6. [Plots](6_plots.md) — publication figures and dependencies
