# Documentation

## Environment setup

```bash
pixi install              # default environment (stages 1–4)
pixi run post_install
pixi install -e regression  # phylogenetic regression (R + brms)
pixi install -e gp          # GP regression (TensorFlow, GPflow)
pixi install -e viz         # publication figures (plotly, seaborn)
```

## Pipeline

All commands use [pixi](https://pixi.sh) task definitions from `pixi.toml`. Most
stages run in the `default` environment (`pixi run ...`); stages that need
additional dependencies use a named environment (`pixi run -e <env> ...`).

| Stage                      | Docs                                       | Environment                   |
| -------------------------- | ------------------------------------------ | ----------------------------- |
| 1. Data download           | [1_download.md](1_download.md)             | `default`                     |
| 2. Language identification | [2_lid.md](2_lid.md)                       | `default`                     |
| 3. Sentence trees          | [3_sentence_trees.md](3_sentence_trees.md) | `default`                     |
| 4. BEAST2                  | [4_beast.md](4_beast.md)                   | `default`                     |
| 5. Post-BEAST analysis     | [5_post_beast.md](5_post_beast.md)         | `default`, `regression`, `gp` |
| 6. Plots                   | [6_plots.md](6_plots.md)                   | `viz`                         |

See also the READMEs in `data/` (`eval/`, `metadata/fleurs-r/`, `resources/`,
`trees/`) for dataset and output descriptions.
