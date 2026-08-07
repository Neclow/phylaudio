# Post-BEAST Analysis

Downstream analyses on the BEAST posterior: NMF population structure, delta
scores, and phylogenetic regression.

## NMF structure

- **Command:**
  `pixi run nmf_structure <run_id> <subdir> [k_min] [k_max] [n_reps] [--plot]`
- **Requires:** R with LEA (bioconductor-lea)
- **Inputs:** `data/trees/beast/<run_id>/<subdir>/__merged_mapped.fa`
- **Outputs:** `<subdir>/nmf/snmf_results.rds`, `Q_K{k}.csv`,
  `cross_entropy.csv`

Runs sNMF K-sweep on the binary FASTA alignment, selects the optimal K by
minimum cross-entropy, and saves Q-matrices. Pass `--plot` to generate
STRUCTURE-style admixture bar plots.

## NMF-PHOIBLE regression

- **Command:** `pixi run nmf_brms <run_id> <subdir> [K] [dataset]`
- **Requires:** R with brms, `nmf_structure`
- **Inputs:** `<subdir>/nmf/snmf_results.rds`, `languages.json` (committed),
  `phoible.csv` (committed)
- **Outputs:** `<subdir>/brms_phoible/component_{01..K}.rds`,
  `nmf_phoible_brms.csv`

Fits K independent Bayesian linear regressions predicting each NMF component
proportion from PHOIBLE binary phonological features and inventory-size counts.
Reports coefficients whose 95% CI excludes zero.

## Delta scores

- **Command:** `pixi run bootstrap_delta <run_id> <subdir> [n_boot] [seed]`
- **Requires:** R (base)
- **Inputs:** `data/trees/beast/<run_id>/<subdir>/__merged_mapped.fa`
- **Outputs:** `<subdir>/_delta.csv` (language, delta, ci_lo, ci_hi)

Computes per-taxon delta scores (Holland et al. 2002 network-likeness statistic)
from the binary FASTA alignment with bootstrap 95% CIs via column resampling.

## Prepare regression data

- **Command:**
  `pixi run prepare_regression_data <run_id> <subdir> [--version N]`
- **Requires:** BEAST MCC trees, SplitsTree `.stree6` files
- **Inputs:** `languages.json`, `glottolog.csv`, `n_speakers.csv`, `phoible.csv`
  (all committed), MCC and `.stree6` files in BEAST dirs
- **Outputs:** `<beast_dir>/metadata.csv`,
  `<beast_dir>/metadata_with_inventory.csv` (for both speech and cognate)

Joins language coordinates (Glottolog), speaker counts, PHOIBLE inventory sizes,
BEAST branch rate medians, and SplitsTree delta scores into regression metadata
CSVs.

## Phylogenetic regression

- **Command:** `pixi run beast_phylolm <run_id> <subdir> [--model_type <type>]`
- **Requires:** R with brms, cmdstanr, `prepare_regression_data`
- **Inputs:** BEAST MCC trees, `metadata.csv` / `metadata_with_inventory.csv`
- **Outputs:** regression results in `<beast_dir>/phyloregression/`

Runs phylogenetic regressions for all combinations of model type and tree type
(speech and cognate).

Model types (`--model_type`):

- `linear_geo` (linear with geographic covariates)
- `gp_geo` (Gaussian process with geographic covariates)

```bash
pixi run beast_phylolm -- ba9f2d2a 0.05_brsupport_dev_test
```

## GP rate surface

- **Command:** `pixi run fit_gp_surface <run_id> <subdir>`
- **Requires:** Python with GPflow, TensorFlow, `prepare_regression_data`
- **Inputs:** `<beast_dir>/metadata_with_inventory.csv`, Natural Earth
  shapefiles, language polygons GeoJSON
- **Outputs:** `<beast_dir>/gp_cache/gp_grid.npz`, `gp_obs.csv`,
  `land_clipped.geojson`

Fits a GPflow Gaussian Process (Matern-3/2 kernel) over language polygons to
produce a smoothed geographic rate surface for the regression map figure.
