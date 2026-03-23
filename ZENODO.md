# External Data (Zenodo)

Files too large for git that must be downloaded separately to reproduce figures
and analyses.

## BEAST2 posterior outputs

Destination: `data/trees/beast/speech/0.01_brsupport/`

| File                             | Size | Used by                        |
| -------------------------------- | ---- | ------------------------------ |
| `input_combined_resampled.trees` | 42M  | fig2_rates, ext_rates_and_maps |
| `input_combined_resampled.log`   | 5.5M | ext_rates_and_maps             |
| `input_combined_resampled.mcc`   | 64K  | prepare_regression_data        |
| `prior_1.log`                    | 6.0M | ext_rates_and_maps             |

## BEAST2 cognate tree (IE-CoR)

Destination: `data/trees/beast/iecor/`

| File                   | Size | Used by            |
| ---------------------- | ---- | ------------------ |
| `prunedtomodern.trees` | 415M | fig2_cognate_rates |
| `raw.log`              | 20M  | ext_rates_and_maps |
| `prior/raw.log`        | 20M  | ext_rates_and_maps |

## NMF results

Destination: `data/trees/beast/speech/0.01_brsupport/nmf/`

| File              | Size | Used by              |
| ----------------- | ---- | -------------------- |
| `sweep_k2_k30.h5` | 73M  | fig1_nmf, fig1_delta |

## Embeddings (XLS-R)

Destination: `data/embeddings/fleurs-r/16fa4383-40db-492c-8420-72488cc60562/`

| File            | Size | Used by            |
| --------------- | ---- | ------------------ |
| `embeddings.pt` | 345M | fig1_pca, fig1_sqa |
| `labels.pt`     | 1.4M | fig1_pca, fig1_sqa |

## LID evaluation

Destination: `data/trees/per_sentence/discrete/`

| File                              | Size     | Used by               |
| --------------------------------- | -------- | --------------------- |
| `{run_id}/_stats.csv` (per model) | ~4K each | fig1_acc_vs_brsupport |

Note: `data/eval/summary.json` and
`data/trees/per_sentence/discrete/summary.csv` are committed to git.

## Phyloregression coefficients (brms)

Archive: `with_inventory.tar.gz` Destination: unpack into
`data/phyloregression/`

Contains brms posterior samples for fig3_geo:

- `coef_samples_linear_geo_*.csv`
- `variance_samples_linear_geo_*.csv`
- `phylolm_linear_geo_*.csv`
- `variance_samples_gp_geo_*.csv`

## Downloadable externally (not Zenodo)

These can be fetched via provided scripts:

| File                                               | Script                         | Source                      |
| -------------------------------------------------- | ------------------------------ | --------------------------- |
| `data/metadata/fleurs-r/language_polygons.geojson` | `pipeline/download_geojson.sh` | Glottography/asher2007world |
