# Plots

All plot scripts live in `plots/` and run in the `viz` environment. Shared
constants (BEAST dirs, palettes, NMF component labels) are in
`plots/_config.py`. Supplementary tables are generated from metadata and
pipeline outputs (no dedicated plot script).

```bash
pixi install -e viz
```

## Figure dependency table

| Figure                                 | Script                              | Pipeline dependency                                                                                   | Data                                                        |
| -------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Fig 1a (acc vs. brsupport)             | `fig1_acc_vs_brsupport.py`          | `extract_embeddings`, `lid`, `lid_summary`, `sentence_discrete`, `sentence_summary`, `sentence_stats` | `discrete3+vote/summary.csv`, `eval/phylaudio2/summary.csv` |
| Fig 1b (NMF STRUCTURE)                 | `fig1_nmf.py`                       | `nmf_structure`                                                                                       | `{beast_run}/nmf/Q_K*.csv`                                  |
| Fig 1c (network)                       | SplitsTree                          | `network_analysis`                                                                                    | `{beast_run}/input_v1.xml`                                  |
| Fig 1d (delta)                         | `fig1_delta.py`                     | `nmf_structure`, `bootstrap_delta`                                                                    | `{beast_run}/nmf/Q_K*.csv`, `{beast_run}/_delta.csv`        |
| Fig 2a (root age)                      | `fig2_root_age.py`                  | BEAST runs                                                                                            | `{beast_run}/input_v1_101.log`                              |
| Fig 2b (speech rates)                  | `fig2_rates.py`                     | BEAST runs                                                                                            | `{beast_run}/input_v1_101.trees`                            |
| Fig 3 (geo regression)                 | `fig3_geo.py`                       | `beast_phylolm`, `fit_gp_surface`                                                                     | `data/phyloregression/`                                     |
| Supp Fig 1 (F1 vs. brsupport)          | `fig1_acc_vs_brsupport.py`          | (same as Fig 1a)                                                                                      | (same as Fig 1a)                                            |
| Supp Fig 2a (PCA)                      | `fig1_pca.py`                       | `extract_embeddings`                                                                                  | XLS-R embeddings                                            |
| Supp Fig 2b (audio quality)            | `fig1_sqa.py`                       | `extract_embeddings`, `squim`                                                                         | XLS-R embeddings, `squim.csv`                               |
| Supp Fig 3 (NMF K selection)           | `fig1_nmf.py`                       | `nmf_structure`                                                                                       | `{beast_run}/nmf/cross_entropy.csv`                         |
| Supp Fig 4 (PHOIBLE regression)        | `fig1_nmf.py`                       | `nmf_brms`                                                                                            | `{beast_run}/brms_phoible/`                                 |
| Supp Fig 5 (cognate root age + rates)  | `fig2_root_age.py`, `fig2_rates.py` | BEAST runs                                                                                            | `iecor/raw.log`, `iecor/prunedtomodern.trees`               |
| Supp Fig 6 (speech vs cognate scatter) | `fig2_rates.py`                     | `prepare_regression_data`                                                                             | `{beast_dir}/metadata_with_inventory.csv`                   |
| Supp Fig 7 (geo regression, cognates)  | `fig3_geo.py`                       | (same as Fig 3)                                                                                       | (same as Fig 3)                                             |
| Supp Fig 8 (BEAST2 sensitivity)        | DensiTree                           | BEAST runs                                                                                            | 2 BEAST runs                                                |
| Supp Table 1 (FLEURS-R metadata)       | -                                   | `download_fleurs_metadata`                                                                            | `metadata/fleurs-r/n_speakers.csv`, `glottolog.csv`         |
| Supp Table 2 (non-IE metadata)         | -                                   | `download_fleurs_metadata`                                                                            | `metadata/fleurs-r/n_speakers.csv`, `glottolog.csv`         |
| Supp Table 3 (model zoo)               | -                                   | -                                                                                                     | `src/models/transformers.py`, `fig1_acc_vs_brsupport.py`    |
| Supp Table 4 (LID performance)         | -                                   | `extract_embeddings`, `lid`, `lid_summary`                                                            | Wandb API data, `eval/phylaudio2/summary.csv`               |
| Supp Table 5 (sentence discrete)       | -                                   | `sentence_discrete`, `sentence_summary`                                                               | `discrete3+vote/summary.csv`                                |
| Supp Table 6 (MRCA priors)             | -                                   | -                                                                                                     | `trees/beast/templates.xml`                                 |
| Supp Table 7 (nested sampling)         | -                                   | `run_beast_ns`                                                                                        | `{beast_run}/ns/results/ns_summary.tsv`                     |

## Commands

Scripts that have `pixi` tasks can be run as follows. All produce PDF output in
`img_v2/`.

```bash
pixi run -e viz fig1_acc_vs_brsupport
pixi run -e viz fig1_nmf
pixi run -e viz fig1_delta
pixi run -e viz fig1_pca
pixi run -e viz fig1_sqa
pixi run -e viz fig2_root_age
pixi run -e viz fig2_rates
pixi run -e viz fig3_geo
```

Fig 1c (network) and Supp Fig 8 (sensitivity) are rendered in SplitsTree and
DensiTree respectively, not from Python scripts.
