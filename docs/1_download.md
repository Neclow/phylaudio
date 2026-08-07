# Data Download

All download scripts write to `data/`. Most require
`data/metadata/fleurs-r/languages.json`, which is committed to the repository
along with all other metadata files (`glottolog.csv`, `phoible.csv`,
`n_speakers.csv`, `genders.csv`, etc.). These files do not need to be
regenerated to reproduce the analysis.

## FLEURS-R recordings

- **Command:** `pixi run download_fleurs`
- **Requires:** HuggingFace access to `google/fleurs` and `google/fleurs-r`
- **Inputs:** `languages.json` (committed)
- **Outputs:** `data/datasets/fleurs-r/{lang}/{lang}/audio/{split}/`,
  `data/datasets/fleurs-r/{lang}/{lang}/{split}.tsv`

Downloads audio recordings and transcripts per language and split from the
FLEURS-R HuggingFace dataset. Expect 350,549 recordings (~201 GB).

## Metadata (committed, no download needed)

All metadata files in `data/metadata/fleurs-r/` are committed to the repository.
The scripts below generated them originally and are kept for provenance, but do
not need to be re-run:

- `pixi run download_fleurs_metadata`: speaker populations (Wikimedia API,
  LinguaMeta), gender distributions (from FLEURS TSVs), Glottolog languoid
  metadata. Outputs `languages.json`, `n_speakers.csv`, `genders.csv`,
  `glottolog.csv`.
- `pixi run download_glottolog`: lineage hierarchies from a local Glottolog
  clone. Outputs `glottolog.csv`.
- `pixi run download_phoible`: PHOIBLE phoneme inventories (median across
  inventories per language). Outputs `phoible.csv`.

## Pre-trained audio models

- **Command:** `pixi run download_models`
- **Requires:** HuggingFace access
- **Inputs:** N/A
- **Outputs:** model files in the cache directory

Downloads all pre-trained audio models registered in `src/models/_model_zoo.py`
(XLS-R, MMS, Whisper, ECAPA-TDNN, etc.) by instantiating each extractor.

## Reference phylogenetic trees

- **Command:** `pixi run download_reference_trees`
- **Requires:** `extern/glottolog`
- **Inputs:** `languages.json` (committed)
- **Outputs:** `data/trees/references/raw/`, `data/trees/references/processed/`,
  `data/trees/beast/iecor/` (posteriors, logs, `prunedtomodern.trees`)

Downloads reference phylogenetic trees from five sources (Glottolog, GLED, ASJP,
IE-CoR MCC tree and posterior), prunes each to match the dataset's language set,
and renames leaves to FLEURS keys. IE-CoR posteriors and logs are written to
`data/trees/beast/iecor/`.

## Language polygons

- **Command:** `pixi run download_geojson`
- **Requires:** nothing
- **Inputs:** N/A
- **Outputs:** language polygons GeoJSON, `data/geo/naturalearth/` (shapefiles)

Downloads language-area polygons from the Glottography project and Natural Earth
shapefiles (110m countries, 50m land) for map visualizations.

## Zenodo data

Pre-computed outputs (BEAST2 posteriors, XLS-R embeddings, regression results)
that are too large for git. Download from
[doi:10.5281/zenodo.21838594](https://doi.org/10.5281/zenodo.21838594). See
[ZENODO.md](../ZENODO.md) for the full manifest.

```bash
# TODO: pixi run prepare_zenodo_data
tar -xzf phylaudio_zenodo.tar.gz
```

Unpacks into `data/`.
