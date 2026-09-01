# Extras

Features available in the codebase but not used in the main analysis pipeline.

## Distance trees

The `sentence_distance` pipeline infers per-sentence trees from pairwise
embedding distances instead of discrete characters. See
[3_sentence_trees.md](3_sentence_trees.md) § Distance trees for full usage.

- **Tree methods:** FastME (default), neighbor-joining, UPGMA
- **Metrics:** euclidean, cosine, angular, manhattan, normalized euclidean,
  squared euclidean
- **Soft-DTW:** pass `--soft-dtw` to use Soft Dynamic Time Warping on
  frame-level embeddings instead of mean-pooled distances

## Alternative tree inference backends

`sentence_discrete` supports backends besides IQ-TREE via `--method`:

- `raxmlng` — RAxML-NG maximum-likelihood inference
- `pratchet` — phangorn parsimony (Fitch + ACCTRAN branch lengths)

## Decomposition alternatives

`sentence_discrete` and `sentence_distance` accept `--decomposition` to reduce
embedding dimensionality before discretization or distance computation:

- `pca` (default when `--n-components` is set) — torch or sklearn PCA
- `kpca` — kernel PCA (sklearn)
- `ica` — FastICA (sklearn)

## Sentence annotation (spaCy)

`pixi run spacy-annotate` runs spaCy English NER and POS tagging on FLEURS
transcripts. Outputs `data/metadata/{dataset}/spacy.csv` with:

- Per-sentence proper noun and named entity flags (PERSON, GPE, ORG, …)
- POS token counts
- Heylighen & Dewaele formality F-score

The `src/data/nlp` module can then filter sentences by these annotations (e.g.
remove sentences with high proper-noun density).

## ESPnet placeholder

`src/models/espnet_.py` contains a stub `EspnetFeatureExtractor` class for the
XEUS model. It is not registered in `MODEL_ZOO`.
