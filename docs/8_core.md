# Core Library (`src/`)

The `src` directory contains non-user-facing functions used by the `pipeline/`
scripts. It is organised into three subpackages.

## `src/models/` — Feature extractors

All extractors implement the `BaseFeatureExtractor` interface defined in
`_base.py`. The `MODEL_ZOO` registry in `_model_zoo.py` maps model ID strings to
extractor/processor classes so that pipeline scripts can instantiate any backend
by name.

| Module         | Backend                                                              |
| -------------- | -------------------------------------------------------------------- |
| `transformers` | HuggingFace models (XLS-R, MMS, HuBERT): mean-pooled hidden states   |
| `whisper`      | OpenAI Whisper: encoder-decoder embeddings from log-mel spectrograms |
| `speechbrain`  | SpeechBrain ECAPA-TDNN (VoxLingua107)                                |
| `nemo`         | NVIDIA NeMo AmberNet speaker encoder                                 |
| `opensmile`    | openSMILE handcrafted acoustic features (eGeMAPSv02)                 |
| `baseline`     | CNN6 / CNN10 classification baselines                                |
| `audio`        | Log-mel spectrogram processor (shared by whisper/baseline)           |
| `embedding`    | Pass-through extractor for pre-cached embeddings                     |

## `src/data/` — Datasets and filtering

| Module       | Description                                                                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets`   | `AudioDataset` (per-utterance), `FleursParallelDataset` (per-sentence grouped by language), `EmbeddingDataset` (pre-cached), `load_dataset()` |
| `glottolog`  | Language filtering by glottocode, min speakers, gender; family mapping                                                                        |
| `speakerpop` | Speaker population downloads (Wikimedia API, LinguaMeta)                                                                                      |
| `nlp`        | Sentence-level NER and POS-based filtering (spaCy)                                                                                            |
| `transforms` | Audio transforms: VAD, Trim, Pad                                                                                                              |

## `src/tasks/` — Pipeline logic

### `src/tasks/feature_extraction/`

| Module            | Description                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `base`            | Per-sentence loop, model/dataset preparation, classifier loading, decomposer fitting                                           |
| `_decomposition`  | Dimensionality reduction: PCA, ICA, kernel PCA                                                                                 |
| `_discretization` | Embedding-to-character discretization: STE, step, quantile, k-means                                                            |
| `_distance`       | Pairwise distance metrics: euclidean, cosine, angular, manhattan, normalized euclidean, squared euclidean; batched computation |

### `src/tasks/language_identification/`

| Module       | Description                                                                      |
| ------------ | -------------------------------------------------------------------------------- |
| `classifier` | MLP architecture with optional STE binarization, `LightningMLP` training wrapper |
| `train`      | End-to-end training/evaluation loop, wandb/CSV logging, CLI argument parsing     |

### `src/tasks/phylo/`

| Module       | Description                                                                    |
| ------------ | ------------------------------------------------------------------------------ |
| `beast`      | BEAST run directory resolution, MCC tree discovery                             |
| `fasta`      | FASTA merging (concatenation, majority vote), BEAST XML injection/extraction   |
| `metrics`    | Tree comparison: Robinson-Foulds (normalized, generalized), quartet similarity |
| `newick`     | Newick leaf-label renaming (FLEURS keys to reference names)                    |
| `nexus`      | PhyloWriter classes for writing Nexus tree files from discrete/distance data   |
| `splitstree` | Delta score extraction from SplitsTree `.stree6` files                         |
| `tree`       | Tree building wrappers: IQ-TREE, FastME, RAxML-NG, phangorn pratchet           |

### Other modules

| Module   | Description                                            |
| -------- | ------------------------------------------------------ |
| `common` | Shared CLI argument parsing, model/dataset preparation |
| `plot`   | Matplotlib axis formatting helpers                     |
