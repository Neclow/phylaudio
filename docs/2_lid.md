# Language Identification

## Embedding extraction (optional)

- **Command:**
  `pixi run extract_embeddings --model-id <model> --dataset fleurs-r`
- **Requires:** `download_fleurs`, `download_models`
- **Inputs:** FLEURS-R audio files, backbone model from `MODEL_ZOO`
- **Outputs:** `data/embeddings/<dataset>/<uuid>/` (`embeddings.pt`,
  `meta.parquet`, `labels.pt`, `taxa.json`, `cfg.json`),
  `data/embeddings/<dataset>/summary.csv`

Pre-caching embeddings avoids re-running the backbone model on every LID
training run, but LID training can also extract embeddings on-the-fly from raw
audio. Extracts frozen per-utterance embeddings from a backbone model (XLS-R,
MMS, Whisper, etc.) over all FLEURS-R languages and splits. No language
filtering is applied at extraction time. Pass `--ckpt` to project through a
trained classifier head. The resulting cache is reused by both LID training and
sentence tree building.

## LID training

- **Command:** `pixi run lid --dataset fleurs-r --model_id <model>`
- **Requires:** `extract_embeddings` (or raw audio on-the-fly)
- **Inputs:** cached embeddings or FLEURS-R audio
- **Outputs:** checkpoints and metrics in `data/eval/`

Trains and evaluates an MLP-based language identification classifier on audio
embeddings. Pass `--ste` to train with straight-through estimator binarization,
which produces discrete-valued embeddings for downstream sentence tree
inference. By default, training metrics are written to a local CSV via
Lightning's `CSVLogger`. To use [Weights & Biases](https://wandb.ai) logging
instead, pass `--project <name>`.

```bash
pixi run lid --dataset fleurs-r --model_id NeMo_ambernet                      # CSV logger
pixi run lid --dataset fleurs-r --model_id NeMo_ambernet --project phylaudio  # wandb
```

## LID summary

- **Command:** `pixi run lid_summary <project>`
- **Requires:** `lid` (finished runs)
- **Inputs:** checkpoints in `data/eval/<project>/`, cached embeddings
- **Outputs:** `data/eval/<project>/summary.csv`

Re-evaluates all finished LID runs on the analysis language subset, producing a
ranked summary CSV sorted by test F1 or accuracy. Loads each checkpoint,
optionally filters by glottocode/min-speakers/gender.

## Speech quality estimation

- **Command:** `pixi run squim fleurs-r`
- **Requires:** `download_fleurs`
- **Inputs:** FLEURS-R audio files
- **Outputs:** `data/metadata/fleurs-r/squim.csv`

Runs torchaudio's SQUIM objective model over the full FLEURS-R dataset to
estimate non-intrusive speech quality metrics (STOI, PESQ, SI-SDR) per
utterance.

## Sentence annotation (optional)

- **Command:** `python -m pipeline.spacy_annotate [dataset]`
- **Requires:** spaCy (`en_core_web_sm`)
- **Inputs:** FLEURS-R transcript TSVs
- **Outputs:** `data/metadata/<dataset>/spacy.csv`

Annotates FLEURS sentences with POS tags, named-entity flags (11 NER types), and
a Heylighen & Dewaele (1999) formality F-score using spaCy's English model. Not
used in the main analysis pipeline but available for exploratory sentence-level
filtering.
