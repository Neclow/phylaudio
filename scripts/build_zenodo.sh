#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/phylaudio_zenodo_v2.tar.gz"

# --- Path constants ---
EMBEDDING_DIR="data/embeddings/fleurs-r/67c9af47-6177-4d06-bcc5-7c64b43e4b06"
SPEECH_BEAST_DIR="data/trees/beast/ba9f2d2a-27f3-4100-a1c0-43f8fe1c39fc/0.05_brsupport_dev_test"
IECOR_DIR="data/trees/beast/iecor"
PER_SENTENCE_DIR="data/trees/per_sentence/discrete3+vote"
EVAL_DIR="data/eval/phylaudio2"
METADATA_DIR="data/metadata/fleurs-r"

FILELIST="$(mktemp)"
trap 'rm -f "$FILELIST"' EXIT

cd "$ROOT"

# --- 1. XLS-R embeddings ---
find "$EMBEDDING_DIR" -type f >> "$FILELIST"

# --- 2. Speech BEAST ---
# Top-level v2 files (chains, logs, trees, states, permeaning, mcc, xml)
find "$SPEECH_BEAST_DIR" -maxdepth 1 -type f \( \
    -name 'input_v2*' -o \
    -name 'chain*input_v2*' -o \
    -name 'prior_v2*' \
\) >> "$FILELIST"
echo "$SPEECH_BEAST_DIR/metadata_with_inventory.csv" >> "$FILELIST"

# Subdirectories
find "$SPEECH_BEAST_DIR/combined_v2" -type f >> "$FILELIST"
find "$SPEECH_BEAST_DIR/ns_v2" -type f >> "$FILELIST"
find "$SPEECH_BEAST_DIR/nmf" -type f >> "$FILELIST"
find "$SPEECH_BEAST_DIR/brms_phoible" -type f >> "$FILELIST"
find "$SPEECH_BEAST_DIR/phyloregression" -type f >> "$FILELIST"
find "$SPEECH_BEAST_DIR/gp_cache" -type f >> "$FILELIST"

# --- 3. IECoR (full) ---
find "$IECOR_DIR" -type f >> "$FILELIST"

# --- 4. Per-sentence (64 runs, excluding .ufboot and .ckp.gz) ---
echo "$PER_SENTENCE_DIR/summary.csv" >> "$FILELIST"
find "$PER_SENTENCE_DIR" -mindepth 2 -type f \
    ! -name '*.ufboot' ! -name '*.ckp.gz' >> "$FILELIST"

# --- 5. Eval ---
find "$EVAL_DIR" -type f >> "$FILELIST"

# --- 6. Metadata (minus opensmile.csv) ---
find "$METADATA_DIR" -type f ! -name 'opensmile.csv' >> "$FILELIST"

# --- Summary ---
N=$(wc -l < "$FILELIST")
echo "Packing $N files into $OUT ..."
echo ""

tar czf "$OUT" -T "$FILELIST"

SIZE=$(du -h "$OUT" | cut -f1)
echo "Done: $OUT ($SIZE, $N files)"
