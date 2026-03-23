#!/usr/bin/env bash
# Combine BEAST2 CoupledMCMC runs: trim run 1 to first 100M, combine all runs,
# and resample to 10% of combined output.
#
# Usage:
#   scripts/combine_v12_logs.sh <run_dir> <prefix> [-r resample_interval]
#
# Example:
#   scripts/combine_v12_logs.sh data/trees/beast/speech/0.01_brsupport input_v12
#
# Run 1 (1201): 0–250M, logEvery=10k → 25001 samples
# Run 2 (1202): 0–100M, logEvery=10k → 10001 samples
#
# After trimming run 1 to 100M, both runs have 10001 samples each.
# Combined (with 0% burnin since we handle burnin externally): ~20001 samples.
# 10% resample: ~2000 samples.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <run_dir> <prefix> [-r resample_interval]" >&2
    echo "  e.g.: $0 data/trees/beast/speech/0.01_brsupport input_v12" >&2
    exit 1
fi

DIR="$(cd "$1" && pwd)"
PREFIX="$2"
shift 2

N_FILES=4
RESAMPLE=""
while getopts "r:" opt; do
    case $opt in
        r) RESAMPLE="$OPTARG" ;;
        *) echo "Usage: $0 <run_dir> <prefix> [-r resample_interval]" >&2; exit 1 ;;
    esac
done
RESAMPLE="${RESAMPLE:-$((10000 * N_FILES))}"

BEAST_BIN="$(cd "$(dirname "$0")/../extern/beast2/bin" && pwd)"
LOGCOMBINER="$BEAST_BIN/logcombiner"
OUTDIR="$DIR/combined_${PREFIX##*_}"

mkdir -p "$OUTDIR"

# --- Step 1: Trim run 1 to first 100M ---
# Run 1 has 250M states logged every 10k = 25001 lines of data.
# We want 0–100M = 10001 lines of data.
# Simpler approach: use head to take the header + first 10001 data lines.

echo "=== Step 1: Trimming run 1 (.log) to first 100M states ==="
n_header=$(grep -c -E "^(#|Sample)" "$DIR/${PREFIX}_1201.log")
n_data=10001
head -n $((n_header + n_data)) "$DIR/${PREFIX}_1201.log" > "$OUTDIR/${PREFIX}_1201_trimmed.log"
echo "  Trimmed log: $(wc -l < "$OUTDIR/${PREFIX}_1201_trimmed.log") lines (${n_header} header + ${n_data} data)"

echo "=== Step 1: Trimming run 1 (.trees) to first 100M states ==="
{
    awk '
        /^tree /{found=1}
        !found{print; next}
        found{
            match($0, /STATE_([0-9]+)/, m)
            state = m[1] + 0
            if (state <= 100000000) print
            else {print "End;"; exit}
        }
    ' "$DIR/${PREFIX}_1201.trees"
} > "$OUTDIR/${PREFIX}_1201_trimmed.trees"
n_trees=$(grep -c "^tree " "$OUTDIR/${PREFIX}_1201_trimmed.trees")
echo "  Trimmed trees: ${n_trees} trees"

# --- Step 2: Combine runs ---
echo ""
echo "=== Step 2: Combining trimmed run 1 + runs 2-${N_FILES} (.log) ==="
LOG_ARGS=(-log "$OUTDIR/${PREFIX}_1201_trimmed.log")
for i in $(seq 1202 $((1200 + N_FILES))); do
    LOG_ARGS+=(-log "$DIR/${PREFIX}_${i}.log")
done
"$LOGCOMBINER" "${LOG_ARGS[@]}" -o "$OUTDIR/${PREFIX}_combined.log"
echo "  Combined log: $(grep -c "^[0-9]" "$OUTDIR/${PREFIX}_combined.log") data lines"

echo ""
echo "=== Step 2: Combining trimmed run 1 + runs 2-${N_FILES} (.trees) ==="
TREE_ARGS=(-log "$OUTDIR/${PREFIX}_1201_trimmed.trees")
for i in $(seq 1202 $((1200 + N_FILES))); do
    TREE_ARGS+=(-log "$DIR/${PREFIX}_${i}.trees")
done
"$LOGCOMBINER" "${TREE_ARGS[@]}" -o "$OUTDIR/${PREFIX}_combined.trees"
n_combined_trees=$(grep -c "^tree " "$OUTDIR/${PREFIX}_combined.trees")
echo "  Combined trees: ${n_combined_trees} trees"

# --- Step 3: Resample + renumber ---
# Default: resample every 10k * N_FILES (keeps ~1/N_FILES of combined samples).
# After resampling, renumber states via awk: divide by N_FILES so output spans
# 0, 10k, 20k, ..., ~100M instead of 0, 40k, 80k, ..., ~400M.

echo ""
echo "=== Step 3: Resampling combined files (resample=${RESAMPLE}) ==="
"$LOGCOMBINER" \
    -log "$OUTDIR/${PREFIX}_combined.log" \
    -o "$OUTDIR/${PREFIX}_combined_resampled.log" \
    -b 0 \
    -resample "$RESAMPLE"
awk -v d="$N_FILES" 'BEGIN{OFS="\t"} /^[0-9]/{$1=$1/d} {print}' \
    "$OUTDIR/${PREFIX}_combined_resampled.log" > "$OUTDIR/${PREFIX}_combined_resampled.log.tmp" \
    && mv "$OUTDIR/${PREFIX}_combined_resampled.log.tmp" "$OUTDIR/${PREFIX}_combined_resampled.log"
n_resampled_log=$(grep -c "^[0-9]" "$OUTDIR/${PREFIX}_combined_resampled.log")
echo "  Resampled log: ${n_resampled_log} data lines"

"$LOGCOMBINER" \
    -log "$OUTDIR/${PREFIX}_combined.trees" \
    -o "$OUTDIR/${PREFIX}_combined_resampled.trees" \
    -b 0 \
    -resample "$RESAMPLE"
awk -v d="$N_FILES" 'match($0, /^tree STATE_([0-9]+)/, a){
    sub(/STATE_[0-9]+/, "STATE_" (a[1]/d))
} {print}' \
    "$OUTDIR/${PREFIX}_combined_resampled.trees" > "$OUTDIR/${PREFIX}_combined_resampled.trees.tmp" \
    && mv "$OUTDIR/${PREFIX}_combined_resampled.trees.tmp" "$OUTDIR/${PREFIX}_combined_resampled.trees"
n_resampled_trees=$(grep -c "^tree " "$OUTDIR/${PREFIX}_combined_resampled.trees")
echo "  Resampled trees: ${n_resampled_trees} trees"

echo ""
echo "=== Done ==="
echo "Output directory: $OUTDIR"
echo "Files:"
ls -lhS "$OUTDIR"
