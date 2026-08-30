#!/bin/bash
# Combine BEAST2 MCMC runs using LogCombiner.
#
# Usage:
#   run_beast_combine.sh [-b burnin_pct] [-n samples_per_run] <uuid> <size> <version>
#
# Examples:
#   run_beast_combine.sh ba9 0.05_brsupport_dev_test 2 -b 10 -n 2500
#   run_beast_combine.sh dd2 0.01 12

set -euo pipefail

BURNIN=10
N_PER_RUN=""
OVERWRITE=false

usage() {
    echo "Usage: $(basename "$0") [-o] [-b burnin_pct] [-n samples_per_run] <uuid> <size> <version>" >&2
    exit 1
}

while [[ "${1:-}" == -* ]]; do
    case "$1" in
        -b) BURNIN="$2"; shift 2 ;;
        -n) N_PER_RUN="$2"; shift 2 ;;
        -o) OVERWRITE=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[[ $# -lt 3 ]] && usage

UUID_PATTERN=$1
SIZE=$2
VERSION=$3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAST_DIR="$(dirname "$SCRIPT_DIR")"
TREES_DIR="$BEAST_DIR/data/trees/beast"
LOGCOMBINER="$BEAST_DIR/extern/beast2/bin/logcombiner"

shopt -s nullglob
MATCHES=("$TREES_DIR"/"$UUID_PATTERN"*)
shopt -u nullglob

if [[ ${#MATCHES[@]} -eq 0 ]]; then
    echo "Error: No directory found matching: $TREES_DIR/$UUID_PATTERN*" >&2
    exit 1
fi
if [[ ${#MATCHES[@]} -gt 1 ]]; then
    echo "Error: Multiple directories match '$UUID_PATTERN':" >&2
    printf "  %s\n" "${MATCHES[@]}" >&2
    exit 1
fi

if [[ "$SIZE" =~ ^[0-9.]+$ ]]; then
    WORKING_DIR="${MATCHES[0]}/${SIZE}_brsupport"
else
    WORKING_DIR="${MATCHES[0]}/${SIZE}"
fi

if [[ ! -d "$WORKING_DIR" ]]; then
    echo "Error: Directory not found: $WORKING_DIR" >&2
    exit 1
fi

PREFIX="input_v${VERSION}"
OUTDIR="$WORKING_DIR/combined_v${VERSION}"
mkdir -p "$OUTDIR"

LOG_FILES=("$WORKING_DIR"/${PREFIX}_*.log)
TREE_FILES=("$WORKING_DIR"/${PREFIX}_*.trees)

if [[ ${#LOG_FILES[@]} -eq 0 ]]; then
    echo "Error: No log files matching ${PREFIX}_*.log in $WORKING_DIR" >&2
    exit 1
fi

SKIP_COMBINE=false
if ! $OVERWRITE && [[ -f "$OUTDIR/${PREFIX}_combined.log" ]]; then
    echo "Combined output already exists. Skipping combine step."
    SKIP_COMBINE=true
fi

SKIP_RESAMPLE=false
if [[ -n "$N_PER_RUN" ]] && ! $OVERWRITE && [[ -f "$OUTDIR/${PREFIX}_resampled.log" ]]; then
    echo "Resampled output already exists. Skipping resample step."
    SKIP_RESAMPLE=true
fi

TREES_IN="$OUTDIR/${PREFIX}_resampled.trees"
[[ ! -f "$TREES_IN" ]] && TREES_IN="$OUTDIR/${PREFIX}_combined.trees"
MCC_FILE="${TREES_IN%.trees}.mcc"
SKIP_MCC=false
if ! $OVERWRITE && [[ -f "$MCC_FILE" ]]; then
    echo "MCC tree already exists. Skipping TreeAnnotator step."
    SKIP_MCC=true
fi

if $SKIP_COMBINE && $SKIP_RESAMPLE && $SKIP_MCC; then
    echo "Nothing to do. Use -o to overwrite."
    exit 0
fi

echo "Combine configuration:"
echo "  Dir: $WORKING_DIR"
echo "  Runs: ${LOG_FILES[*]##*/}"
echo "  Burnin: ${BURNIN}%"
echo "  Resample: ${N_PER_RUN:-none}"
echo "  Output: $OUTDIR"
$SKIP_COMBINE && echo "  [skip] combine"
$SKIP_RESAMPLE && echo "  [skip] resample"
$SKIP_MCC && echo "  [skip] treeannotator"
echo ""

read -rp "Proceed? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

if [[ -n "$N_PER_RUN" ]]; then
    log_every=$(awk '/^[0-9]/{if(++n==2){print $1; exit}}' "${LOG_FILES[0]}")
    total_samples=$(grep -c "^[0-9]" "${LOG_FILES[0]}")
    post_burnin=$(( total_samples * (100 - BURNIN) / 100 ))
    skip=$(( (post_burnin + N_PER_RUN - 1) / N_PER_RUN ))
    resample=$(( log_every * skip ))
    echo "logEvery=${log_every}, ${total_samples} samples/run, ${post_burnin} post-burnin -> resample ${resample} (~$(( post_burnin / skip )) per run)"
fi

if ! $SKIP_COMBINE; then
    for ext in log trees; do
        if [[ "$ext" == "log" ]]; then
            FILES=("${LOG_FILES[@]}")
            count_cmd() { grep -c "^[0-9]" "$1"; }
        else
            FILES=("${TREE_FILES[@]}")
            count_cmd() { grep -c "^tree " "$1"; }
        fi

        COMBINED="$OUTDIR/${PREFIX}_combined.${ext}"

        echo ""
        echo "=== Combining .${ext} files ==="
        ARGS=()
        for f in "${FILES[@]}"; do ARGS+=(-log "$f"); done
        "$LOGCOMBINER" -b "$BURNIN" "${ARGS[@]}" -o "$COMBINED"
        echo "  Combined: $(count_cmd "$COMBINED") entries"
    done
fi

if [[ -n "$N_PER_RUN" ]] && ! $SKIP_RESAMPLE; then
    for ext in log trees; do
        if [[ "$ext" == "log" ]]; then
            count_cmd() { grep -c "^[0-9]" "$1"; }
        else
            count_cmd() { grep -c "^tree " "$1"; }
        fi

        COMBINED="$OUTDIR/${PREFIX}_combined.${ext}"
        RESAMPLED="$OUTDIR/${PREFIX}_resampled.${ext}"

        echo ""
        echo "=== Resampling .${ext} (interval=${resample}) ==="
        "$LOGCOMBINER" -b 0 -resample "$resample" -log "$COMBINED" -o "$RESAMPLED"
        echo "  Resampled: $(count_cmd "$RESAMPLED") entries"
    done
fi

if ! $SKIP_MCC; then
    TREES_IN="$OUTDIR/${PREFIX}_resampled.trees"
    [[ ! -f "$TREES_IN" ]] && TREES_IN="$OUTDIR/${PREFIX}_combined.trees"
    MCC_FILE="${TREES_IN%.trees}.mcc"

    echo ""
    echo "=== Running TreeAnnotator (MCC) ==="
    pixi run treeannotator "$TREES_IN" "$MCC_FILE"
    echo "MCC tree written to: $MCC_FILE"
fi

echo ""
echo "=== Done ==="
echo "Output: $OUTDIR"
ls -lhS "$OUTDIR"/${PREFIX}_*
