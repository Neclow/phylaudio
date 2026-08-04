#!/bin/bash

set -e

# Show help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 <dirname> [OPTIONS]"
    echo ""
    echo "Run ASTRAL/wASTRAL, compute tree stats, and summarise metrics"
    echo "for all runs under a per-sentence tree directory."
    echo ""
    echo "Arguments:"
    echo "  <dirname>            Name of the subdirectory in data/trees/per_sentence"
    echo ""
    echo "Options:"
    echo "  --include s1[,s2]    Only include these splits (comma-separated: train,dev,test)"
    echo "  --exclude s1[,s2]    Exclude these splits (comma-separated: train,dev,test)"
    echo "  --method METHOD      Species-tree method: astral4 or wastral (default: astral4)"
    echo "  -t, --threads N      Number of threads (default: 16)"
    echo "  --overwrite          Overwrite existing output files"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 discrete3+vote --exclude train"
    echo "  $0 discrete3+vote --include dev,test --method wastral"
    exit 0
fi

# Default values
DIRNAME=""
INCLUDE=""
EXCLUDE=""
METHOD="astral4"
NUM_THREADS=16
OVERWRITE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --include)   INCLUDE="$2";    shift 2 ;;
        --exclude)   EXCLUDE="$2";    shift 2 ;;
        --method)    METHOD="$2";     shift 2 ;;
        -t|--threads) NUM_THREADS="$2"; shift 2 ;;
        --overwrite) OVERWRITE=true;  shift ;;
        -h|--help)   shift ;;
        *)
            if [ -z "$DIRNAME" ]; then
                DIRNAME="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$DIRNAME" ]; then
    echo "Error: Missing required argument <dirname>"
    echo "Usage: $0 <dirname> [OPTIONS]"
    exit 1
fi

# Build flags to forward
SPLIT_FLAGS=""
SPLITS_CSV=""
if [ -n "$INCLUDE" ]; then
    SPLIT_FLAGS="--include $INCLUDE"
    SPLITS_CSV="$INCLUDE"
elif [ -n "$EXCLUDE" ]; then
    SPLIT_FLAGS="--exclude $EXCLUDE"
fi

OVERWRITE_FLAG=""
if [ "$OVERWRITE" = true ]; then
    OVERWRITE_FLAG="--overwrite"
fi

# --- Step 1: ASTRAL / wASTRAL ---
echo "=== Step 1/3: $METHOD ==="
bash pipeline/sentence_trees_astral.sh "$DIRNAME" \
    $SPLIT_FLAGS \
    --method "$METHOD" \
    -t "$NUM_THREADS" \
    $OVERWRITE_FLAG

# Resolve SPLITS_CSV when --exclude was used
if [ -n "$EXCLUDE" ] && [ -z "$SPLITS_CSV" ]; then
    for s in train dev test; do
        case ",$EXCLUDE," in
            *",$s,"*) ;;
            *) SPLITS_CSV="${SPLITS_CSV:+$SPLITS_CSV,}$s" ;;
        esac
    done
fi

STATS_SPLIT_FLAG=""
SUMMARY_SPLIT_FLAG=""
if [ -n "$SPLITS_CSV" ]; then
    STATS_SPLIT_FLAG="--splits $SPLITS_CSV"
    SUMMARY_SPLIT_FLAG="--splits $SPLITS_CSV"
fi

# --- Step 2: Tree stats ---
echo ""
echo "=== Step 2/3: Tree stats ==="
Rscript pipeline/sentence_trees_stats.R "$DIRNAME" $STATS_SPLIT_FLAG $OVERWRITE_FLAG

# --- Step 3: Summary ---
echo ""
echo "=== Step 3/3: Summary ==="
python -m pipeline.sentence_trees_summary "$DIRNAME" \
    -ot "$METHOD" \
    $SUMMARY_SPLIT_FLAG \
    -nt "$NUM_THREADS" \
    $OVERWRITE_FLAG

echo ""
echo "=== Done ==="
