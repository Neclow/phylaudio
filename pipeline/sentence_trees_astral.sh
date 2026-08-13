#!/bin/bash

set -e

PER_SENTENCE_DIR="data/trees/per_sentence"
VALID_SPLITS="train dev test"

validate_splits() {
    local IFS=','
    for s in $1; do
        case " $VALID_SPLITS " in
            *" $s "*) ;;
            *) echo "Error: invalid split '$s' (must be train, dev, or test)"; exit 1 ;;
        esac
    done
}

# Show help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 <dirname> [OPTIONS]"
    echo ""
    echo "Run ASTRAL4 on the _trees.nex file(s) under a directory."
    echo ""
    echo "Arguments:"
    echo "  <dirname>            Name of the subdirectory in $PER_SENTENCE_DIR"
    echo ""
    echo "Options:"
    echo "  --include s1[,s2]    Only include these splits (comma-separated: train,dev,test)"
    echo "  --exclude s1[,s2]    Exclude these splits (comma-separated: train,dev,test)"
    echo "  --method METHOD      Inference method: astral4 or wastral (default: astral4)"
    echo "  -t, --threads N      Number of threads to use (default: 16)"
    echo "  --overwrite          Overwrite existing output files"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 discrete/25ee134c-... --threads 32"
    echo "  $0 discrete/25ee134c-... --include dev,test"
    echo "  $0 discrete/25ee134c-... --exclude train --method wastral"
    exit 0
fi

# Default values
NUM_THREADS=16
OVERWRITE=false
INPUT_DIR=""
INCLUDE=""
EXCLUDE=""
METHOD="astral4"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --include)
            INCLUDE="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        -h|--help)
            # Already handled above
            shift
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="${PER_SENTENCE_DIR}/$1"
            fi
            shift
            ;;
    esac
done

# Fail if no directory argument provided
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Missing required argument <dirname>"
    echo "Usage: $0 <dirname> [OPTIONS]"
    echo "Use --help for more information"
    exit 1
fi

# Validate flags
if [ -n "$INCLUDE" ] && [ -n "$EXCLUDE" ]; then
    echo "Error: --include and --exclude are mutually exclusive"
    exit 1
fi

if [ -n "$INCLUDE" ]; then
    validate_splits "$INCLUDE"
fi
if [ -n "$EXCLUDE" ]; then
    validate_splits "$EXCLUDE"
fi

case "$METHOD" in
    astral4|wastral) ;;
    *) echo "Error: --method must be astral4 or wastral (got '$METHOD')"; exit 1 ;;
esac

# Build the set of splits to keep
if [ -n "$INCLUDE" ]; then
    SPLITS="$INCLUDE"
elif [ -n "$EXCLUDE" ]; then
    SPLITS=""
    for s in $VALID_SPLITS; do
        case ",$EXCLUDE," in
            *",$s,"*) ;;
            *) SPLITS="${SPLITS:+$SPLITS,}$s" ;;
        esac
    done
fi

# Glob input_dir/*/_trees.nex
in_pattern="_trees.nex"
input_files=$(find "${INPUT_DIR}" -type f -name "*${in_pattern}")

n_files=$(echo "$input_files" | wc -l)
if [ -z "$input_files" ]; then
    echo "No input files found in $INPUT_DIR with pattern *${in_pattern}."
    exit 1
fi

echo "Found $n_files files to process."

# Output suffix depends on whether a split filter was requested
if [ -n "$SPLITS" ]; then
    # Sorted, underscore-joined label (e.g. "dev_test")
    splits_label=$(echo "$SPLITS" | tr ',' '\n' | sort | tr '\n' '_' | sed 's/_$//')
    out_pattern="_trees_${splits_label}_${METHOD}.txt"
else
    out_pattern="_trees_${METHOD}.txt"
fi

# Run astral on each file
i=1
for input_file in $input_files; do
    echo "Processing file $i/$n_files: $input_file"

    # Increment counter
    i=$((i + 1))

    output_file="${input_file/%${in_pattern}/${out_pattern}}"
    log_file="${output_file%.txt}.log"

    # Skip if output file exists and not overwriting
    if [ -f "$output_file" ] && [ "$OVERWRITE" = false ]; then
        echo "Output file $output_file already exists. Skipping..."
        continue
    fi

    # When a split filter is active, write a filtered Nexus keeping the structural
    # lines (#NEXUS, TAXA block, BEGIN TREES;, END;) and only the matching trees.
    astral_input="$input_file"
    if [ -n "$SPLITS" ]; then
        astral_input="${input_file/%${in_pattern}/_trees_${splits_label}.nex}"
        # Build awk regex alternation from the splits (e.g. "dev|test")
        awk_pattern=$(echo "$SPLITS" | tr ',' '|')
        awk -v pat="$awk_pattern" '
            /^[[:space:]]*[Tt]ree / { if ($0 ~ "[Tt]ree (" pat ")_") print; next }
            { print }
        ' "$input_file" > "$astral_input"

        n_trees=$(grep -cE "^[[:space:]]*[Tt]ree ($awk_pattern)_" "$astral_input")
        if [ "$n_trees" -eq 0 ]; then
            echo "  No matching trees found in $input_file. Skipping..."
            rm -f "$astral_input"
            continue
        fi
        echo "  Selected $n_trees trees (${SPLITS}) -> $astral_input"
    fi

    "$METHOD" -i "$astral_input" -o "$output_file" -t "$NUM_THREADS" --moreround 2> "$log_file"
done

echo "Done"
