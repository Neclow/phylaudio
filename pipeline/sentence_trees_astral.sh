#!/bin/bash

set -e

PER_SENTENCE_DIR="data/trees/per_sentence"

# Show help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 <dirname> [OPTIONS]"
    echo ""
    echo "Run ASTRAL4 on multiple tree files in a directory."
    echo ""
    echo "Arguments:"
    echo "  <dirname>        Name of the subdirectory in $PER_SENTENCE_DIR"
    echo ""
    echo "Options:"
    echo "  -t, --threads N  Number of threads to use (default: 16)"
    echo "  --overwrite      Overwrite existing output files"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 pdist --threads 8 --overwrite"
    exit 0
fi

# Default values
NUM_THREADS=16
OVERWRITE=false
INPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            NUM_THREADS="$2"
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
            # Assume it's the dirname if INPUT_DIR is not set
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

# Glob input_dir/*/*_trees.txt
in_pattern="_trees.txt"
input_files=$(find "${INPUT_DIR}" -type f -name "*${in_pattern}")

n_files=$(echo "$input_files" | wc -l)
if [ "$n_files" -eq 0 ]; then
    echo "No input files found in $INPUT_DIR with pattern *${in_pattern}."
    exit 1
fi

echo "Found $n_files files to process."

out_pattern="_trees_astral4.txt"

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

    astral4 -i "$input_file" -o "$output_file" -t "$NUM_THREADS" --moreround 2> "$log_file"
done

echo "Done"
