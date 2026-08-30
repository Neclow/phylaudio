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
    echo "  --filter TYPE        Sentence filter applied after split selection:"
    echo "                         ner  = remove sentences with proper nouns or named entities"
    echo "  --method METHOD      Inference method: astral4 or wastral (default: astral4)"
    echo "  -t, --threads N      Number of threads to use (default: 16)"
    echo "  --overwrite          Overwrite existing output files"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 discrete/25ee134c-... --threads 32"
    echo "  $0 discrete/25ee134c-... --include dev,test"
    echo "  $0 discrete/25ee134c-... --exclude train --method wastral"
    echo "  $0 discrete/25ee134c-... --exclude train --filter ner"
    exit 0
fi

# Default values
NUM_THREADS=16
OVERWRITE=false
INPUT_DIR=""
INCLUDE=""
EXCLUDE=""
FILTER=""
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
        --filter)
            FILTER="$2"
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

# Validate --filter
if [ -n "$FILTER" ]; then
    case "$FILTER" in
        ner) ;;
        *) echo "Error: --filter must be 'ner' (got '$FILTER')"; exit 1 ;;
    esac
fi

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

# Build filter ID set from spacy.csv (if --filter is active)
FILTER_FILE=""
if [ -n "$FILTER" ]; then
    # Resolve dataset from the first run's cfg.json
    first_cfg=$(find "${INPUT_DIR}" -type f -name "cfg.json" | head -1)
    if [ -z "$first_cfg" ]; then
        echo "Error: no cfg.json found in $INPUT_DIR"
        exit 1
    fi
    dataset=$(python3 -c "import json; print(json.load(open('$first_cfg'))['dataset'])")
    spacy_csv="data/metadata/${dataset}/spacy.csv"
    if [ ! -f "$spacy_csv" ]; then
        echo "Error: $spacy_csv not found. Run 'pixi run spacy-annotate' first."
        exit 1
    fi
    # Extract split_ids where all flag columns are 0
    FILTER_FILE=$(mktemp)
    python3 -c "
import pandas as pd, sys
df = pd.read_csv('$spacy_csv', index_col='split_id')
flags = ['propn','PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW','LANGUAGE']
ids = df.index[df[flags].sum(axis=1) == 0]
print('\n'.join(ids))
" > "$FILTER_FILE"
    n_filter=$(wc -l < "$FILTER_FILE")
    echo "Filter '$FILTER': $n_filter allowed sentence IDs loaded from $spacy_csv"
    trap 'rm -f "$FILTER_FILE"' EXIT
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

# Output suffix depends on split filter and sentence filter
suffix=""
if [ -n "$SPLITS" ]; then
    splits_label=$(echo "$SPLITS" | tr ',' '\n' | sort | tr '\n' '_' | sed 's/_$//')
    suffix="_${splits_label}"
fi
if [ -n "$FILTER" ]; then
    suffix="${suffix}_${FILTER}"
fi
if [ -n "$suffix" ]; then
    out_pattern="_trees${suffix}_${METHOD}.txt"
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

    # When a split or sentence filter is active, write a filtered Nexus keeping
    # the structural lines (#NEXUS, TAXA block, BEGIN TREES;, END;) and only
    # the matching trees.
    astral_input="$input_file"
    if [ -n "$SPLITS" ] || [ -n "$FILTER" ]; then
        astral_input="${input_file/%${in_pattern}/_trees${suffix}.nex}"

        if [ -n "$SPLITS" ]; then
            awk_pattern=$(echo "$SPLITS" | tr ',' '|')
        else
            awk_pattern=""
        fi

        awk -v pat="$awk_pattern" -v ffile="$FILTER_FILE" '
            BEGIN {
                if (ffile != "") {
                    while ((getline line < ffile) > 0) allowed[line] = 1
                    close(ffile)
                }
            }
            /^[[:space:]]*[Tt]ree / {
                # Extract split_id (e.g. "dev_1544" from "Tree dev_1544_-1")
                split($0, parts)
                tree_name = parts[2]
                n = split(tree_name, tok, "_")
                split_id = tok[1] "_" tok[2]
                # Apply split filter
                if (pat != "" && tree_name !~ "^(" pat ")_") next
                # Apply sentence filter
                if (ffile != "" && !(split_id in allowed)) next
                print
                next
            }
            { print }
        ' "$input_file" > "$astral_input"

        n_trees=$(grep -cE "^[[:space:]]*[Tt]ree " "$astral_input")
        if [ "$n_trees" -eq 0 ]; then
            echo "  No matching trees found in $input_file. Skipping..."
            rm -f "$astral_input"
            continue
        fi
        echo "  Selected $n_trees trees -> $astral_input"
    fi

    "$METHOD" -i "$astral_input" -o "$output_file" -t "$NUM_THREADS" --moreround 2> "$log_file"
done

echo "Done"
