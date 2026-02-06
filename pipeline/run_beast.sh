#!/bin/bash
# Run BEAST2 with GPU acceleration
set -e

RESUME=false
OVERWRITE=false
VALIDATE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [-r] [-o] [-v] [-h|--help] <uuid> <size> <version> <run_number>

Run BEAST2 with GPU acceleration (BEAGLE).

Options:
  -r           Resume from existing state file
  -o           Overwrite existing output files
  -v           Validate XML without running BEAST2
  -h, --help   Show this help message and exit

Arguments:
  uuid         Run UUID under data/trees/beast/ (supports partial matching)
               e.g., "c59" matches "c5969221-5f5d-4f4b-874e-dcf96e9831b9"
  size         Branch support threshold (e.g., 0.01, 0.05)
  version      XML version number (e.g., 11 for input_v11.xml)
  run_number   Run number (1-99), used to compute seed

Seed is computed as: version * 100 + run_number
  e.g., version=11, run=3 -> seed=1103

Examples:
  run_beast.sh c59 0.01 11 3         # c5969221-.../0.01_brsupport, v11, seed 1103
  run_beast.sh -r c59 0.01 11 3      # Resume from state file
  run_beast.sh -o c59 0.01 11 3      # Overwrite existing output
  run_beast.sh abc123 0.05 12 1      # abc123.../0.05_brsupport, v12, seed 1201

Output files (in matched directory):
  input_v{version}_{seed}.log        # Parameter trace
  input_v{version}_{seed}.trees      # Tree trace
  input_v{version}_{seed}.xml.state  # Checkpoint for resuming
EOF
    exit 1
}

while [[ "$1" == -* ]]; do
    case "$1" in
        -r) RESUME=true; shift ;;
        -o) OVERWRITE=true; shift ;;
        -v) VALIDATE=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[[ $# -lt 4 || "$1" == "-h" || "$1" == "--help" ]] && usage

UUID_PATTERN=$1
SIZE=$2
VERSION=$3
RUN=$4

# Seed = version * 100 + run_number (e.g., v11 run 3 = 1103)
SEED="${VERSION}$(printf '%02d' "$RUN")"

# Alternate GPUs: odd seed (1,3,5...) -> GPU 1, even seed (2,4,6...) -> GPU 2
if (( SEED % 2 == 1 )); then
    BEAGLE_ORDER=1
else
    BEAGLE_ORDER=2
fi

# BEAST_DIR is parent of pipeline/ (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAST_DIR="$(dirname "$SCRIPT_DIR")"
TREES_DIR="$BEAST_DIR/data/trees/beast"

# Expand partial UUID pattern (e.g., "c59" -> "c5969221-5f5d-4f4b-874e-dcf96e9831b9")
shopt -s nullglob
MATCHES=("$TREES_DIR"/"$UUID_PATTERN"*)
shopt -u nullglob

if [[ ${#MATCHES[@]} -eq 0 ]]; then
    echo "Error: No directory found matching: $TREES_DIR/$UUID_PATTERN*"
    exit 1
fi
if [[ ${#MATCHES[@]} -gt 1 ]]; then
    echo "Error: Multiple directories match '$UUID_PATTERN':"
    printf "  %s\n" "${MATCHES[@]}"
    exit 1
fi

WORKING_DIR="${MATCHES[0]}/${SIZE}_brsupport"
if [[ ! -d "$WORKING_DIR" ]]; then
    echo "Error: Directory not found: $WORKING_DIR"
    exit 1
fi

INPUT_FILE="$WORKING_DIR/input_v${VERSION}.xml"
STATE_FILE="$WORKING_DIR/input_v${VERSION}_${SEED}.xml.state"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

if $RESUME && [[ ! -f "$STATE_FILE" ]]; then
    echo "Error: State file not found for resume: $STATE_FILE"
    exit 1
fi

echo "BEAST2 run configuration:"
echo "  Version: v${VERSION}"
echo "  Run: ${RUN}"
echo "  Seed: ${SEED}"
echo "  Dir: ${WORKING_DIR}"
echo "  Input: ${INPUT_FILE}"
echo "  State: ${STATE_FILE}"
echo "  Resume: ${RESUME}"
echo "  Overwrite: ${OVERWRITE}"
echo "  Validate only: ${VALIDATE}"
echo ""

read -rp "Proceed? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

BEAST_ARGS=(
    -beagle_GPU
    -beagle_order "$BEAGLE_ORDER"
    -seed "$SEED"
    -working
    -packagedir "$BEAST_DIR/.beast"
    -statefile "$STATE_FILE"
)

if $VALIDATE; then
    BEAST_ARGS+=(-validate)
fi

if $RESUME; then
    BEAST_ARGS+=(-resume)
fi

if $OVERWRITE; then
    BEAST_ARGS+=(-overwrite)
fi

cd "$BEAST_DIR"
pixi run beast2 "${BEAST_ARGS[@]}" "$INPUT_FILE"
