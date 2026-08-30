#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <uuid> <size> <version> [seed]

Run all nested sampling (NS) models for a BEAST2 run.

Arguments:
  uuid      Run UUID under data/trees/beast/ (supports partial matching)
            e.g., "ba9" matches "ba9f2d2a-27f3-4100-a1c0-43f8fe1c39fc"
  size      Branch support threshold (e.g., 0.05_brsupport_dev_test)
  version   NS version number — runs XMLs in ns_v{version}/
  seed      Random seed (default: 101)

All input_ns_*.xml files in the ns_v{version}/ subdirectory are auto-discovered.

Examples:
  run_beast_ns.sh ba9 0.05_brsupport_dev_test 3
  run_beast_ns.sh ba9 0.05_brsupport_dev_test 3 42
EOF
    exit 1
}

[[ $# -lt 3 || "$1" == "-h" || "$1" == "--help" ]] && usage

UUID_PATTERN=$1
SIZE=$2
VERSION=$3
SEED=${4:-101}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAST_DIR="$(dirname "$SCRIPT_DIR")"
TREES_DIR="$BEAST_DIR/data/trees/beast"

# Expand partial UUID
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

NS_DIR="${MATCHES[0]}/${SIZE}/ns_v${VERSION}"
if [[ ! -d "$NS_DIR" ]]; then
    echo "Error: NS directory not found: $NS_DIR"
    exit 1
fi

# Auto-discover NS XML files
shopt -s nullglob
XMLS=("$NS_DIR"/input_ns_*.xml)
shopt -u nullglob

if [[ ${#XMLS[@]} -eq 0 ]]; then
    echo "Error: No input_ns_*.xml files found in $NS_DIR"
    exit 1
fi

BEAST_FLAGS=(-overwrite -working -beagle_GPU -beagle_order 1 -packagedir "$BEAST_DIR/.beast" -seed "${SEED}")

echo "=== NS production runs (v${VERSION}, seed=${SEED}) ==="
echo "=== Directory: $NS_DIR ==="
echo "=== ${#XMLS[@]} models ==="
echo ""

for i in "${!XMLS[@]}"; do
    xml_path="${XMLS[$i]}"
    xml="$(basename "$xml_path")"
    n=$((i + 1))
    stdout="${NS_DIR}/${xml%.xml}_${SEED}.stdout"

    echo "[${n}/${#XMLS[@]}] ${xml}"
    pixi run beast2 "${BEAST_FLAGS[@]}" "${xml_path}" 2>&1 | tee "$stdout"

    if grep -q 'RuntimeException\|Fatal Error\|SAXParseException\|Exception in thread' "$stdout"; then
        echo "FATAL: ${xml} failed. See ${stdout}"
        exit 1
    fi
    if ! grep -q 'Marginal likelihood:' "$stdout"; then
        echo "FATAL: ${xml} produced no marginal likelihood. See ${stdout}"
        exit 1
    fi
    echo ""
done

echo "=== All done. Extracting summary ==="

mkdir -p "${NS_DIR}/results"
SUMMARY="${NS_DIR}/results/ns_summary.tsv"
TMPFILE="${SUMMARY}.tmp"
printf "Model\tML\tSD\tH\tln_BF\n" > "$SUMMARY"
for xml_path in "${XMLS[@]}"; do
    xml="$(basename "$xml_path")"
    log="${NS_DIR}/${xml%.xml}_${SEED}.stdout"
    name="${xml#input_ns_}"
    name="${name%.xml}"
    if [[ -f "$log" ]]; then
        ml=$(grep -oP 'Marginal likelihood: \K[-0-9.]+' "$log" | tail -1)
        sd=$(grep -oP 'SD=\(\K[0-9.]+' "$log" | tail -1)
        h=$(grep -oP 'Information: \K[0-9.]+' "$log" | tail -1)
        printf "%s\t%s\t%s\t%s\n" "$name" "$ml" "$sd" "$h"
    else
        printf "%s\t-\t-\t-\n" "$name"
    fi
done >> "$SUMMARY"

# Sort by group: M0 (baseline), S* (substitution), C* (constraints+), R* (constraints-), A* (calibrations-)
# Then compute ln BF = ML_model - ML_M0
head -1 "$SUMMARY" > "$TMPFILE"
tail -n +2 "$SUMMARY" | awk -F'\t' '{
    if ($1 ~ /^M/) o=1; else if ($1 ~ /^S/) o=2; else if ($1 ~ /^C/) o=3;
    else if ($1 ~ /^R/) o=4; else if ($1 ~ /^A/) o=5; else o=9;
    print o "\t" $0
}' | sort -t$'\t' -k1,1n -k2,2 | cut -f2- | awk -F'\t' -v OFS='\t' '
    NR==1 { ml0=$2 }
    { print $1, $2, $3, $4, ($2=="-" ? "-" : $2-ml0) }
' >> "$TMPFILE"
mv "$TMPFILE" "$SUMMARY"

echo "Summary written to ${NS_DIR}/results/ns_summary.tsv"
