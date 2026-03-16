#!/usr/bin/env bash
set -euo pipefail

# Download IECoR posterior .trees and .log files from MPI-EVA share
# Source: https://share.eva.mpg.de/index.php/s/E4Am2bbBA3qLngC
# Files from: 01_Main_Analysis_M3/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin

DEST_DIR="data/trees/beast"
mkdir -p "$DEST_DIR"

BASE_URL="https://share.eva.mpg.de/public.php/dav/files/E4Am2bbBA3qLngC/01_Main_Analysis_M3/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin/"

STEM="IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined"

for EXT in trees log; do
    echo "Downloading ${STEM}.${EXT} ..."
    wget -O "${DEST_DIR}/${STEM}.${EXT}" \
        "${BASE_URL}${STEM}.${EXT}"
done

MCC="IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_mcc.tree"
echo "Downloading ${MCC} ..."
wget -O "${DEST_DIR}/${MCC}" "${BASE_URL}${MCC}"

# Prior log (for root age comparison plots)
mkdir -p "${DEST_DIR}/priors"
PRIOR="${STEM}_PRIOR.log"
echo "Downloading ${PRIOR} ..."
wget -O "${DEST_DIR}/priors/${PRIOR}" "${BASE_URL}${PRIOR}"

echo "Done. Files saved to ${DEST_DIR}/"
