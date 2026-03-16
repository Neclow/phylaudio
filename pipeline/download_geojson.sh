#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/metadata/fleurs-r
wget -O data/metadata/fleurs-r/dataset.geojson \
  "https://raw.githubusercontent.com/Glottography/asher2007world/main/raw/dataset.geojson"
