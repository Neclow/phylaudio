"""Download GeoJSON and Natural Earth shapefiles."""

import os
import zipfile
from io import BytesIO

import requests

from src._config import DEFAULT_GEO_DIR, GEOJSON_PATH

TIMEOUT = 30

GEOJSON_URL = (
    "https://raw.githubusercontent.com/Glottography/asher2007world"
    "/5590fcc8870d8883a323ff0defd5a1a6a03e9611/raw/dataset.geojson"
)


def download_file(url, output_path):
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def download_and_unzip(url, output_dir):
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        zf.extractall(output_dir)


def main():
    os.makedirs(DEFAULT_GEO_DIR, exist_ok=True)
    print(f"Downloading language polygons → {GEOJSON_PATH}")
    download_file(GEOJSON_URL, GEOJSON_PATH)

    ne_dir = f"{DEFAULT_GEO_DIR}/naturalearth"
    os.makedirs(ne_dir, exist_ok=True)

    for dataset in ["ne_110m_admin_0_countries", "ne_50m_land"]:
        category = "cultural" if "countries" in dataset else "physical"
        scale = dataset.split("_")[1]
        url = f"https://naciscdn.org/naturalearth/{scale}/{category}/{dataset}.zip"
        print(f"Downloading {dataset} → {ne_dir}")
        download_and_unzip(url, ne_dir)


if __name__ == "__main__":
    main()
