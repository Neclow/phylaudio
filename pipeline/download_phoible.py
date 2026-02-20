"""Extract phoneme information from a language dataset using phoible"""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from io import StringIO

import pandas as pd
import requests

from src._config import DEFAULT_METADATA_DIR

PHOIBLE_URL = "https://raw.githubusercontent.com/phoible/dev/7030ae02863f0e1ddaf67f0f950c0ea1477cd4ee/data/phoible.csv"


def parse_args():
    parser = ArgumentParser(
        description="Arguments for phoible lineage extraction",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help=(
            "Dataset. Example: `fleurs`. "
            f"Has to have a folder in `{DEFAULT_METADATA_DIR}` with a `languages.json` file.",
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_meta_dir = f"{DEFAULT_METADATA_DIR}/{args.dataset}"

    with open(f"{dataset_meta_dir}/languages.json", "r", encoding="utf-8") as f:
        languages = json.load(f)

    glottocodes = {v["glottolog"] for v in languages.values()}

    response = requests.get(PHOIBLE_URL, timeout=10)
    response.raise_for_status()
    raw_data = pd.read_csv(StringIO(response.content.decode("utf-8")))

    data = raw_data.query("Glottocode.isin(@glottocodes)")

    output_file = f"{dataset_meta_dir}/phoible.csv"

    print(
        f"Found data for {data.Glottocode.nunique()} languages. Saving to {output_file}."
    )

    print(f"Missing: {set(glottocodes) - set(data.Glottocode.unique())}")

    data.to_csv(output_file, index=False)
