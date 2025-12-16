"""Extract lineages from a language dataset using glottolog"""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd
from pyglottolog import Glottolog
from tqdm import tqdm

from src._config import DEFAULT_METADATA_DIR


def parse_args():
    parser = ArgumentParser(
        description="Arguments for Glottolog lineage extraction",
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
    parser.add_argument(
        "-g",
        "--glottolog-dir",
        default="extern/glottolog",
        help="Path to glottolog folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_meta_dir = f"{DEFAULT_METADATA_DIR}/{args.dataset}"

    g = Glottolog(args.glottolog_dir)

    glottolog_data = {}

    with open(f"{dataset_meta_dir}/languages.json", "r", encoding="utf-8") as f:
        languages = json.load(f)

    print(f"Found {len(languages)} languages")

    for code in tqdm(languages):
        glottocode = languages[code]["glottolog"]

        languoid = g.languoid(id_=glottocode)

        lineage_codes = [x[1] for x in languoid.lineage]

        glottolog_data[code] = {
            "name": languoid.name,
            "glottocode": glottocode,
            "n_speakers": languages[code]["speakers"],
            "longitude": languoid.longitude,
            "latitude": languoid.latitude,
            "codes": lineage_codes,
        }

    df = pd.DataFrame.from_dict(glottolog_data, orient="index")

    df = (
        df.merge(
            pd.DataFrame(df.codes.to_list(), index=df.index).add_prefix("H"),
            left_index=True,
            right_index=True,
        )
        .drop("codes", axis=1)
        .reset_index(names="code")
    )

    output_file = f"{dataset_meta_dir}/glottolog.csv"

    df.to_csv(output_file, index=False)
