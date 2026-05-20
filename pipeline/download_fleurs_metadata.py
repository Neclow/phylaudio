# pylint: disable=redefined-outer-name
import csv
import json
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import pandas as pd

from src._config import (
    _FLEURS_NAMES,
    _FLEURS_TO_IECOR,
    _FLEURS_TO_INDO1319_FAMILIES,
    DEFAULT_METADATA_DIR,
)
from src.data.glottolog import get_languoid_data
from src.data.speakerpop import download_speakerpop


_TSV_COLS = [
    "sentence_index", "fname", "sentence", "sentence_lower",
    "chars", "num_samples", "gender",
]


def compute_gender_distribution(dataset):
    """Aggregate per-language gender distribution from per-sentence TSVs.

    Per-recording counts (FLEURS deliberately omits speaker IDs, so this is a
    proxy for sex balance — a prolific recordist inflates the apparent share).
    """
    dfs = []
    for tsv in sorted(glob(f"data/datasets/{dataset}/*/*/*.tsv")):
        fleurs_dir = Path(tsv).parents[0].stem
        df = pd.read_csv(tsv, sep="\t", names=_TSV_COLS, quoting=csv.QUOTE_NONE)
        df["fleurs_dir"] = fleurs_dir
        dfs.append(df)

    data = pd.concat(dfs).dropna(subset=["gender"])
    genders = (
        data.groupby(["fleurs_dir", "gender"])
            .agg(n_recordings=("fname", "count"),
                 n_samples=("num_samples", "sum"))
            .reset_index()
            .pivot(index="fleurs_dir", columns="gender",
                   values=["n_recordings", "n_samples"])
            .fillna(0)
    )
    genders.columns = [f"{a}_{b}".lower() for a, b in genders.columns]
    genders = genders.reset_index()
    genders["n_recordings_total"] = (
        genders["n_recordings_male"] + genders["n_recordings_female"]
    )
    genders["pct_female"] = (
        100 * genders["n_recordings_female"] / genders["n_recordings_total"]
    )
    return genders


def parse_args():
    parser = ArgumentParser(
        description="Download Wikimedia articles for FLEURS languages."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=("fleurs-r", "fleurs"),
        help="Dataset to download Wikimedia articles for.",
    )
    parser.add_argument(
        "-g",
        "--glottolog-dir",
        default="extern/glottolog",
        help="Path to glottolog folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(_FLEURS_TO_INDO1319_FAMILIES)

    # Download speaker population data
    speakerpop_data = download_speakerpop(
        dataset=args.dataset, overwrite=args.overwrite
    ).set_index("fleurs_dir")
    speakerpop_data.to_csv(f"{DEFAULT_METADATA_DIR}/{args.dataset}/n_speakers.csv")

    # Compute per-language gender distribution from per-sentence TSVs
    gender_data = compute_gender_distribution(args.dataset)
    gender_data.to_csv(
        f"{DEFAULT_METADATA_DIR}/{args.dataset}/genders.csv", index=False
    )
    print(f"Wrote {len(gender_data)} languages to genders.csv")

    missing_glottocodes = {
        # Arabic FLEURS speakers are Egyptian speakers speaking Modern Standard Arabic
        # Modern Standard Arabic
        "Arabic": "stan1318",
        # Estonian
        "Estonian": "esto1258",
        # Standard Malay
        "Malay": "stan1306",
        # West-Central Oromo, which is a lingua franca in the area
        "Oromo": "west2721",
        # We assume "Persian" in FLEURS refers to Western Persian from FLORES, although it could include Dari
        # I couldn't find a definitive source on this, neither a "Persian dialect" speech classifier to check the data
        "Persian": "west2369",
    }

    for lang, glottocode in missing_glottocodes.items():
        speakerpop_data.loc[speakerpop_data["Language"] == lang, "glottocode"] = (
            glottocode
        )

    # Download glottolog data
    glottolog_data = get_languoid_data(args.glottolog_dir, speakerpop_data.glottocode)
    glottolog_data.index = speakerpop_data.index
    glottolog_data.to_csv(f"{DEFAULT_METADATA_DIR}/{args.dataset}/glottolog.csv")

    # Make languages.json file
    # key: fleurs dir
    # value: {full, fleurs, fleurs_iso639-3, glottolog, speakers, iecor, taxonset}
    languages = {}
    for fleurs_dir, fleurs_info in _FLEURS_NAMES.items():
        fleurs_name = fleurs_info["fleurs"]

        # Get speaker data
        row = speakerpop_data.loc[fleurs_dir, :]
        if row.empty:
            print(f"Warning: No speaker data for {fleurs_dir}")
            continue

        # Get glottocode & full name
        glottocode, full_name = glottolog_data.loc[fleurs_dir, ["glottocode", "name"]]

        # Build language entry (convert NaN to None for valid JSON)
        lang_entry = {
            "full": full_name,
            "fleurs": fleurs_name,
            "fleurs_iso639-3": row["ISO_639-3"],
            "glottolog": glottocode,
            "speakers": {
                "wikimedia": round(row["speakers_wikimedia"], 1),
                "linguameta": round(row["speakers_linguameta"], 1),
            },
        }

        # Add taxon set and iecor language name if available
        if fleurs_dir in _FLEURS_TO_INDO1319_FAMILIES:
            lang_entry["taxonset"] = _FLEURS_TO_INDO1319_FAMILIES[fleurs_dir]
        if fleurs_dir in _FLEURS_TO_IECOR:
            lang_entry["iecor"] = _FLEURS_TO_IECOR[fleurs_dir]

        languages[fleurs_dir] = lang_entry

    # Write languages.json
    output_path = f"{DEFAULT_METADATA_DIR}/{args.dataset}/languages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(languages, f, indent=2)
    print(f"Wrote {len(languages)} languages to {output_path}")
