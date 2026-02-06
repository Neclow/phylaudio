# pylint: disable=redefined-outer-name
import json
from argparse import ArgumentParser

from src._config import (
    _FLEURS_NAMES,
    _FLEURS_TO_IECOR,
    _FLEURS_TO_INDO1319_FAMILIES,
    DEFAULT_METADATA_DIR,
)
from src.data.glottolog import get_languoid_data
from src.data.speakerpop import download_speakerpop


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
