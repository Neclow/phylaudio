# pylint: disable=redefined-outer-name
import json
from argparse import ArgumentParser

import pandas as pd

from src._config import _FLEURS_NAMES, DEFAULT_METADATA_DIR, FLEURS_TO_IECOR
from src.data.glottolog import get_languoid_data
from src.data.speakerpop import download_speakerpop

# Taxonset memberships by IECOR name (from BEAST2 template.xml)
TAXONSETS = {
    "germanic": [
        "Afrikaans",
        "Danish",
        "Dutch",
        "English",
        "German",
        "Icelandic",
        "Luxembourgish",
        "NorwegianBokmal",
        "Swedish",
    ],
    "slavic": [
        "Belarusian",
        "Bulgarian",
        "Czech",
        "Macedonian",
        "Polish",
        "Russian",
        "SerboCroatian",
        "Slovak",
        "Slovene",
        "Ukrainian",
    ],
    "indoaryan": [
        "Assamese",
        "Bengali",
        "Gujarati",
        "Hindi",
        "Marathi",
        "Nepali",
        "Oriya",
        "Punjabi",
        "Sindhi",
        "Urdu",
    ],
    "iranian": ["KurdishCJafi", "Pashto", "PersianTehran", "Tajik"],
    "latinofaliscan": [
        "Catalan",
        "French",
        "Galician",
        "Italian",
        "Portuguese",
        "Romanian",
        "Spanish",
    ],
    "baltic": ["Latvian", "Lithuanian"],
    "armenian": ["ArmenianEastern"],
    "Greek": ["Greek"],
}

# Invert taxonsets: iecor_name -> taxonset
IECOR_TO_TAXONSET = {
    iecor: taxonset for taxonset, iecors in TAXONSETS.items() for iecor in iecors
}


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

    # Download speaker population data
    speakerpop_data = download_speakerpop(
        dataset=args.dataset, overwrite=args.overwrite
    )
    speakerpop_data.to_csv(
        f"{DEFAULT_METADATA_DIR}/{args.dataset}/n_speakers.csv", index=False
    )

    missing_glottocodes = {
        "Arabic": "stan1318",  # Modern Standard Arabic
        "Estonian": "esto1258",  # Estonian
        "Malay": "stan1306",  # Standard Malay
        "Oromo": "west2721",  # West-Central Oromo, which is a lingua franca in the area
        "Persian": "west2369",  # Western Persian, which contains the standard Tehran dialect
    }

    for lang, glottocode in missing_glottocodes.items():
        speakerpop_data.loc[speakerpop_data["Language"] == lang, "glottocode"] = (
            glottocode
        )

    # Download glottolog data
    glottolog_data = get_languoid_data(args.glottolog_dir, speakerpop_data.glottocode)
    glottolog_data.to_csv(f"{DEFAULT_METADATA_DIR}/{args.dataset}/glottolog.csv")

    # Make languages.json file
    # key: fleurs dir
    # value: {full, fleurs, fleurs_iso639-3, glottolog, speakers, iecor, taxonset}
    languages = {}
    for fleurs_dir, fleurs_info in _FLEURS_NAMES.items():
        fleurs_name = fleurs_info["fleurs"]

        # Get speaker data
        row = speakerpop_data[speakerpop_data["fleurs_dir"] == fleurs_dir]
        if row.empty:
            print(f"Warning: No speaker data for {fleurs_dir}")
            continue

        row = row.iloc[0]
        glottocode = row["glottocode"]

        # Get glottolog full name
        glottolog_row = glottolog_data[glottolog_data["glottocode"] == glottocode]
        full_name = glottolog_row.index[0] if not glottolog_row.empty else fleurs_name

        # Build language entry (convert NaN to None for valid JSON)
        wikimedia = row["speakers_wikimedia"]
        linguameta = row["speakers_linguameta"]
        lang_entry = {
            "full": full_name,
            "fleurs": fleurs_name,
            "fleurs_iso639-3": row["ISO_639-3"],
            "glottolog": glottocode,
            "speakers": {
                "wikimedia": round(wikimedia, 1),
                "linguameta": round(linguameta, 1),
            },
        }

        # Add iecor and taxonset if language is in FLEURS_TO_IECOR
        if fleurs_dir in FLEURS_TO_IECOR:
            iecor_name = FLEURS_TO_IECOR[fleurs_dir]
            lang_entry["iecor"] = iecor_name
            if iecor_name in IECOR_TO_TAXONSET:
                lang_entry["taxonset"] = IECOR_TO_TAXONSET[iecor_name]

        languages[fleurs_dir] = lang_entry

    # Write languages.json
    output_path = f"{DEFAULT_METADATA_DIR}/{args.dataset}/languages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(languages, f, indent=2)
    print(f"Wrote {len(languages)} languages to {output_path}")
