"""Download PHOIBLE phoneme inventories and aggregate to one row per language.

Outputs a CSV with columns: Glottocode, n_phonemes, n_consonants, n_vowels,
plus a binary indicator per phonological feature:
  - has_{feat}: binary (1 if any segment has [+feat], else 0)

Aggregation: computes features per inventory, then takes the median across
all inventories per language (following Anderson et al. 2023, J. Lang. Evol.).
"""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from io import StringIO

import pandas as pd
import requests

from src._config import DEFAULT_METADATA_DIR

PHOIBLE_URL = "https://raw.githubusercontent.com/phoible/dev/7030ae02863f0e1ddaf67f0f950c0ea1477cd4ee/data/phoible.csv"

# Source priority hierarchy from Urban & Moran (2021), PLoS One 16(2):e0245522.
# Maximizes one-inventory-per-doculect and inclusion of contrastive tone.
SOURCE_PRIORITY = ["ph", "gm", "saphon", "uz", "ea", "er", "spa", "aa", "ra", "upsid"]

# Additional glottocodes to download for languages whose PHOIBLE entry
# uses a different glottocode than the one in our metadata.
GLOTTOCODE_REMAPPINGS = {
    "latv1249": "stan1325",  # Latvian: Standard Latvian in PHOIBLE
    "nucl1276": "nort2646",  # Pashto: Northern Pashto (most spoken variety)
    "pedi1238": "sout2807",  # Northern Sotho: Sesotho (closest in PHOIBLE)
    "west2721": "east2652",  # Oromo: Eastern Oromo
    "uzbe1247": "nort2690",  # Uzbek: Northern Uzbek (UPSID)
}

# Languages absent from PHOIBLE (not remapped):
# - Belarusian (bela1254): closest neighbours are Ukrainian (first) and Russian (second), both of which are in PHOIBLE.
#   Source: https://doi.org/10.1515/9783110542431-007; https://doi.org/10.1017/CBO9780511486807
# - Bosnian (bosn1245): mutually intelligible with Serbian/Croatian (PHOIBLE has both),
#   but treated as a separate language in FLEURS.


def parse_args():
    parser = ArgumentParser(
        description="Download and aggregate PHOIBLE phoneme inventories",
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
    glottocodes |= set(GLOTTOCODE_REMAPPINGS.values())

    # Download raw PHOIBLE data
    response = requests.get(PHOIBLE_URL, timeout=10)
    response.raise_for_status()
    raw_data = pd.read_csv(StringIO(response.content.decode("utf-8")), low_memory=False)

    data = raw_data.query("Glottocode.isin(@glottocodes)").copy()

    # Remap glottocodes back to our metadata's codes
    reverse_map = {v: k for k, v in GLOTTOCODE_REMAPPINGS.items()}
    data["Glottocode"] = data["Glottocode"].replace(reverse_map)

    # Filter to known sources
    data = data[data["Source"].str.lower().isin(SOURCE_PRIORITY)].copy()

    # Identify feature columns (from "tone" onward)
    feat_cols = list(data.columns[data.columns.get_loc("tone") :])

    # Step 1: Compute features per inventory
    def _agg_inventory(group):
        row = {
            "n_phonemes": len(group),
            "n_consonants": (group["SegmentClass"] == "consonant").sum(),
            "n_vowels": (group["SegmentClass"] == "vowel").sum(),
        }
        for feat in feat_cols:
            row[f"has_{feat}"] = int((group[feat] == "+").any())
        return pd.Series(row)

    per_inv = data.groupby(["Glottocode", "InventoryID"]).apply(
        _agg_inventory, include_groups=False
    )

    n_inventories = per_inv.groupby("Glottocode").size()
    print(
        f"Inventories per language: min={n_inventories.min()}, "
        f"median={n_inventories.median():.0f}, max={n_inventories.max()}"
    )

    # Step 2: Median across inventories per language.
    # For size columns (n_phonemes, etc.) this gives the median count.
    # For has_{feat} binary indicators this is a majority vote (>50% of inventories).
    agg = per_inv.groupby("Glottocode").median()

    # Report coverage
    expected = {v["glottolog"] for v in languages.values()}
    missing = expected - set(agg.index)

    output_file = f"{dataset_meta_dir}/phoible.csv"
    agg.to_csv(output_file)

    print(f"Saved {len(agg)} languages x {len(agg.columns)} features to {output_file}")
    if missing:
        print(f"Missing: {missing}")
