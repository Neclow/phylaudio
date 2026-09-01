"""Prepare regression metadata from BEAST MCC trees and reference data.

Inputs:  a BEAST run ID and subdirectory
Options: --version, --cognate_beast_dir, --dataset
Flow:    Load MCC trees (speech + cognate)
                    |
                    v
         Extract branch rate medians and SplitsTree delta scores
                    |
                    v
         Join with Glottolog coordinates, speaker counts, PHOIBLE inventories
Outputs: metadata.csv, metadata_with_inventory.csv in the BEAST run directory
"""

import argparse
import json
import os
import re

import pandas as pd

from src._config import COGNATE_BEAST_DIR, DEFAULT_METADATA_DIR
from src.tasks.phylo.beast import find_beast_mcc, resolve_beast_dir
from src.tasks.phylo.splitstree import extract_delta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare regression metadata from BEAST MCC trees."
    )
    parser.add_argument("run_id", help="BEAST run UUID, prefix, or full path")
    parser.add_argument("subdir", help="Subdirectory name or prefix within the run")
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="BEAST input version (default: 1, tree = input_v{version}.mcc)",
    )
    parser.add_argument(
        "--cognate_beast_dir",
        default=COGNATE_BEAST_DIR,
        help="BEAST directory for cognates (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        default="fleurs-r",
        help="Dataset name (default: %(default)s)",
    )
    return parser.parse_args()


def extract_beast_rates(nex_path):
    """Parse a BEAST annotated nexus file, return {taxon_name: rate_median}."""
    with open(nex_path, "r", encoding="utf-8") as f:
        text = f.read()

    translate = {}
    in_translate = False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("Translate"):
            in_translate = True
            continue
        if in_translate:
            if s == ";":
                break
            parts = s.rstrip(",;").split()
            if len(parts) == 2:
                translate[parts[0]] = parts[1]

    tree_line = None
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"tree\s", s, re.IGNORECASE):
            tree_line = s
            break

    if tree_line is None:
        raise ValueError(f"No tree line found in {nex_path}")

    tip_pattern = re.compile(r"[(,](\w+)\[&([^\]]+)\]")
    rate_pattern = re.compile(r"rate_median=([0-9.eE+-]+)")

    rates = {}
    for match in tip_pattern.finditer(tree_line):
        tip_id = match.group(1)
        annotation = match.group(2)
        taxon = translate.get(tip_id, tip_id if not tip_id.isdigit() else None)
        if taxon is None:
            continue
        rate_match = rate_pattern.search(annotation)
        if rate_match:
            rates[taxon] = float(rate_match.group(1))

    return rates


def parse_taxa(path):
    taxa, inside = [], False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if "Taxlabels" in s or "taxlabels" in s:
                inside = True
                continue
            if inside:
                if s == ";":
                    break
                taxa.append(s)
    return taxa


if __name__ == "__main__":
    args = parse_args()

    speech_beast_dir = resolve_beast_dir(args.run_id, args.subdir)
    cognate_beast_dir = args.cognate_beast_dir
    metadata_dir = os.path.join(DEFAULT_METADATA_DIR, args.dataset)

    speech_mcc = find_beast_mcc(speech_beast_dir, version=args.version)
    cognate_mcc = find_beast_mcc(cognate_beast_dir)

    print(f"Speech BEAST dir: {speech_beast_dir}")
    print(f"  tree: {speech_mcc}")
    print(f"Cognate BEAST dir: {cognate_beast_dir}")
    print(f"  tree: {cognate_mcc}")

    TREE_CONFIG = {
        "speech": {
            "nex": speech_mcc,
            "stree6": os.path.join(speech_beast_dir, "__merged_splitstree.stree6"),
            "beast_dir": speech_beast_dir,
            "name_col": "fleurs",
        },
        "cognate": {
            "nex": cognate_mcc,
            "stree6": os.path.join(cognate_beast_dir, "__merged_splitstree.stree6"),
            "beast_dir": cognate_beast_dir,
            "name_col": "iecor",
        },
    }

    # Load sources
    with open(f"{metadata_dir}/languages.json", "r", encoding="utf-8") as f:
        langs = json.load(f)
    glottolog = pd.read_csv(f"{metadata_dir}/glottolog.csv")
    speakers = pd.read_csv(f"{metadata_dir}/n_speakers.csv")

    speech_taxa = set(parse_taxa(speech_mcc))
    cognate_taxa = set(parse_taxa(cognate_mcc))

    print(f"Speech taxa before in nexus: {len(speech_taxa)}")
    print(f"Cognate taxa before in nexus: {len(cognate_taxa)}")

    # remove Afrikaans and Kabuverdianu (out of European continent)
    speech_taxa = speech_taxa - {"Afrikaans", "Kabuverdianu"}
    cognate_taxa = cognate_taxa - {"Afrikaans", "Kabuverdianu"}

    # Build base metadata (no PHOIBLE yet)
    rows = []
    for key, v in langs.items():
        rows.append(
            dict(
                fleurs_dir=key,
                fleurs=v["fleurs"],
                iecor=v.get("iecor"),
                glottocode=v["glottolog"],
            )
        )
    meta = pd.DataFrame(rows)
    meta = meta.merge(
        glottolog[["fleurs_dir", "longitude", "latitude"]], on="fleurs_dir", how="left"
    )
    meta = meta.merge(
        speakers[["fleurs_dir", "speakers_linguameta"]], on="fleurs_dir", how="left"
    )
    meta = meta.rename(columns={"speakers_linguameta": "n_speakers"})

    # Filter taxa (version without inventory — before PHOIBLE)
    speech_df_no_inv = (
        meta[meta["fleurs"].isin(speech_taxa)][
            ["fleurs", "longitude", "latitude", "n_speakers"]
        ]
        .rename(columns={"fleurs": "language"})
        .sort_values("language")
        .reset_index(drop=True)
    )

    cognate_df_no_inv = meta[meta["iecor"].isin(cognate_taxa)].copy()
    print(cognate_df_no_inv[cognate_df_no_inv["iecor"] == "SerboCroatian"])
    # keep only croa1245 - Croatian row
    cognate_df_no_inv = cognate_df_no_inv[
        ~(
            (cognate_df_no_inv["iecor"] == "SerboCroatian")
            & (cognate_df_no_inv["glottocode"] != "croa1245")
        )
    ]
    cognate_df_no_inv = (
        cognate_df_no_inv[["iecor", "longitude", "latitude", "n_speakers"]]
        .rename(columns={"iecor": "language"})
        .sort_values("language")
        .reset_index(drop=True)
    )

    print(f"Speech taxa after in nexus: {len(speech_df_no_inv)}")
    print(f"Cognate taxa after in nexus: {len(cognate_df_no_inv)}")

    print(
        f"Speech taxa filtered out: {speech_taxa - set(speech_df_no_inv['language'])}"
    )
    print(
        f"Cognate taxa filtered out: {cognate_taxa - set(cognate_df_no_inv['language'])}"
    )

    # PHOIBLE: load pre-computed n_phonemes from summary CSV
    phoible = pd.read_csv(f"{metadata_dir}/phoible.csv")[["Glottocode", "n_phonemes"]]
    meta_inv = meta.merge(
        phoible, left_on="glottocode", right_on="Glottocode", how="left"
    ).drop(columns="Glottocode")

    # Filter taxa (version with inventory — after PHOIBLE)
    speech_df_inv = (
        meta_inv[meta_inv["fleurs"].isin(speech_taxa)][
            ["fleurs", "longitude", "latitude", "n_speakers", "n_phonemes"]
        ]
        .rename(columns={"fleurs": "language"})
        .sort_values("language")
        .reset_index(drop=True)
    )

    cognate_df_inv = meta_inv[meta_inv["iecor"].isin(cognate_taxa)].copy()
    cognate_df_inv = cognate_df_inv[
        ~(
            (cognate_df_inv["iecor"] == "SerboCroatian")
            & (cognate_df_inv["glottocode"] != "croa1245")
        )
    ]
    cognate_df_inv = (
        cognate_df_inv[["iecor", "longitude", "latitude", "n_speakers", "n_phonemes"]]
        .rename(columns={"iecor": "language"})
        .sort_values("language")
        .reset_index(drop=True)
    )

    # Report
    missing_inv = set(speech_df_inv[speech_df_inv["n_phonemes"].isna()]["language"])
    speech_inv_final = len(speech_df_inv) - len(missing_inv)
    print(f"Speech (no inv):   {len(speech_df_no_inv)}/{len(speech_taxa)} taxa matched")
    print(
        f"Speech (with inv): {speech_inv_final} languages "
        f"(dropped {len(missing_inv)} missing n_phonemes: {missing_inv or 'none'})"
    )

    missing_inv_c = set(cognate_df_inv[cognate_df_inv["n_phonemes"].isna()]["language"])
    cognate_inv_final = len(cognate_df_inv) - len(missing_inv_c)
    print(f"\nCognate (no inv):   {len(cognate_df_no_inv)} languages")
    print(
        f"Cognate (with inv): {cognate_inv_final} languages "
        f"(dropped {len(missing_inv_c)} missing n_phonemes: {missing_inv_c or 'none'})"
    )

    # Merge rate_median and delta, save both versions
    datasets = {
        "speech": (speech_df_no_inv, speech_df_inv),
        "cognate": (cognate_df_no_inv, cognate_df_inv),
    }

    for stem, (df_no_inv, df_inv) in datasets.items():
        cfg = TREE_CONFIG[stem]
        beast_dir = cfg["beast_dir"]

        rates = extract_beast_rates(cfg["nex"])
        delta_df = extract_delta(cfg["stree6"])
        delta_map = delta_df["delta.score"].to_dict()

        df_no_inv = df_no_inv.copy()
        df_no_inv["rate_median"] = df_no_inv["language"].map(rates)
        df_no_inv["delta"] = df_no_inv["language"].map(delta_map)
        df_no_inv = df_no_inv.dropna(
            subset=["n_speakers", "rate_median", "delta"]
        ).reset_index(drop=True)

        df_inv = df_inv.copy()
        df_inv["rate_median"] = df_inv["language"].map(rates)
        df_inv["delta"] = df_inv["language"].map(delta_map)
        df_inv = df_inv.dropna(
            subset=["n_speakers", "n_phonemes", "rate_median", "delta"]
        ).reset_index(drop=True)

        path_no_inv = f"{beast_dir}/metadata.csv"
        path_inv = f"{beast_dir}/metadata_with_inventory.csv"
        df_no_inv.to_csv(path_no_inv, index=False)
        df_inv.to_csv(path_inv, index=False)

        print(f"\n--- {stem} ---")
        print(f"  without inventory ({len(df_no_inv)} languages) -> {path_no_inv}")
        print(f"  with inventory    ({len(df_inv)} languages) -> {path_inv}")

    print("\nDone.")
