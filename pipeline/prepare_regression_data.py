import pandas as pd
import json
import re

from src.tasks.phylo.splitstree import extract_delta

base = "data/metadata/fleurs-r"
mcc_speech = "data/trees/beast/input_v12_combined_resampled.mcc"
mcc_cognate = "data/trees/references/raw/iecor.nex"

TREE_CONFIG = {
    "input_v12_combined_resampled": {
        "nex": mcc_speech,
        "stree6": "data/trees/beast/dd208931-4817-41ad-b18d-aa6a050a3f42/0.01_brsupport/__merged_splitstree.stree6",
        "name_col": "fleurs",
    },
    "heggarty2024_raw": {
        "nex": mcc_cognate,
        "stree6": "data/trees/beast/iecor/__merged_splitstree.stree6",
        "name_col": "iecor",
    },
}

# ─── Helper: extract rate_median from BEAST annotated nexus ──────────────────

def extract_beast_rates(nex_path):
    """Parse a BEAST annotated nexus file, return {taxon_name: rate_median}."""
    with open(nex_path) as f:
        text = f.read()

    # Parse Translate block: number -> taxon name
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

    # Find the tree line
    tree_line = None
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"tree\s", s, re.IGNORECASE):
            tree_line = s
            break

    if tree_line is None:
        raise ValueError(f"No tree line found in {nex_path}")

    # Extract tip annotations: <id>[&annotations]
    # id can be a number (with Translate block) or a taxon name (without)
    tip_pattern = re.compile(r"[(,](\w+)\[&([^\]]+)\]")
    rate_pattern = re.compile(r"rate_median=([0-9.eE+-]+)")

    rates = {}
    for match in tip_pattern.finditer(tree_line):
        tip_id = match.group(1)
        annotation = match.group(2)
        # Resolve via Translate block if available, otherwise use name directly
        taxon = translate.get(tip_id, tip_id if not tip_id.isdigit() else None)
        if taxon is None:
            continue
        rate_match = rate_pattern.search(annotation)
        if rate_match:
            rates[taxon] = float(rate_match.group(1))

    return rates


# ─── Parse taxa from nex files ───────────────────────────────────────────────

def parse_taxa(path):
    taxa, inside = [], False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if "Taxlabels" in s or "taxlabels" in s: inside = True; continue
            if inside:
                if s == ";": break
                taxa.append(s)
    return taxa


# ─── Load sources ────────────────────────────────────────────────────────────

with open(f"{base}/languages.json") as f:
    langs = json.load(f)
glottolog = pd.read_csv(f"{base}/glottolog.csv")
speakers = pd.read_csv(f"{base}/n_speakers.csv")

speech_taxa = set(parse_taxa(mcc_speech))
cognate_taxa = set(parse_taxa(mcc_cognate))


# number of languages before
print(f"Speech taxa before in nexus: {len(speech_taxa)}")
print(f"Cognate taxa before in nexus: {len(cognate_taxa)}")

# remove Afrikaans and Kabuverdianu
speech_taxa = speech_taxa - {"Afrikaans", "Kabuverdianu"}
cognate_taxa = cognate_taxa - {"Afrikaans", "Kabuverdianu"}

# ─── Build base metadata (no PHOIBLE yet) ────────────────────────────────────

rows = []
for key, v in langs.items():
    rows.append(dict(fleurs_dir=key, fleurs=v["fleurs"], iecor=v.get("iecor"),
                     glottocode=v["glottolog"]))
meta = pd.DataFrame(rows)
meta = meta.merge(glottolog[["fleurs_dir", "longitude", "latitude"]], on="fleurs_dir", how="left")
meta = meta.merge(speakers[["fleurs_dir", "speakers_linguameta"]], on="fleurs_dir", how="left")
meta = meta.rename(columns={"speakers_linguameta": "n_speakers"})

# ─── Filter taxa (version without inventory — before PHOIBLE) ────────────────

speech_df_no_inv = (
    meta[meta["fleurs"].isin(speech_taxa)][["fleurs", "longitude", "latitude", "n_speakers"]]
    .rename(columns={"fleurs": "language"})
    .sort_values("language").reset_index(drop=True)
)

cognate_df_no_inv = meta[meta["iecor"].isin(cognate_taxa)].copy()
# print rows of serbocroatian
print(cognate_df_no_inv[cognate_df_no_inv["iecor"] == "SerboCroatian"])
# keep only croa1245 - Croatian row
cognate_df_no_inv = cognate_df_no_inv[
    ~((cognate_df_no_inv["iecor"] == "SerboCroatian") & (cognate_df_no_inv["glottocode"] != "croa1245"))
]
cognate_df_no_inv = (
    cognate_df_no_inv[["iecor", "longitude", "latitude", "n_speakers"]]
    .rename(columns={"iecor": "language"})
    .sort_values("language").reset_index(drop=True)
)

# after merging with glottolog
print(f"Speech taxa after in nexus: {len(speech_df_no_inv)}")
print(f"Cognate taxa after in nexus: {len(cognate_df_no_inv)}")

# which languages were filtered out compared to set
print(f"Speech taxa filtered out: {speech_taxa - set(speech_df_no_inv['language'])}")
print(f"Cognate taxa filtered out: {cognate_taxa - set(cognate_df_no_inv['language'])}")

# ─── PHOIBLE: load pre-computed n_phonemes from summary CSV ───────────────────

phoible = pd.read_csv(f"{base}/phoible.csv")[["Glottocode", "n_phonemes"]]
meta_inv = meta.merge(phoible, left_on="glottocode", right_on="Glottocode", how="left").drop(columns="Glottocode")

# ─── Filter taxa (version with inventory — after PHOIBLE) ────────────────────

speech_df_inv = (
    meta_inv[meta_inv["fleurs"].isin(speech_taxa)][["fleurs", "longitude", "latitude", "n_speakers", "n_phonemes"]]
    .rename(columns={"fleurs": "language"})
    .sort_values("language").reset_index(drop=True)
)

cognate_df_inv = meta_inv[meta_inv["iecor"].isin(cognate_taxa)].copy()
cognate_df_inv = cognate_df_inv[
    ~((cognate_df_inv["iecor"] == "SerboCroatian") & (cognate_df_inv["glottocode"] != "croa1245"))
]
cognate_df_inv = (
    cognate_df_inv[["iecor", "longitude", "latitude", "n_speakers", "n_phonemes"]]
    .rename(columns={"iecor": "language"})
    .sort_values("language").reset_index(drop=True)
)

# ─── Report ──────────────────────────────────────────────────────────────────

missing_inv = set(speech_df_inv[speech_df_inv["n_phonemes"].isna()]["language"])
speech_inv_final = len(speech_df_inv) - len(missing_inv)
print(f"Speech (no inv):   {len(speech_df_no_inv)}/{len(speech_taxa)} taxa matched")
print(f"Speech (with inv): {speech_inv_final} languages (dropped {len(missing_inv)} missing n_phonemes: {missing_inv or 'none'})")

missing_inv_c = set(cognate_df_inv[cognate_df_inv["n_phonemes"].isna()]["language"])
cognate_inv_final = len(cognate_df_inv) - len(missing_inv_c)
print(f"\nCognate (no inv):   {len(cognate_df_no_inv)} languages")
print(f"Cognate (with inv): {cognate_inv_final} languages (dropped {len(missing_inv_c)} missing n_phonemes: {missing_inv_c or 'none'})")

# ─── Merge rate_median and delta, save both versions ─────────────────────────

datasets = {
    "input_v12_combined_resampled": ("speech",  speech_df_no_inv,  speech_df_inv),
    "heggarty2024_raw": ("cognate", cognate_df_no_inv, cognate_df_inv),
}

for tree_name, (stem, df_no_inv, df_inv) in datasets.items():
    cfg = TREE_CONFIG[tree_name]

    rates = extract_beast_rates(cfg["nex"])
    delta_df = extract_delta(cfg["stree6"])
    delta_map = delta_df["delta.score"].to_dict()

    # ── Version without inventory (pre-PHOIBLE) ──
    df_no_inv = df_no_inv.copy()
    df_no_inv["rate_median"] = df_no_inv["language"].map(rates)
    df_no_inv["delta"]       = df_no_inv["language"].map(delta_map)
    df_no_inv = df_no_inv.dropna(subset=["n_speakers", "rate_median", "delta"]).reset_index(drop=True)

    # ── Version with inventory (post-PHOIBLE) ──
    df_inv = df_inv.copy()
    df_inv["rate_median"] = df_inv["language"].map(rates)
    df_inv["delta"]       = df_inv["language"].map(delta_map)
    df_inv = df_inv.dropna(subset=["n_speakers", "n_phonemes", "rate_median", "delta"]).reset_index(drop=True)

    path_no_inv = f"data/phyloregression/{stem}_metadata.csv"
    path_inv    = f"data/phyloregression/{stem}_metadata_with_inventory.csv"
    df_no_inv.to_csv(path_no_inv, index=False)
    df_inv.to_csv(path_inv, index=False)

    print(f"\n--- {tree_name} ---")
    print(f"  without inventory ({len(df_no_inv)} languages) -> {path_no_inv}")
    print(f"  with inventory    ({len(df_inv)} languages) -> {path_inv}")
