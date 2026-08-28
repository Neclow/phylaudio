"""Diagnostic: scatter of within-language variance vs. log(1 + pre-training hours)."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from plots._config import DEFAULT_STYLE
from src._config import _FLEURS_LANG_TO_ID

POP_TRAITS_FILE = "data/trees/continuous/gelu/v7zqs3lv/population_traits.pt"
POP_LABELS_FILE = "data/trees/continuous/gelu/v7zqs3lv/population_labels.json"
HOURS_FILE = "tmp/dataset_hours_comparison.csv"
IMG_DIR = "img_v3/fig2"

NAME_TO_ISO = {name: code for name, code in _FLEURS_LANG_TO_ID.items()}
# FLEURS uses nb (Bokmål) but the hours CSV uses no (Norwegian macro)
_ISO_ALIAS = {"nb": "no"}


def compute_within_lang_variance(pop_traits_file, pop_labels_file):
    X = torch.load(pop_traits_file, weights_only=True).numpy()
    with open(pop_labels_file) as f:
        labels = json.load(f)
    label_arr = np.array(labels)
    variances = {}
    for lang in sorted(set(labels)):
        mask = label_arr == lang
        variances[lang] = np.mean(np.var(X[mask], axis=0, ddof=1))
    return variances


def load_pretrain_hours():
    df = pd.read_csv(HOURS_FILE)
    return dict(zip(df["code"], df["hours_xlsr_pretrain"].fillna(0)))


if __name__ == "__main__":
    within_var = compute_within_lang_variance(POP_TRAITS_FILE, POP_LABELS_FILE)
    pretrain_hours = load_pretrain_hours()

    rows = []
    for lang in sorted(within_var):
        iso = NAME_TO_ISO.get(lang)
        if iso is None:
            continue
        hours = pretrain_hours.get(_ISO_ALIAS.get(iso, iso), 0.0)
        rows.append({
            "language": lang,
            "within_var": within_var[lang],
            "pretrain_hours": hours,
            "log_pretrain_hours": np.log1p(hours),
        })

    df = pd.DataFrame(rows).dropna()
    print(f"{len(df)} languages")

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df["log_pretrain_hours"], df["within_var"], s=20, zorder=3)

        for _, row in df.iterrows():
            ax.annotate(
                row["language"],
                (row["log_pretrain_hours"], row["within_var"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=5,
                color="0.3",
            )

        ax.set_xlabel("log(1 + pre-training hours)")
        ax.set_ylabel("Within-language variance")
        fig.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/diag_var_vs_hours.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved to {IMG_DIR}/diag_var_vs_hours.{{pdf,svg}}")
        plt.close(fig)
