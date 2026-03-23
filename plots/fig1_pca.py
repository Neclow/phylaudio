#!/usr/bin/env python3
"""Extended Figure: PCA of XLS-R embeddings."""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D

from src.tasks.feature_extraction._decomposition import decompose, fit_decomposer

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_DIR = "img/fig1"
EMB_DIR = "data/embeddings/fleurs-r/facebook_wav2vec2-xls-r-300m"

PALETTE_MAP = {
    "germanic": "Reds_r",
    "celtic": ["orange", "darkorange"],
    "indoaryan": "Purples",
    "slavic": "Greens_r",
    "baltic": "blend:mediumpurple,lavender",
    "romance": "blend:darkkhaki,olive",
    "iranian": "blend:midnightblue,steelblue",
    "greek": ["gold"],
    "armenian": ["cyan"],
}

TAXONSET_ORDER = [
    "germanic", "celtic", "slavic", "romance",
    "indoaryan", "iranian", "baltic", "greek", "armenian",
]

TAXONSET_DISPLAY = {
    "germanic": "Germanic", "celtic": "Celtic", "slavic": "Slavic",
    "romance": "Romance", "indoaryan": "Indo-Aryan", "iranian": "Iranian",
    "baltic": "Baltic", "greek": "Greek", "armenian": "Armenian",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def clear_axes(ax=None, top=True, right=True, left=False, bottom=False, minorticks_off=True):
    if ax is None:
        axes = plt.gcf().axes
    else:
        axes = [ax]
    for ax_i in axes:
        sns.despine(ax=ax_i, top=top, right=right, left=left, bottom=bottom)
        if minorticks_off:
            ax_i.minorticks_off()
        ax_i.tick_params(axis="x", which="both", top=not top)
        ax_i.tick_params(axis="y", which="both", right=not right)
        ax_i.tick_params(axis="y", which="both", left=not left)
        ax_i.tick_params(axis="x", which="both", bottom=not bottom)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    X_emb = torch.load(f"{EMB_DIR}/embeddings.pt", map_location=device)
    y_emb = torch.load(f"{EMB_DIR}/labels.pt", map_location=device)

    with open("data/metadata/fleurs-r/labels.txt") as f:
        all_labels = [l.strip().split(" => ")[0].strip("'") for l in f.readlines()]

    with open("data/metadata/fleurs-r/languages.json") as f:
        mapping = json.load(f)

    labels_pca = {i: code for i, code in enumerate(all_labels) if i in y_emb.unique().tolist()}

    # Fit PCA
    decomposer = fit_decomposer(X_emb, method="pca", n_components=0.99, standardize=True, device=device, seed=42)
    X_pca = decompose(decomposer, X_emb)
    var_exp = decomposer.explained_variance_ratio_

    # Build color map
    taxonset_counts = defaultdict(int)
    for lang_code in labels_pca.values():
        ts = mapping[lang_code]["taxonset"]
        taxonset_counts[ts] += 1

    hex_map = {}
    for ts, pal in PALETTE_MAP.items():
        n = max(taxonset_counts.get(ts, 1), 1)
        hex_map[ts] = sns.color_palette(pal, n_colors=n).as_hex()

    counts = defaultdict(int)
    color_map = {}
    for yi in y_emb.cpu().unique().tolist():
        lang_code = labels_pca[yi]
        taxonset = mapping[lang_code]["taxonset"]
        palette = hex_map[taxonset]
        color_map[yi] = palette[counts[taxonset]]
        counts[taxonset] += 1

    return X_pca, y_emb, var_exp, labels_pca, mapping, color_map, hex_map


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(X_pca, y_emb, var_exp, labels_pca, mapping, color_map, hex_map):
    with plt.style.context(".matplotlib/paper.mplstyle"):
        fig, ax = plt.subplots(figsize=(6, 6))

        rng = np.random.default_rng(42)
        idxs = rng.choice(X_pca.shape[0], size=min(15000, X_pca.shape[0]), replace=False)

        for taxonset in TAXONSET_ORDER:
            for class_id in y_emb.cpu().unique().tolist():
                lang_code = labels_pca[class_id]
                if mapping[lang_code]["taxonset"] != taxonset:
                    continue

                class_mask = y_emb == class_id
                class_idxs = [i for i in idxs if class_mask[i]]
                if not class_idxs:
                    continue

                ax.scatter(
                    X_pca[class_idxs, 0].cpu().numpy(),
                    X_pca[class_idxs, 1].cpu().numpy(),
                    s=3, alpha=0.15, color=color_map[class_id],
                    rasterized=True,
                )

        # Centroids with labels
        for class_id in y_emb.cpu().unique().tolist():
            class_mask = y_emb == class_id
            centroid = X_pca[class_mask, :2].mean(dim=0).cpu().numpy()
            lang_code = labels_pca[class_id]
            lang_name = mapping[lang_code]["fleurs"].split(" ")[0]

            ax.scatter(
                centroid[0], centroid[1], s=40,
                color=color_map[class_id],
                edgecolor="k", linewidth=0.5, zorder=5,
            )
            ax.annotate(
                lang_name, (centroid[0], centroid[1]),
                fontsize=5, ha="center", va="bottom",
                xytext=(0, 4), textcoords="offset points",
            )

        legend_handles = [
            Line2D([], [], marker="o", color=hex_map[ts][0], linestyle="",
                   markersize=5, label=TAXONSET_DISPLAY[ts])
            for ts in TAXONSET_ORDER if ts in hex_map
        ]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=7)

        ax.set_xlabel(f"PC 1 ({var_exp[0]*100:.1f}%)")
        ax.set_ylabel(f"PC 2 ({var_exp[1]*100:.1f}%)")
        clear_axes(ax)
        plt.savefig(f"{IMG_DIR}/ext_fig1_pca_xlsr.pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{IMG_DIR}/ext_fig1_pca_xlsr.svg", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    X_pca, y_emb, var_exp, labels_pca, mapping, color_map, hex_map = load_data()
    print(f"PCA: {X_pca.shape}, PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}%")
    plot(X_pca, y_emb, var_exp, labels_pca, mapping, color_map, hex_map)
