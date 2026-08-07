"""Extended Figure: PCA of XLS-R embeddings."""

# pylint: disable=redefined-outer-name, invalid-name

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D

from src._config import DEFAULT_METADATA_DIR
from src.data.glottolog import filter_languages
from src.tasks.feature_extraction._decomposition import decompose, fit_decomposer
from src.tasks.plot import clear_axes

from ._config import (
    DATASET,
    DEFAULT_IMG_DIR,
    DEFAULT_STYLE,
    TAXONSET_DISPLAY,
    TAXONSET_ORDER,
    TAXONSET_PALETTE,
    XLS_R_EMBEDDING_DIR,
)

IMG_DIR = f"{DEFAULT_IMG_DIR}/fig1"


# Data loading
def load_data():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    X_emb = torch.load(f"{XLS_R_EMBEDDING_DIR}/embeddings.pt", map_location=device)
    y_emb = torch.load(f"{XLS_R_EMBEDDING_DIR}/labels.pt", map_location=device)

    with open(
        f"{DEFAULT_METADATA_DIR}/{DATASET}/labels.txt", "r", encoding="utf-8"
    ) as f:
        all_labels = [l.strip().split(" => ")[0].strip("'") for l in f.readlines()]

    ie_languages = filter_languages(DATASET, "indo1319")

    all_labels_dict = {
        i: code for i, code in enumerate(all_labels) if i in y_emb.unique().tolist()
    }
    ie_ids = {i for i, code in all_labels_dict.items() if code in ie_languages}
    ie_mask = torch.tensor([int(y) in ie_ids for y in y_emb], device=device)
    X_emb = X_emb[ie_mask]
    y_emb = y_emb[ie_mask]
    labels_pca = {i: all_labels_dict[i] for i in ie_ids}

    # Fit PCA
    decomposer = fit_decomposer(
        X_emb, method="pca", n_components=0.99, standardize=True, device=device, seed=42
    )
    X_pca = decompose(decomposer, X_emb)
    var_exp = decomposer.explained_variance_ratio_

    # Build color map
    taxonset_counts = defaultdict(int)
    for lang_code in labels_pca.values():
        ts = ie_languages[lang_code]["taxonset"]
        taxonset_counts[ts] += 1

    hex_map = {}
    for ts, pal in TAXONSET_PALETTE.items():
        n = max(taxonset_counts.get(ts, 1), 1)
        hex_map[ts] = sns.color_palette(pal, n_colors=n).as_hex()

    counts = defaultdict(int)
    color_map = {}
    for yi in y_emb.cpu().unique().tolist():
        lang_code = labels_pca[yi]
        taxonset = ie_languages[lang_code]["taxonset"]
        palette = hex_map[taxonset]
        color_map[yi] = palette[counts[taxonset]]
        counts[taxonset] += 1

    return X_pca, y_emb, var_exp, labels_pca, ie_languages, color_map, hex_map


# Plot
def plot_pca(X_pca, y_emb, var_exp, labels_pca, mapping, color_map, hex_map):
    with plt.style.context(DEFAULT_STYLE):
        _, ax = plt.subplots(figsize=(6, 6))

        rng = np.random.default_rng(42)
        idxs = rng.choice(
            X_pca.shape[0], size=min(15000, X_pca.shape[0]), replace=False
        )

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
                    s=3,
                    alpha=0.15,
                    color=color_map[class_id],
                    rasterized=True,
                )

        # Centroids with labels
        for class_id in y_emb.cpu().unique().tolist():
            class_mask = y_emb == class_id
            centroid = X_pca[class_mask, :2].mean(dim=0).cpu().numpy()
            lang_code = labels_pca[class_id]
            lang_name = mapping[lang_code]["fleurs"].split(" ")[0]

            ax.scatter(
                centroid[0],
                centroid[1],
                s=40,
                color=color_map[class_id],
                edgecolor="k",
                linewidth=0.5,
                zorder=5,
            )
            ax.annotate(
                lang_name,
                (centroid[0], centroid[1]),
                fontsize=5,
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )

        legend_handles = [
            Line2D(
                [],
                [],
                marker="o",
                color=hex_map[ts][0],
                linestyle="",
                markersize=5,
                label=TAXONSET_DISPLAY[ts],
            )
            for ts in TAXONSET_ORDER
            if ts in hex_map
        ]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=7)

        ax.set_xlabel(f"PC 1 ({var_exp[0]*100:.1f}%)")
        ax.set_ylabel(f"PC 2 ({var_exp[1]*100:.1f}%)")
        clear_axes(ax)
        stem = f"{IMG_DIR}/figS2a_pca_xlsr"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{stem}.svg", bbox_inches="tight")
        print(f"Saved {stem}.{{pdf,svg}}")
        plt.show()


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    X_pca, y_emb, var_exp, labels_pca, mapping, color_map, hex_map = load_data()
    print(f"PCA: {X_pca.shape[0]} samples, {X_pca.shape[1]} components (99% var), PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}%")
    plot_pca(X_pca, y_emb, var_exp, labels_pca, mapping, color_map, hex_map)
