#!/usr/bin/env python3
"""Figure 1 Panel B: NMF structure plot (K=12)."""

import h5py
import matplotlib.pyplot as plt
import numpy as np

from src.tasks.phylo.nmf import normalize_rows_to_proportions

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_DIR = "img/fig1"
# NMF HDF5 is too large for git; kept in the original run directory (external archive)
NMF_H5_PATH = "data/trees/beast/speech/0.01_brsupport/nmf/sweep_k2_k30.h5"

# ggthemes Tableau 20 palette (R order)
TABLEAU20 = [
    "#4E79A7",
    "#A0CBE8",
    "#F28E2B",
    "#FFBE7D",
    "#59A14F",
    "#8CD17D",
    "#B6992D",
    "#F1CE63",
    "#499894",
    "#86BCB6",
    "#E15759",
    "#FF9D9A",
    "#79706E",
    "#BAB0AC",
    "#D37295",
    "#FABFD2",
    "#B07AA1",
    "#D4A6C8",
    "#9D7660",
    "#D7B5A6",
]

COMP_LABELS = {
    1: "NW. European",
    2: "W. South Asian",
    3: "E. South Asian",
    4: "E. Slavic",
    5: "Italo-Lusitanic",
    6: "W. Balkan",
    7: "Iberian",
    8: "E. Balkan",
    9: "Iran. Plateau",
    10: "British-Irish Isles",
    11: "Scandinavian",
    12: "C. European",
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_nmf(h5_path=NMF_H5_PATH):
    with h5py.File(h5_path, "r") as f:
        labels = [l.decode() for l in f["labels"][()]]
        W = f["K12"]["W"][()]

    K = W.shape[1]
    P = normalize_rows_to_proportions(W)

    # Sort languages: by dominant component, then by its proportion (descending)
    max_comp = np.argmax(P, axis=1)
    max_val = P[np.arange(len(labels)), max_comp]
    order = np.lexsort((-max_val, max_comp))

    P_sorted = P[order]
    labels_sorted = np.array(labels)[order]
    return P_sorted, labels_sorted, K


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(P_sorted, labels_sorted, K):
    with plt.style.context(".matplotlib/paper.mplstyle"):
        fig, ax = plt.subplots(figsize=(3.25, 8))

        colors = TABLEAU20[:K][::-1]
        n_lang = len(labels_sorted)
        y = np.arange(n_lang)
        left = np.zeros(n_lang)

        for j in range(K):
            ax.barh(
                y,
                P_sorted[:, j],
                left=left,
                height=1.0,
                color=colors[j],
                label=COMP_LABELS[j + 1],
                edgecolor="none",
            )
            left += P_sorted[:, j]

        ax.set_ylim(-0.5, n_lang - 0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_sorted, fontsize=7.5, alpha=0.8)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", color=(0, 0, 0, 0.3), pad=1)
        ax.invert_yaxis()

        # X-axis at top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(
            "Component proportion", alpha=0.7, fontdict={"size": 8.5}, labelpad=6
        )
        for label in ax.get_xticklabels():
            label.set_alpha(0.7)
            label.set_fontsize(7.5)

        for spine in ax.spines.values():
            spine.set_visible(False)

        leg = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.005),
            fontsize=7.5,
            ncol=2,
            title="Acoustic components",
            title_fontproperties={"weight": "bold", "size": 8},
            columnspacing=0.6,
            handletextpad=0.3,
            labelspacing=0.15,
            alignment="left",
        )
        leg.get_title().set_position((0, 1))
        ax.grid(axis="x", linestyle="dashed", alpha=0.1, color="k")

        plt.savefig(
            f"{IMG_DIR}/fig1b_nmf_structure_K12.pdf",
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.show()


if __name__ == "__main__":
    import os

    os.makedirs(IMG_DIR, exist_ok=True)
    P_sorted, labels_sorted, K = load_nmf()
    print(f"Loaded W matrix: {len(labels_sorted)} languages x {K} components")
    plot(P_sorted, labels_sorted, K)
