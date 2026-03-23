#!/usr/bin/env python3
"""Figure 1 Panel D: Contribution to network structure per language (delta)."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.tasks.phylo.nmf import normalize_rows_to_proportions

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_DIR = "img/fig1"
BEAST_DIR = "data/trees/beast/speech/0.01_brsupport"
DELTA_CSV_PATH = f"{BEAST_DIR}/_delta.csv"
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def clear_axes(
    ax=None, top=True, right=True, left=False, bottom=False, minorticks_off=True
):
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
    # Load NMF for color assignment
    with h5py.File(NMF_H5_PATH, "r") as f:
        nmf_labels = [l.decode() for l in f["labels"][()]]
        W = f["K12"]["W"][()]

    K = W.shape[1]
    P = normalize_rows_to_proportions(W)
    max_comp_per_lang = np.argmax(P, axis=1)
    colors = TABLEAU20[:K][::-1]
    lang_to_color = {
        lang: colors[max_comp_per_lang[i]] for i, lang in enumerate(nmf_labels)
    }

    # Load delta scores
    delta_df = pd.read_csv(DELTA_CSV_PATH)
    delta_df = delta_df.sort_values("delta", ascending=True).reset_index(drop=True)
    delta_df["color"] = delta_df.language.map(lang_to_color).fillna("grey50")

    return delta_df


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(delta_df):
    delta_mean = delta_df.delta.mean()
    has_ci = "ci_lo" in delta_df.columns and delta_df.ci_lo.notna().all()

    with plt.style.context(".matplotlib/paper.mplstyle"):
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.set_axisbelow(True)

        x = np.arange(len(delta_df))
        ax.bar(x, delta_df.delta, color=delta_df.color, edgecolor="none", width=0.7)

        if has_ci:
            ax.errorbar(
                x,
                delta_df.delta,
                yerr=[delta_df.delta - delta_df.ci_lo, delta_df.ci_hi - delta_df.delta],
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=1.0,
                zorder=3,
            )

        ax.axhline(
            delta_mean, color="firebrick", linestyle="--", linewidth=0.6, zorder=4
        )

        q025 = delta_df.delta.quantile(0.025)
        q975 = delta_df.delta.quantile(0.975)
        ax.axhline(q025, color="black", linestyle=":", linewidth=0.5, zorder=4)
        ax.axhline(q975, color="black", linestyle=":", linewidth=0.5, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(
            delta_df.language,
            rotation=55,
            ha="right",
            fontsize=8,
            alpha=0.8,
            rotation_mode="anchor",
        )
        ylabel = r"$\bar{\delta}$ (95% bootstrap CI)" if has_ci else r"$\delta$"
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.5, len(delta_df) - 0.5)
        ax.set_ylim(0.25)
        ax.tick_params(axis="both", length=3, width=0.5)
        ax.grid(axis="y", alpha=0.15)
        for label in ax.get_yticklabels():
            label.set_alpha(0.8)

        clear_axes(ax)
        plt.savefig(f"{IMG_DIR}/fig1d_delta_speech.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    import os

    os.makedirs(IMG_DIR, exist_ok=True)
    delta_df = load_data()
    has_ci = "ci_lo" in delta_df.columns and delta_df.ci_lo.notna().all()
    print(f"Delta: {len(delta_df)} languages, mean={delta_df.delta.mean():.4f}")
    if has_ci:
        print("Bootstrap CIs loaded (95%)")
    plot(delta_df)
