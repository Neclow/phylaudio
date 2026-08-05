"""Figure 1d: per-language delta scores colored by NMF component proportions."""

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._config import BEAST_DIR, DEFAULT_IMG_DIR, DEFAULT_STYLE, NMF_COMP_ORDER, PALETTE

IMG_DIR = f"{DEFAULT_IMG_DIR}/fig1"
NMF_DIR = f"{BEAST_DIR}/nmf"
DELTA_CSV = f"{BEAST_DIR}/_delta.csv"


def load_data():
    hits = sorted(glob(f"{NMF_DIR}/Q_K*.csv"))
    if not hits:
        raise FileNotFoundError(f"No Q_K*.csv found in {NMF_DIR}/")
    df_q = pd.read_csv(hits[0])
    nmf_labels = df_q["language"].values
    P = df_q.drop(columns=["language"]).values
    K = P.shape[1]

    P = P[:, NMF_COMP_ORDER]
    lang_to_props = {lang: P[i] for i, lang in enumerate(nmf_labels)}

    delta_df = pd.read_csv(DELTA_CSV)
    delta_df = delta_df.sort_values("delta", ascending=True).reset_index(drop=True)

    return delta_df, lang_to_props


def plot(delta_df, lang_to_props):
    delta_mean = delta_df.delta.mean()
    has_ci = "ci_lo" in delta_df.columns and delta_df.ci_lo.notna().all()
    K = len(PALETTE)
    colors = PALETTE[:K]
    ymin = 0.25

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.set_axisbelow(True)

        x = np.arange(len(delta_df))
        visible_h = (delta_df.delta.values - ymin).clip(0)
        bottom = np.full(len(delta_df), ymin)
        for j in range(K):
            props = np.array([
                lang_to_props.get(lang, np.zeros(K))[j]
                for lang in delta_df.language
            ])
            heights = visible_h * props
            ax.bar(
                x, heights, bottom=bottom, color=colors[j],
                edgecolor="none", width=0.7,
            )
            bottom += heights

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

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        output_path = f"{IMG_DIR}/fig1d_delta_speech.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
        plt.show()


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    delta_df, lang_to_props = load_data()
    has_ci = "ci_lo" in delta_df.columns and delta_df.ci_lo.notna().all()
    print(f"Delta: {len(delta_df)} languages, mean={delta_df.delta.mean():.4f}")
    if has_ci:
        print("Bootstrap CIs loaded (95%)")
    plot(delta_df, lang_to_props)
