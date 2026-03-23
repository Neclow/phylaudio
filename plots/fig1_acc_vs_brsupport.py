#!/usr/bin/env python3
"""Figure 1 Panel A: LID accuracy vs. mean bootstrap support."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import seaborn as sns
from matplotlib.lines import Line2D

from src._config import DEFAULT_EVAL_DIR, DEFAULT_PER_SENTENCE_DIR

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_DIR = "img/fig1"

MODEL_DETAILS = {
    "openai/whisper-small": {"arch": "Whisper", "nparam": 240.6},
    "facebook/wav2vec2-xls-r-300m": {"arch": "wav2vec2", "nparam": 315.4},
    "openai/whisper-base": {"arch": "Whisper", "nparam": 71.8},
    "openai/whisper-tiny": {"arch": "Whisper", "nparam": 37.2},
    "facebook/mms-lid-256": {"arch": "wav2vec2", "nparam": 964.6},
    "openai/whisper-medium": {"arch": "Whisper", "nparam": 762.3},
    "speechbrain/lang-id-voxlingua107-ecapa": {"arch": "Other (CNN)", "nparam": 21.2},
    "NeMo_ambernet": {"arch": "Other (CNN)", "nparam": 28.9},
    "baseline/CNN6": {"arch": "Other (CNN)", "nparam": 1.2},
    "baseline/CNN10": {"arch": "Other (CNN)", "nparam": 4.7},
    "openai/whisper-large-v3-turbo": {"arch": "Whisper", "nparam": 807.0},
    "facebook/mms-lid-126": {"arch": "wav2vec2", "nparam": 964.6},
    "facebook/mms-lid-4017": {"arch": "wav2vec2", "nparam": 964.6},
    "mms-meta/mms-zeroshot-300m": {"arch": "wav2vec2", "nparam": 315.4},
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


def align_legend_title(fig, leg):
    """Shift the legend title right so it aligns with the handle markers."""
    fig.canvas.draw()
    offset = leg.handlelength * plt.rcParams["font.size"] / 2
    title = leg.get_title()
    title.set_position((offset, 0))


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    # Load accuracy data
    phylo_summary = pd.read_csv(
        f"{DEFAULT_PER_SENTENCE_DIR}/discrete/summary.csv", index_col=0
    ).query("min_speakers == 0.0")
    id2model = phylo_summary.model_id.to_dict()

    eval_summary = pd.read_json(
        f"{DEFAULT_EVAL_DIR}/summary.json", orient="index"
    ).reset_index(names="run_id")
    eval_summary.model_size /= 1e6

    # Load per-sentence brsupport, average per model
    stat_dfs = []
    for run_id, model_id in id2model.items():
        f = f"{DEFAULT_PER_SENTENCE_DIR}/discrete/{run_id}/_stats.csv"
        df = pd.read_csv(f, index_col=0).query("Ntips > 38")
        if df.empty:
            continue
        df_mean = df.drop("Ntips", axis=1).mean()
        df_mean["run_id"] = run_id
        df_mean["model_id"] = model_id
        stat_dfs.append(df_mean)

    stat_df = pd.DataFrame(stat_dfs)

    # Model metadata
    model_details = pd.DataFrame.from_dict(MODEL_DETAILS, orient="index").sort_index()

    # Merge
    merged = (
        eval_summary
        .merge(model_details, left_on="model_id", right_index=True)
        .merge(stat_df, on="model_id")
        .dropna(subset=["brsupport"])
    )
    merged["test_accuracy"] *= 100
    return merged


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(merged):
    with plt.style.context(".matplotlib/paper.mplstyle"):
        fig, ax = plt.subplots(figsize=(3.5, 3))

        base = 8
        anchors = np.array([1, 30, 300, 1000])
        anchor_sizes = np.array([base, 3 * base, 9 * base, 27 * base])

        size_func = interp.interp1d(np.log10(anchors), anchor_sizes, fill_value="extrapolate")
        merged["_size"] = size_func(np.log10(merged.nparam))

        sns.scatterplot(
            x="test_accuracy",
            y="brsupport",
            data=merged.sort_values(by="nparam", ascending=False),
            size="_size",
            hue="arch",
            hue_order=["wav2vec2", "Whisper", "Other (CNN)"],
            palette="viridis",
            sizes=(merged["_size"].min(), merged["_size"].max()),
            color="k",
            edgecolor="k",
            legend=False,
        )

        sns.regplot(
            x="test_accuracy",
            y="brsupport",
            data=merged,
            scatter=False,
            ax=ax,
            color="grey",
            line_kws={"linestyle": "--"},
        )

        # Manual arch legend handles
        arch_palette = sns.color_palette("viridis", 3)
        arch_order = ["wav2vec2", "Whisper", "Other (CNN)"]
        archhandles = [
            Line2D([], [], marker="o", color=c, linestyle="", markersize=5, markeredgecolor="k")
            for c in arch_palette
        ]

        legend_kw = dict(
            handletextpad=0.3, labelspacing=0.1, borderpad=0.4, borderaxespad=0.5,
            title_fontproperties={"weight": "bold"}, alignment="left",
        )

        # Manual size legend
        sizevalues = [1, 30, 300, 1000]
        size_cmap = plt.cm.Greys
        sizehandles = [
            Line2D([], [], marker="o", color=size_cmap(0.3 + 0.6 * i / 3),
                    linestyle="", markersize=np.sqrt(size_func(np.log10(val))), markeredgecolor="k")
            for i, val in enumerate(sizevalues)
        ]

        size_legend = ax.legend(
            handles=sizehandles, labels=[str(v) for v in sizevalues],
            title="Size (M)", loc="lower right",
            **legend_kw,
        )
        align_legend_title(fig, size_legend)
        ax.add_artist(size_legend)

        # Architecture legend
        arch_legend = ax.legend(
            handles=archhandles, labels=arch_order,
            title="Architecture", loc="upper left", **legend_kw,
        )
        align_legend_title(fig, arch_legend)

        ax.set_xlabel("Language identification accuracy (%)")
        ax.set_ylabel("Mean bootstrap support")
        xlsr = merged[merged.model_id == "facebook/wav2vec2-xls-r-300m"]
        ax.annotate(
            "XLS-R", (xlsr.test_accuracy.values[0], xlsr.brsupport.values[0]),
            fontsize=7.5, fontweight="bold",
            xytext=(8, -25), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
        )
        clear_axes()
        plt.grid(alpha=0.2)
        plt.savefig(f"{IMG_DIR}/fig1a_accuracy_vs_brsupport.pdf", bbox_inches="tight")
        plt.savefig(f"{IMG_DIR}/fig1a_accuracy_vs_brsupport.svg", format="svg", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    import os

    os.makedirs(IMG_DIR, exist_ok=True)
    merged = load_data()
    print(f"Merged {len(merged)} models")
    plot(merged)
