"""Figure 1a / Supp Fig 1: LID accuracy (or F1) vs. mean bootstrap support.

Caption: Language identification performance across speech models versus their
historical signal derived from phylogenetic branch supports inferred from
learned binarized embeddings. Supp Fig 1 shows the same with F1.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import seaborn as sns
from matplotlib.lines import Line2D

from src._config import DEFAULT_EVAL_DIR, DEFAULT_PER_SENTENCE_DIR
from src.tasks.plot import clear_axes

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

IMG_DIR = f"{DEFAULT_IMG_DIR}/fig1"

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
    "utter-project/mHuBERT-147": {"arch": "HuBERT", "nparam": 94.4},
    "facebook/mms-1b-all": {"arch": "wav2vec2", "nparam": 964.6},
}

ARCH_ORDER = ["wav2vec2", "HuBERT", "Whisper", "Other (CNN)"]

BASE_SIZE = 8
SIZE_ANCHORS = np.array([1, 30, 300, 1000])
SIZE_VALUES = np.array([BASE_SIZE, 3 * BASE_SIZE, 9 * BASE_SIZE, 27 * BASE_SIZE])
SIZE_FUNC = interp.interp1d(
    np.log10(SIZE_ANCHORS), SIZE_VALUES, fill_value="extrapolate"
)


def _align_legend_title(fig, leg):
    fig.canvas.draw()
    offset = leg.handlelength * plt.rcParams["font.size"] / 2
    leg.get_title().set_position((offset, 0))


def load_data():
    tree_summary = pd.read_csv(f"{DEFAULT_PER_SENTENCE_DIR}/discrete3+vote/summary.csv")

    stat_dfs = []
    for _, row in tree_summary.iterrows():
        f = f"{DEFAULT_PER_SENTENCE_DIR}/discrete3+vote/{row.run_id}/_stats.csv"
        df = pd.read_csv(f, index_col=0).query("Ntips > 38")
        if df.empty:
            continue
        df_mean = df.drop("Ntips", axis=1).mean()
        df_mean["ckpt"] = row.ckpt
        stat_dfs.append(df_mean)
    stat_df = pd.DataFrame(stat_dfs)

    eval_df = pd.read_csv(f"{DEFAULT_EVAL_DIR}/phylaudio2_summary.csv")

    model_details = pd.DataFrame.from_dict(MODEL_DETAILS, orient="index").sort_index()

    merged = (
        eval_df.merge(stat_df[["ckpt", "brsupport"]], on="ckpt")
        .merge(model_details, left_on="model_id", right_index=True)
        .dropna(subset=["brsupport"])
    )
    merged["test_accuracy"] *= 100
    merged["test_f1"] *= 100

    # Keep the hidden_dim with best bootstrap support per model
    best_idx = merged.groupby("model_id")["brsupport"].idxmax()
    return merged.loc[best_idx].reset_index(drop=True)


def _plot(ax, fig, data, x_col, xlabel):
    arch_palette = dict(zip(ARCH_ORDER, sns.color_palette("viridis", len(ARCH_ORDER))))
    data = data.copy()
    data["_size"] = SIZE_FUNC(np.log10(data.nparam))

    for arch in ARCH_ORDER:
        group = data[data.arch == arch]
        if group.empty:
            continue
        ax.scatter(
            group[x_col],
            group["brsupport"],
            s=group["_size"],
            c=[arch_palette[arch]],
            edgecolor="k",
            linewidth=0.5,
            zorder=2,
        )

    sns.regplot(
        x=x_col,
        y="brsupport",
        data=data,
        scatter=False,
        ax=ax,
        color="grey",
        ci=None,
        line_kws={"linestyle": "--"},
    )

    legend_kw = dict(
        handletextpad=0.3,
        labelspacing=0.1,
        borderpad=0.4,
        borderaxespad=0.5,
        title_fontproperties={"weight": "bold"},
        alignment="left",
    )

    # Architecture legend
    present = [a for a in ARCH_ORDER if a in data.arch.values]
    archhandles = [
        Line2D(
            [],
            [],
            marker="o",
            color=arch_palette[a],
            linestyle="",
            markersize=5,
            markeredgecolor="k",
        )
        for a in present
    ]
    arch_legend = ax.legend(
        handles=archhandles,
        labels=present,
        title="Architecture",
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        **legend_kw,
    )
    _align_legend_title(fig, arch_legend)
    ax.add_artist(arch_legend)

    # Size legend
    size_legend_values = [1, 30, 300, 1000]
    # pylint: disable=no-member
    size_cmap = plt.cm.Greys
    # pylint: enable=no-member
    sizehandles = [
        Line2D(
            [],
            [],
            marker="o",
            color=size_cmap(0.3 + 0.6 * i / (len(size_legend_values) - 1)),
            linestyle="",
            markersize=np.sqrt(SIZE_FUNC(np.log10(val))),
            markeredgecolor="k",
        )
        for i, val in enumerate(size_legend_values)
    ]
    size_legend = ax.legend(
        handles=sizehandles,
        labels=[str(v) for v in size_legend_values],
        title="Size (M)",
        loc="lower left",
        bbox_to_anchor=(1.0, 0.0),
        **legend_kw,
    )
    _align_legend_title(fig, size_legend)

    # XLS-R annotation
    xlsr = data[data.model_id == "facebook/wav2vec2-xls-r-300m"]
    if not xlsr.empty:
        ax.annotate(
            "XLS-R",
            (float(xlsr[x_col].iloc[0]), float(xlsr.brsupport.iloc[0])),
            fontsize=7.5,
            fontweight="bold",
            xytext=(8, -25),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean bootstrap support")
    clear_axes(ax)
    ax.grid(alpha=0.2)

    return [arch_legend, size_legend]


def plot_acc_vs_brsupport(data, output_name="fig1a_acc_vs_brsupport"):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        legends = _plot(
            ax, fig, data, "test_accuracy", "Language identification accuracy (%)"
        )
        for ext in ("pdf", "svg"):
            fig.savefig(
                f"{IMG_DIR}/{output_name}.{ext}",
                bbox_inches="tight",
                bbox_extra_artists=legends,
            )
        print(f"Saved {IMG_DIR}/{output_name}.{{pdf,svg}}")
        plt.show()


def plot_f1_vs_brsupport(data, output_name="figS1_f1_vs_brsupport"):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        legends = _plot(ax, fig, data, "test_f1", "Macro F1 score (%)")
        for ext in ("pdf", "svg"):
            fig.savefig(
                f"{IMG_DIR}/{output_name}.{ext}",
                bbox_inches="tight",
                bbox_extra_artists=legends,
            )
        print(f"Saved {IMG_DIR}/{output_name}.{{pdf,svg}}")
        plt.show()


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    data = load_data()
    print(f"Loaded {len(data)} models")
    print(data[["model_id", "hidden_dim", "brsupport"]].to_string(index=False))
    plot_acc_vs_brsupport(data)
    plot_f1_vs_brsupport(data)
