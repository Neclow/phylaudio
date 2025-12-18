from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from src._config import DEFAULT_EVAL_DIR, DEFAULT_PER_SENTENCE_DIR
from src.models._model_zoo import MODEL_ZOO
from src.tasks.plot import clear_axes


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-t",
        "--tree_dir",
        type=str,
        default="discrete",
        help="Path to per-sentence tree dir.",
    )
    parser.add_argument(
        "-e",
        "--eval_path",
        type=str,
        default=f"{DEFAULT_EVAL_DIR}/summary.json",
        help="Path to eval summary JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="img/brsupport_vs_accuracy_indo1319.pdf",
        help="Output file for the plot.",
    )
    parser.add_argument(
        "-c",
        "--clf-metric",
        type=str,
        default="test_accuracy",
        help="Classification metric to plot against brsupport.",
    )

    return parser.parse_args()


def load_data(tree_dir, eval_path):
    tree_summary = pd.read_csv(
        f"{DEFAULT_PER_SENTENCE_DIR}/{tree_dir}/summary.csv", index_col=0
    )

    id2model = tree_summary.model_id.to_dict()

    all_tree_stats = []
    stat_files = sorted(glob(f"{DEFAULT_PER_SENTENCE_DIR}/{tree_dir}/*/_stats.csv"))
    for f in stat_files:
        df = pd.read_csv(f, index_col=0).query("Ntips > 38")
        df_mean = df.drop("Ntips", axis=1).mean()
        df_mean["model_id"] = id2model[Path(f).parent.name]
        all_tree_stats.append(df_mean)

    model_details_dict = {
        model_id: {
            "type": MODEL_ZOO[model_id].get("type"),
            "arch": MODEL_ZOO[model_id].get("arch"),
        }
        for model_id in MODEL_ZOO.keys()
    }

    model_details = pd.DataFrame.from_dict(
        model_details_dict, orient="index"
    ).sort_index()

    tree_stats = pd.DataFrame(all_tree_stats)
    eval_stats = (
        pd.read_json(eval_path, orient="index")
        .reset_index(names="run_id")
        .rename(columns={"model_size": "nparam"})
    )

    merged_ = (
        eval_stats.loc[:, ["model_id", "nparam", "test_accuracy", "test_f1"]]
        .merge(model_details, left_on="model_id", right_index=True)
        .merge(tree_stats, on="model_id")
    )
    return merged_


def plot(
    data,
    clf_metric,
    output_file,
    figsize=(4, 2.7),
    max_pad=8,
    sizes=(13, 130),
    sizevalues=(1, 30, 300, 1000),
):

    with plt.style.context(".matplotlib/paper.mplstyle"):
        _, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            x=clf_metric,
            y="brsupport",
            data=data.sort_values(by="nparam", ascending=False),
            size="nparam",
            hue="arch",
            hue_order=["wav2vec2", "Whisper", "Other (CNN)"],
            palette="colorblind",
            sizes=sizes,
            color="k",
            edgecolor="k",
        )

        # Add a linear regression line
        sns.regplot(
            x=clf_metric,
            y="brsupport",
            data=data,
            scatter=False,
            ax=ax,
            color="grey",
            line_kws={"linestyle": "--"},
        )

        # Get the num_params legend
        handles, labels = ax.get_legend_handles_labels()

        archhandles = handles[1 : labels.index("nparam")]
        archlabels = labels[1 : labels.index("nparam")]

        vmin, vmax = data.nparam.min(), data.nparam.max()

        def map_size(val):
            """Map a num_params value to the marker size used in the plot"""
            return sizes[0] + (sizes[1] - sizes[0]) * (val - vmin) / (vmax - vmin)

        sizelabels = [f"{v}{'':>{max_pad-len(str(int(v)))+1}}" for v in sizevalues]

        sizehandles = [
            Line2D(
                [],
                [],
                marker="o",
                color="k",
                linestyle="",
                markersize=np.sqrt(map_size(val)),
                alpha=0.3,
            )
            for val in sizevalues
        ]

        size_legend = ax.legend(
            handles=sizehandles,
            labels=sizelabels,
            title="Model\nSize (M)",
            bbox_to_anchor=(1, -0.05),
            loc="lower left",
        )
        size_legend._legend_box.align = "right"  # aligns title box to right

        ax.add_artist(size_legend)
        ax.legend(
            handles=archhandles,
            labels=archlabels,
            title="Model\nArchitecture",
            bbox_to_anchor=(1, 0.55),
            loc="lower left",
            alignment="right",
        )

        ax.set_xlabel("Language identification accuracy (%)")
        ax.set_ylabel("Mean bootstrap support")

        clear_axes()
        plt.grid(alpha=0.2)
        plt.savefig(output_file, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    args = parse_args()

    merged = load_data(tree_dir=args.tree_dir, eval_path=args.eval_path)

    plot(data=merged, clf_metric=args.clf_metric, output_file=args.output_file)

    print(f"Plot saved to {args.output_file}")
