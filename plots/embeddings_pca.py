# pylint: disable=redefined-outer-name
import json
from argparse import ArgumentParser, ArgumentTypeError
from collections import defaultdict
from math import ceil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src._config import (
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_PER_SENTENCE_DIR,
    DEFAULT_ROOT_DIR,
    RANDOM_STATE,
    SAMPLE_RATE,
)
from src.models.audio import AudioProcessor
from src.tasks.feature_extraction._decomposition import decompose, fit_decomposer
from src.tasks.feature_extraction.base import prepare_dataset

DISCRETE_DIR = f"{DEFAULT_PER_SENTENCE_DIR}/discrete"

palette_map = {
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

sizes = {
    "germanic": 9,
    "celtic": 2,
    "indoaryan": 10,
    "slavic": 12,
    "romance": 10,
    "iranian": 4,
    "greek": 1,
    "armenian": 1,
    "baltic": 2,
}

hex_map = {
    k: sns.color_palette(v, n_colors=sizes[k]).as_hex() for k, v in palette_map.items()
}


def get_colors(y, mapping, labels):
    counts = defaultdict(int)

    color_map = {}

    for yi in y.cpu().unique():
        taxonset = mapping[labels[yi.item()]]["taxonset"]
        palette = hex_map[taxonset]
        color = palette[counts[taxonset]]
        color_map[yi.item()] = color
        counts[taxonset] += 1

    colors = [color_map[yi.item()] for yi in y.cpu()]

    return color_map, colors


def get_sentences(parallel_dataset):
    sentence_idxs = parallel_dataset.data.sentence_index.unique()

    sentence_dfs = []

    # pylint: disable=unused-variable
    for sentence_index in sentence_idxs:
        sentence_df = parallel_dataset.data.query("sentence_index == @sentence_index")

        if sentence_df.language.nunique() < 4:
            continue

        sentence_dfs.append(sentence_df)
    # pylint: enable=unused-variable

    return pd.concat(sentence_dfs, axis=0).loc[:, "sentence"]


def parse_args():
    parser = ArgumentParser(description="Visualize embeddings with PCA using Plotly")

    def int_or_float(value):
        """
        Custom type function for argparse to accept either an int or a float in (0, 1).
        """
        try:
            num = float(value)
            if num == int(num):
                return int(num)
            if 0 < num < 1:
                return num
            raise ValueError("Float must be in (0, 1).")
        except ValueError as err:
            raise ArgumentTypeError(f"'{value}' is not a valid int or float.") from err

    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name (e.g., `fleurs-r`)",
    )
    parser.add_argument(
        "-i",
        "--run-id",
        type=str,
        required=True,
        help="Run ID for the experiment.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int_or_float,
        help="Number of samples to plot (int) or fraction of total samples (float in (0, 1)).",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to load embeddings onto (e.g., 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "-nc",
        "--n-components",
        type=int_or_float,
        default=0.9,
    )

    return parser.parse_args()


def parse_int_or_float(value, n_samples):
    size_type = np.asarray(value).dtype.kind

    if size_type == "f" and (0 < value <= 1.0):
        n_subset = ceil(value * n_samples)
    elif size_type == "i" and 0 < value <= n_samples:
        n_subset = value
    else:
        raise ValueError(
            f"size={value} should be a number either positive and "
            f"smaller than the number of samples {n_samples} or a float "
            "in the (0, 1) range"
        )

    return n_subset


def plot(
    Z,
    y,
    sentences,
    idxs,
    mapping,
    labels,
    color_map,
    tag="fleurs",
):
    print("Creating interactive Plotly visualization...")

    # Prepare data for plotting
    plot_data = []

    # Add scatter points
    for class_id in y.unique():
        class_mask = y == class_id
        class_idxs = [i for i in idxs if class_mask[i]]

        if len(class_idxs) > 0:
            lang_name = mapping[labels[class_id.item()]][tag].split(" ")[0]
            taxonset = mapping[labels[class_id.item()]]["taxonset"]

            # Get sentences for this class
            class_sentences = [sentences.iloc[i] for i in class_idxs]

            plot_data.append(
                go.Scatter(
                    x=Z[class_idxs, 0].cpu().numpy(),
                    y=Z[class_idxs, 1].cpu().numpy(),
                    mode="markers",
                    name=lang_name,
                    marker={
                        "size": 5,
                        "color": color_map[class_id.item()],
                        "opacity": 0.3,
                        "line": {"width": 0},
                    },
                    text=class_sentences,
                    customdata=[[lang_name]] * len(class_idxs),
                    hovertemplate="<b>%{customdata[0]}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br><br><i>%{text}</i><extra></extra>",
                    legendgroup=taxonset,
                    legendgrouptitle_text=taxonset,
                    showlegend=True,
                )
            )

    # Add centroids
    for class_id in y.unique():
        class_mask = y == class_id
        class_mean = Z[class_mask, :2].mean(dim=0).cpu().numpy()
        lang_name = mapping[labels[class_id.item()]][tag].split(" ")[0]
        taxonset = mapping[labels[class_id.item()]]["taxonset"]

        plot_data.append(
            go.Scatter(
                x=[class_mean[0]],
                y=[class_mean[1]],
                mode="markers+text",
                name=f"{lang_name} (centroid)",
                marker={
                    "size": 12,
                    "color": color_map[class_id.item()],
                    "line": {"color": "black", "width": 2},
                    "symbol": "circle",
                },
                text=[lang_name],
                textposition="top center",
                textfont={"size": 10, "color": "black"},
                hovertemplate=f"<b>{lang_name} (centroid)</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
                legendgroup=taxonset,
                showlegend=False,
            )
        )

    # Create figure
    fig = go.Figure(data=plot_data)

    fig.update_layout(
        # xaxis={"range": [-24, 18]},
        # yaxis={"range": [-17, 17]},
        width=900,
        height=900,
        hovermode="closest",
        template="plotly_white",
        font={
            "family": "Arial, sans-serif",
            "size": 12,
        },
        legend={
            "groupclick": "toggleitem",
            "itemsizing": "constant",
            "tracegroupgap": 10,
        },
    )

    return fig


if __name__ == "__main__":
    viz_args = parse_args()

    # Parse cfg from run_dir
    run_dir = f"{DEFAULT_EMBEDDING_DIR}/{viz_args.dataset}/{viz_args.run_id}"
    with open(f"{run_dir}/cfg.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args = SimpleNamespace(**cfg)

    # Parse embeddings and labels and PCA size
    X = torch.load(f"{run_dir}/embeddings.pt", map_location=args.device)
    y = torch.load(f"{run_dir}/labels.pt", map_location=args.device)

    n_samples = X.shape[0]
    print(f"Loaded {n_samples} samples from {run_dir}")
    assert (
        X.shape[0] == y.shape[0]
    ), "Number of samples in embeddings and labels must match."

    n_subset = parse_int_or_float(viz_args.size, n_samples)

    with open(
        f"{DEFAULT_ROOT_DIR}/metadata/{viz_args.dataset}/languages.json",
        "r",
        encoding="utf-8",
    ) as f:
        mapping = json.load(f)

    # Load labels and sentences
    parallel_dataset = prepare_dataset(
        args,
        processor=AudioProcessor(sr=SAMPLE_RATE, max_length=args.max_length),
        split=False,
        fleurs_parallel=True,
        glottocode=args.glottocode,
        min_speakers=args.min_speakers,
    )[0]

    the_sentence_df = get_sentences(parallel_dataset)
    num_classes = len(parallel_dataset.label_encoder)
    labels = parallel_dataset.label_encoder.decode_torch(torch.arange(num_classes))
    color_map, colors = get_colors(y, mapping, labels)

    # Sample points from the transformed data for plotting
    idxs = np.random.choice(X.shape[0], size=n_subset, replace=False)

    # PCA
    decomposer = fit_decomposer(
        X,
        method=method,
        n_components=viz_args.n_components,
        standardize=True,
        device=viz_args.device,
        seed=RANDOM_STATE,
    )

    Z = decompose(decomposer, X)
    print(f"Decomposed data shape: {Z.shape}")

    # Cross-validation
    # for n_pc in np.geomspace(2, 128, 7, dtype=np.int64):
    #     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    #     from sklearn.model_selection import cross_val_score

    #     pca_sub = Z[:, :n_pc].cpu().numpy()
    #     lda = LinearDiscriminantAnalysis()
    #     acc = cross_val_score(lda, pca_sub, y.cpu().numpy(), cv=5).mean()
    #     print(f"PCs={n_pc}, CV accuracy={acc:.3f}")
    # 32 to 64 was a good spot

    methods = ("pca", "lda")

    for method in methods:
        if method == "lda":
            lda = LinearDiscriminantAnalysis(n_components=2)
            Z_lda = lda.fit_transform(Z.cpu().numpy(), y.cpu().numpy())
            Z = torch.from_numpy(Z_lda).to(viz_args.device)

        print(
            f"Original dimensionality: {X.shape[1]}, {method.upper()} dimensionality: {Z.shape[1]}"
        )

        # Plot
        fig = plot(
            Z=Z,
            y=y,
            sentences=the_sentence_df,
            idxs=idxs,
            mapping=mapping,
            labels=labels,
            color_map=color_map,
            tag="fleurs",
        )

        axis_titles = {}
        for i, axis in enumerate(["x", "y"]):
            if method == "pca":
                var_explained = decomposer.explained_variance_ratio_[i] * 100
                axis_titles[axis] = f"PC {i+1} ({var_explained:.2f}%)"
            else:
                axis_titles[axis] = f"{method.upper()} {i+1}"

        # Update layout
        fig.update_layout(
            title=f"{method.upper()} Visualization of {viz_args.dataset.upper()} Embeddings\n(Model: {args.model_id})",
            xaxis_title=axis_titles["x"],
            yaxis_title=axis_titles["y"],
        )

        output_file = f"img/{viz_args.dataset}_{method}_2d_{viz_args.run_id}.html"
        fig.write_html(output_file)
        print(f"Interactive plot saved to {output_file}")
