# pylint: disable=redefined-outer-name
import json
from argparse import ArgumentParser
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch

from src._config import DEFAULT_PER_SENTENCE_DIR, DEFAULT_ROOT_DIR
from src.tasks.feature_extraction._decomposition import decompose, fit_decomposer
from src.tasks.feature_extraction.base import prepare_dataset, prepare_model

DISCRETE_DIR = f"{DEFAULT_PER_SENTENCE_DIR}/discrete"

palette_map = {
    "germanic": "blend:olive,#5C7450",
    "indoaryan": "blend:maroon,#824C5C",
    "iranian": "blend:lightpink,hotpink",
    "slavic": "blend:orange,#D2B673",
    "baltic": "blend:red,darkred",
    "latinofaliscan": "blend:#9AC26B,greenyellow",
    "Greek": ["deepskyblue"],
    "armenian": ["tab:purple"],
}

sizes = {
    "germanic": 7,
    "indoaryan": 10,
    "slavic": 12,
    "latinofaliscan": 7,
    "iranian": 4,
    "Greek": 1,
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

    parser.add_argument(
        "-i",
        "--run-id",
        type=str,
        required=True,
        help="Run ID for the experiment.",
    )

    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="img/xls-r_finetuned_pca_2d_15k_interactive.html",
        help="Output HTML file for the interactive plot.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    viz_args = parse_args()
    run_id = viz_args.run_id
    with open(
        f"{DISCRETE_DIR}/{run_id}/cfg.json",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)
    cfg["ebs"] = 64
    cfg["device"] = "cuda:0"

    args = SimpleNamespace(**cfg)

    meta_dir = f"{DEFAULT_ROOT_DIR}/metadata/{args.dataset}"
    mapping_file = f"{meta_dir}/languages.json"

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    processor, feature_extractor = prepare_model(args, training=False)

    parallel_dataset = prepare_dataset(
        args,
        processor=processor,
        split=False,
        fleurs_parallel=True,
        glottocode=args.glottocode,
        min_speakers=args.min_speakers,
    )[0]

    num_classes = len(parallel_dataset.label_encoder)

    labels = parallel_dataset.label_encoder.decode_torch(torch.arange(num_classes))

    X = torch.load("tmp/emb/xls-r_finetuned.pt", map_location="cuda:0")

    y = torch.load("tmp/emb/labels.pt", map_location="cuda:0")

    decomposer = fit_decomposer(
        X,
        method="pca",
        n_components=0.99,
        standardize=True,
        device=args.device,
        seed=42,
    )

    X_pca = decompose(decomposer, X)

    print(X_pca.shape)  # Should be (1000, 50)

    n_classes = len(set(y.cpu().tolist()))

    color_map, colors = get_colors(y, mapping, labels)

    print("Creating interactive Plotly visualization...")

    the_sentence_df = get_sentences(parallel_dataset)

    # Sample points for better performance
    idxs = np.random.choice(X_pca.shape[0], size=15000, replace=False)

    # Get sentences for the sampled points
    sentences_sampled = the_sentence_df.iloc[idxs].values

    # Prepare data for plotting
    plot_data = []

    # Add scatter points
    for class_id in y.unique():
        class_mask = y == class_id
        class_idxs = [i for i in idxs if class_mask[i]]

        if len(class_idxs) > 0:
            lang_name = mapping[labels[class_id.item()]]["full"].split(" ")[0]
            taxonset = mapping[labels[class_id.item()]]["taxonset"]

            # Get sentences for this class
            class_sentences = [the_sentence_df.iloc[i] for i in class_idxs]

            plot_data.append(
                go.Scatter(
                    x=X_pca[class_idxs, 0].cpu().numpy(),
                    y=X_pca[class_idxs, 1].cpu().numpy(),
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
        class_mean = X_pca[class_mask, :2].mean(dim=0).cpu().numpy()
        lang_name = mapping[labels[class_id.item()]]["full"].split(" ")[0]
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

    # Update layout
    fig.update_layout(
        title="PCA Visualization of FLEURS-R Embeddings\n(Model: XLS-R finetuned on VoxLingua107)",
        xaxis_title=f"PC 1 ({decomposer.explained_variance_ratio_[0]*100:.2f}%)",
        yaxis_title=f"PC 2 ({decomposer.explained_variance_ratio_[1]*100:.2f}%)",
        xaxis={"range": [-24, 18]},
        yaxis={"range": [-17, 17]},
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

    # Save as HTML
    fig.write_html(viz_args.output_file)
    print(f"Interactive plot saved to {viz_args.output_file}")
