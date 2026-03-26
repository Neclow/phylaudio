#!/usr/bin/env python3
"""Interactive PCA of XLS-R embeddings (Plotly HTML)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from plots.fig1_pca import EMB_DIR, TAXONSET_DISPLAY, TAXONSET_ORDER, load_data
from src._config import DEFAULT_EMBEDDING_DIR, DEFAULT_ROOT_DIR, SAMPLE_RATE
from src.models.audio import AudioProcessor
from src.tasks.feature_extraction.base import prepare_dataset

OUTPUT_FILE = "docs/index.html"


def get_sentences(parallel_dataset):
    """Extract sentences from parallel dataset, filtering to those with >= 4 languages."""
    sentence_idxs = parallel_dataset.data.sentence_index.unique()
    sentence_dfs = []

    for sentence_index in sentence_idxs:
        sentence_df = parallel_dataset.data.query("sentence_index == @sentence_index")
        if sentence_df.language.nunique() < 4:
            continue
        sentence_dfs.append(sentence_df)

    return pd.concat(sentence_dfs, axis=0).loc[:, "sentence"]


def load_sentences():
    """Load the parallel dataset to get sentence transcripts."""
    run_dir = EMB_DIR
    with open(f"{run_dir}/cfg.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args = SimpleNamespace(**cfg)

    parallel_dataset = prepare_dataset(
        args,
        processor=AudioProcessor(sr=SAMPLE_RATE, max_length=args.max_length),
        split=False,
        fleurs_parallel=True,
        glottocode=args.glottocode,
        min_speakers=args.min_speakers,
    )[0]

    sentences = get_sentences(parallel_dataset)
    num_classes = len(parallel_dataset.label_encoder)
    labels = parallel_dataset.label_encoder.decode_torch(torch.arange(num_classes))

    return sentences, labels


def plot_plotly(X_pca, y_emb, var_exp, labels_pca, mapping, color_map, sentences=None):
    rng = np.random.default_rng(42)
    idxs = rng.choice(X_pca.shape[0], size=min(15000, X_pca.shape[0]), replace=False)

    traces = []

    # Scatter points grouped by taxonset
    for taxonset in TAXONSET_ORDER:
        for class_id in y_emb.cpu().unique().tolist():
            lang_code = labels_pca[class_id]
            if mapping[lang_code]["taxonset"] != taxonset:
                continue

            class_mask = y_emb == class_id
            class_idxs = [i for i in idxs if class_mask[i]]
            if not class_idxs:
                continue

            lang_name = mapping[lang_code]["fleurs"].split(" ")[0]

            scatter_kwargs = {
                "x": X_pca[class_idxs, 0].cpu().numpy(),
                "y": X_pca[class_idxs, 1].cpu().numpy(),
                "z": X_pca[class_idxs, 2].cpu().numpy(),
                "mode": "markers",
                "name": lang_name,
                "marker": {
                    "size": 3,
                    "color": color_map[class_id],
                    "opacity": 0.3,
                    "line": {"width": 0},
                },
                "legendgroup": taxonset,
                "legendgrouptitle_text": TAXONSET_DISPLAY[taxonset],
                "showlegend": True,
            }

            if sentences is not None:
                class_sentences = [sentences.iloc[i] for i in class_idxs]
                scatter_kwargs["text"] = class_sentences
                scatter_kwargs["customdata"] = [[lang_name]] * len(class_idxs)
                scatter_kwargs["hovertemplate"] = (
                    "<b>%{customdata[0]}</b><br>"
                    "PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br><br>"
                    "<i>%{text}</i><extra></extra>"
                )
            else:
                scatter_kwargs["hovertemplate"] = (
                    f"<b>{lang_name}</b><br>"
                    "PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>"
                )

            traces.append(go.Scatter3d(**scatter_kwargs))

    # Centroids
    for class_id in y_emb.cpu().unique().tolist():
        class_mask = y_emb == class_id
        centroid = X_pca[class_mask, :3].mean(dim=0).cpu().numpy()
        lang_code = labels_pca[class_id]
        lang_name = mapping[lang_code]["fleurs"].split(" ")[0]
        taxonset = mapping[lang_code]["taxonset"]

        traces.append(
            go.Scatter3d(
                x=[centroid[0]],
                y=[centroid[1]],
                z=[centroid[2]],
                mode="markers+text",
                name=f"{lang_name} (centroid)",
                marker={
                    "size": 6,
                    "color": color_map[class_id],
                    "line": {"color": "black", "width": 2},
                    "symbol": "circle",
                },
                text=[lang_name],
                textposition="top center",
                textfont={"size": 10, "color": "black"},
                hovertemplate=(
                    f"<b>{lang_name} (centroid)</b><br>"
                    f"PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>"
                ),
                legendgroup=taxonset,
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="PCA of XLS-R Embeddings (FLEURS-R, Indo-European)",
        scene={
            "xaxis_title": f"PC 1 ({var_exp[0]*100:.1f}%)",
            "yaxis_title": f"PC 2 ({var_exp[1]*100:.1f}%)",
            "zaxis_title": f"PC 3 ({var_exp[2]*100:.1f}%)",
        },
        width=1000,
        height=800,
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 12},
        legend={
            "groupclick": "toggleitem",
            "itemsizing": "constant",
            "tracegroupgap": 10,
        },
    )

    return fig


if __name__ == "__main__":
    X_pca, y_emb, var_exp, labels_pca, mapping, color_map, _ = load_data()
    print(
        f"PCA: {X_pca.shape}, PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}%, PC3: {var_exp[2]*100:.1f}"
    )

    sentences, _ = load_sentences()
    print(f"Loaded {len(sentences)} sentences for hover text")

    fig = plot_plotly(X_pca, y_emb, var_exp, labels_pca, mapping, color_map, sentences)
    fig.write_html(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
