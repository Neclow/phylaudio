#!/usr/bin/env python3
"""Extended Figure: Silhouette score vs SI-SDR + audio quality correlation matrix."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from scipy.stats import linregress, pearsonr

from src._config import DEFAULT_METADATA_DIR, MIN_LANGUAGES
from src.data.glottolog import filter_languages_from_glottocode

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_DIR = "img/fig1"
EMB_DIR = "data/embeddings/fleurs-r/facebook_wav2vec2-xls-r-300m"

PALETTE_MAP = {
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


@torch.jit.script
def silhouette_scores(X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Silhouette score implemented in PyTorch for GPU acceleration."""
    unique_labels = torch.unique(labels)
    n_samples = labels.size(0)

    intra_dist = torch.zeros(n_samples, dtype=X.dtype, device=X.device)
    for label in unique_labels:
        mask = labels == label
        where = mask.nonzero().squeeze(1)
        distances = torch.cdist(X[where], X[where])
        intra_dist[where] = distances.sum(dim=1) / (distances.shape[0] - 1)

    inter_dist = torch.full((n_samples,), torch.inf, dtype=X.dtype, device=X.device)
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            label_a = unique_labels[i]
            label_b = unique_labels[j]

            where_a = (labels == label_a).nonzero().squeeze(1)
            where_b = (labels == label_b).nonzero().squeeze(1)

            dist = torch.cdist(X[where_a], X[where_b])
            dist_a = dist.mean(dim=1)
            dist_b = dist.mean(dim=0)

            inter_dist[where_a] = torch.minimum(dist_a, inter_dist[where_a])
            inter_dist[where_b] = torch.minimum(dist_b, inter_dist[where_b])

    sil_samples = (inter_dist - intra_dist) / torch.maximum(intra_dist, inter_dist)
    return sil_samples.nan_to_num()


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    X_emb = torch.load(f"{EMB_DIR}/embeddings.pt", map_location=device)
    y_emb = torch.load(f"{EMB_DIR}/labels.pt", map_location=device)

    with open("data/metadata/fleurs-r/labels.txt") as f:
        all_labels = [l.strip().split(" => ")[0].strip("'") for l in f.readlines()]

    with open("data/metadata/fleurs-r/languages.json") as f:
        mapping = json.load(f)

    labels_dict = {i: code for i, code in enumerate(all_labels) if i in y_emb.unique().tolist()}

    # Compute silhouette scores
    scores = silhouette_scores(X_emb, y_emb)
    scores_df = pd.DataFrame({
        "silhouette_score": scores.cpu().numpy(),
        "label": y_emb.cpu().numpy(),
    })
    scores_df["fleurs"] = scores_df["label"].map(lambda x: mapping[labels_dict[x]]["fleurs"])
    scores_df_mean = scores_df.groupby("fleurs")["silhouette_score"].mean()

    # Load SQUIM audio quality metrics
    dataset = "fleurs-r"
    squim_df = pd.read_csv(f"{DEFAULT_METADATA_DIR}/{dataset}/squim.csv")
    squim_df["fleurs"] = squim_df["language"].map(lambda x: mapping[x]["fleurs"])

    indo1319_languages = filter_languages_from_glottocode(dataset, "indo1319")
    squim_indo1319 = squim_df[squim_df["language"].isin(indo1319_languages)]

    lang_counts = squim_indo1319.groupby("sentence_index")["language"].nunique()
    valid_sentences = lang_counts[lang_counts >= MIN_LANGUAGES].index
    squim_filtered = squim_indo1319[squim_indo1319["sentence_index"].isin(valid_sentences)]

    audio_df_mean = (
        squim_filtered
        .loc[:, ["stoi", "pesq", "si_sdr", "fleurs"]]
        .groupby("fleurs")
        .mean()
    )

    # Merge
    taxonset_mapping = {v["fleurs"]: v["taxonset"] for v in mapping.values() if "taxonset" in v}
    palette = {k: sns.color_palette(v)[0] for k, v in PALETTE_MAP.items()}

    audio_and_scores = pd.concat([audio_df_mean, scores_df_mean], axis=1).reset_index()
    audio_and_scores["taxonset"] = audio_and_scores.fleurs.map(taxonset_mapping)
    audio_and_scores["count"] = audio_and_scores.fleurs.map(squim_df.fleurs.value_counts())

    return audio_and_scores, palette


# ── Plot: Silhouette vs SI-SDR ───────────────────────────────────────────────
def plot_silhouette_vs_sisdr(audio_and_scores, palette):
    with plt.style.context(".matplotlib/paper.mplstyle"):
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.scatterplot(
            x="si_sdr", y="silhouette_score", hue="taxonset", size="count",
            palette=palette, data=audio_and_scores,
            sizes=(20, 200), legend=False, alpha=0.7, ax=ax,
        )
        sns.regplot(
            x="si_sdr", y="silhouette_score", data=audio_and_scores,
            scatter=False, ax=ax, color="black",
            line_kws={"linestyle": "--", "alpha": 0.7}, ci=None,
        )

        # Annotate extremes
        for _, row in audio_and_scores.query("silhouette_score < 0.2").iterrows():
            ax.text(
                row.si_sdr, row.silhouette_score, row.fleurs,
                ha="left" if row.fleurs not in ["Bosnian"] else "right",
                va="top" if row.fleurs not in ["Serbian", "Urdu", "Hindi"] else "bottom",
            )
        for _, row in audio_and_scores.query("silhouette_score > 0.5")[::-1].iterrows():
            ax.text(
                row.si_sdr, row.silhouette_score, row.fleurs,
                ha="left" if row.fleurs not in ["Norwegian", "Persian"] else "right",
                va="bottom" if row.fleurs != "Norwegian" else "top",
            )

        # Regression annotation
        slope, intercept, r_value, p_value, *_ = linregress(
            audio_and_scores["si_sdr"], audio_and_scores["silhouette_score"],
        )
        x_left = audio_and_scores["si_sdr"].min()
        y_left = intercept + slope * x_left

        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        aspect = (y_range / x_range) * (ax.get_window_extent().width / ax.get_window_extent().height)
        angle = np.degrees(np.arctan(slope / aspect))

        ax.text(x_left, y_left, f"r = {r_value:.2f}", va="bottom", ha="left",
                rotation=angle, rotation_mode="anchor", color="black", alpha=0.7)
        ax.text(x_left, y_left - 0.005, f"p = {p_value:.2f}", va="top", ha="left",
                rotation=angle, rotation_mode="anchor", color="black", alpha=0.7)

        ax.set_ylabel("Silhouette Score")
        ax.set_xlabel("SI-SDR (dB)")
        ax.grid(alpha=0.2)

        # Taxonset legend
        taxonset_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[ts],
                   markersize=8, linestyle="None")
            for ts in palette
        ]
        legend1 = ax.legend(
            taxonset_handles, [ts.capitalize() for ts in palette],
            frameon=False, loc="upper left", bbox_to_anchor=(0.97, 1.0), ncol=1,
        )
        ax.add_artist(legend1)

        # Size legend
        count_min = audio_and_scores["count"].min()
        count_max = audio_and_scores["count"].max()
        size_values = [1000, 2000, 3000, 4000, 5000]
        size_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markersize=np.sqrt(20 + (200 - 20) * (v - count_min) / (count_max - count_min)),
                   linestyle="None")
            for v in size_values
        ]
        ax.legend(
            size_handles, [str(v) for v in size_values],
            title="# Samples", frameon=False,
            loc="lower left", bbox_to_anchor=(1.0, 0.0),
        )

        fig.subplots_adjust(right=0.78)
        fig.savefig(f"{IMG_DIR}/ext_fig1_silhouette_vs_sisdr.pdf", bbox_inches="tight")
        plt.show()


# ── Plot: Correlation heatmap ─────────────────────────────────────────────────
def plot_correlation_heatmap(audio_and_scores):
    corr_cols = ["stoi", "pesq", "si_sdr", "silhouette_score"]
    corr_labels = ["STOI", "PESQ", "SI-SDR", "Silhouette"]
    n = len(corr_cols)

    corr_matrix = audio_and_scores[corr_cols].corr()
    p_matrix = pd.DataFrame(np.ones((n, n)), index=corr_cols, columns=corr_cols)
    for i in range(n):
        for j in range(i + 1, n):
            _, p = pearsonr(audio_and_scores[corr_cols[i]], audio_and_scores[corr_cols[j]])
            p_matrix.iloc[i, j] = p
            p_matrix.iloc[j, i] = p

    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{corr_matrix.iloc[i, j]:.2f}{sig_stars(p_matrix.iloc[i, j])}"

    with plt.style.context(".matplotlib/paper.mplstyle"):
        fig, ax = plt.subplots(figsize=(4, 3.5))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, mask=mask, annot=annot, fmt="",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            xticklabels=corr_labels, yticklabels=corr_labels,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        )

        fig.savefig(f"{IMG_DIR}/ext_fig1_audio_corr_heatmap.pdf", bbox_inches="tight")
        fig.savefig(f"{IMG_DIR}/ext_fig1_audio_corr_heatmap.svg", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    import os

    os.makedirs(IMG_DIR, exist_ok=True)
    audio_and_scores, palette = load_data()
    print(f"Merged: {len(audio_and_scores)} languages")
    plot_silhouette_vs_sisdr(audio_and_scores, palette)
    plot_correlation_heatmap(audio_and_scores)
