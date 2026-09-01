"""Figure 1b / Supp Figs 3-4: sNMF structure plot, K selection, and PHOIBLE regression.

Caption: (1b) Sparse NMF of XLS-R embeddings, dividing IE samples into clusters.
(Supp 3) NMF K selection by cross-entropy stability with local maximum at K=12.
(Supp 4) Bayesian ridge regression of NMF component proportions on PHOIBLE
phonological features — heatmap and forest plots of posterior means with 95% CIs.
"""

import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb

from ._config import (
    DEFAULT_IMG_DIR,
    DEFAULT_STYLE,
    NMF_COMP_LABELS,
    NMF_COMP_ORDER,
    PALETTE,
    SPEECH_BEAST_DIR,
)

IMG_DIR = f"{DEFAULT_IMG_DIR}/fig1"
NMF_DIR = f"{SPEECH_BEAST_DIR}/nmf"
BRMS_DIR = f"{SPEECH_BEAST_DIR}/brms_phoible"

K = 9


# Data loading
def load_nmf(nmf_dir=NMF_DIR):
    hits = sorted(glob(f"{nmf_dir}/Q_K*.csv"))
    if not hits:
        raise FileNotFoundError(f"No Q_K*.csv found in {nmf_dir}/")
    csv_path = hits[0]

    df = pd.read_csv(csv_path)
    labels = df["language"].values
    P = df.drop(columns=["language"]).values
    k_star = P.shape[1]
    assert k_star == K, f"Expected K={K} but Q matrix has {k_star} components"

    P = P[:, NMF_COMP_ORDER]

    # Sort languages: by dominant component, then by its proportion (descending)
    max_comp = np.argmax(P, axis=1)
    max_val = P[np.arange(len(labels)), max_comp]
    order = np.lexsort((-max_val, max_comp))

    P_sorted = P[order]
    labels_sorted = labels[order]
    return P_sorted, labels_sorted


def load_ce(nmf_dir=NMF_DIR):
    ce_path = os.path.join(nmf_dir, "cross_entropy.csv")
    if not os.path.isfile(ce_path):
        raise FileNotFoundError(f"No cross_entropy.csv found in {nmf_dir}/")
    return pd.read_csv(ce_path)


# Fig 1b
def plot_structure(P_sorted, labels_sorted):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.25, 8))

        colors = PALETTE[:K]
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
                label=NMF_COMP_LABELS[j],
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
            ncol=3,
            title="Acoustic components",
            title_fontproperties={"weight": "bold", "size": 8},
            columnspacing=0.6,
            handletextpad=0.3,
            labelspacing=0.15,
            alignment="left",
        )
        leg.get_title().set_position((0, 1))
        ax.grid(axis="x", linestyle="dashed", alpha=0.1, color="k")

        stem = f"{IMG_DIR}/fig1b_nmf_structure_K{K:02d}"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
        print(f"Saved {stem}.pdf")
        plt.show()


# Supp Fig. 3
def plot_ce(df_ce):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(4, 3))

        ax.plot(
            df_ce["K"],
            df_ce["CE"],
            "_",
            color="k",
            markersize=8,
            markeredgewidth=1.5,
        )
        ax.axvline(K, ls="--", color="grey", lw=0.8, label=f"K* = {K}")

        ax.set_xlabel("Number of NMF components (K)")
        ax.set_ylabel("Cross-entropy (masked)")
        # ax.set_xticks(df_ce["K"].values[::2])
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.15)

        stem = f"{IMG_DIR}/figS3_nmf_ce_vs_k"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        print(f"Saved {stem}.pdf")
        plt.show()


def _feat_label(s):
    rename = {
        "n_phonemes": "# phonemes",
        "n_consonants": "# consonants",
        "n_vowels": "# vowels",
    }
    if s in rename:
        return rename[s]
    if s.startswith("has_"):
        raw = s[len("has_") :]
        out = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)
        return out[0].upper() + out[1:]
    return s


def load_brms(brms_dir=BRMS_DIR):
    df = pd.read_csv(os.path.join(brms_dir, "nmf_phoible_brms.csv"))
    df["feature_label"] = df["feature"].map(_feat_label)

    feat_labels = df[df.component == 1]["feature_label"].tolist()

    coef = df.pivot(index="feature_label", columns="component", values="estimate")
    ci_lo = df.pivot(index="feature_label", columns="component", values="ci_lower")
    ci_hi = df.pivot(index="feature_label", columns="component", values="ci_upper")
    r2 = df.groupby("component")["r2_mean"].first().values

    col_order = [c + 1 for c in NMF_COMP_ORDER]
    coef = coef.loc[feat_labels, col_order].values
    ci_lo = ci_lo.loc[feat_labels, col_order].values
    ci_hi = ci_hi.loc[feat_labels, col_order].values
    r2 = r2[NMF_COMP_ORDER]

    sig = (ci_lo > 0) | (ci_hi < 0)
    return feat_labels, coef, ci_lo, ci_hi, sig, r2


# Supp Fig. 4a
def plot_phoible_heatmap(feat_labels, coef, sig):
    with plt.style.context(DEFAULT_STYLE):
        alpha_idx = np.argsort(feat_labels)
        feat_sorted = [feat_labels[i] for i in alpha_idx]
        vmax = np.abs(coef).max()

        col_labels = [str(i + 1) for i in range(K)]
        coef_hm = pd.DataFrame(coef[alpha_idx], index=feat_sorted, columns=col_labels)
        annot_hm = pd.DataFrame(
            np.where(sig[alpha_idx], "•", ""),
            index=feat_sorted,
            columns=col_labels,
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            coef_hm,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            annot=annot_hm,
            fmt="s",
            annot_kws={"fontsize": 12, "fontweight": "bold"},
            cbar_kws={"label": "Posterior mean (standardized)"},
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_xlabel("Acoustic component")
        ax.set_ylabel("")

        stem = f"{IMG_DIR}/figS4a_nmf_phoible_heatmap"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        print(f"Saved {stem}.pdf")
        plt.show()


# Supp Fig. 4b
def _darken(hex_color, factor=0.7):
    r, g, b = to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def plot_phoible_forest(feat_labels, coef, ci_lo, ci_hi, sig, r2):
    forest_colors = [_darken(c) for c in PALETTE[:K]]

    with plt.style.context(DEFAULT_STYLE):
        alpha_order = np.argsort(feat_labels)[::-1]
        sorted_names = [feat_labels[i] for i in alpha_order]
        n_feat = len(feat_labels)

        ncols = 3
        nrows = (K + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(8, 2.5 * nrows), sharey=True, sharex=True
        )

        for j in range(K):
            ax = axes.flat[j]
            for rank, fi in enumerate(alpha_order):
                is_sig = sig[fi, j]
                color = forest_colors[j] if is_sig else "#999999"
                a = 1.0 if is_sig else 0.3
                ax.plot(
                    [ci_lo[fi, j], ci_hi[fi, j]],
                    [rank, rank],
                    color=color,
                    alpha=a,
                    linewidth=1.2,
                    solid_capstyle="round",
                )
                ax.plot(coef[fi, j], rank, "o", color=color, alpha=a, markersize=2.5)

            ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.set_title(
                f"Comp {j + 1}\n({NMF_COMP_LABELS[j]})", fontsize=8, linespacing=1.4
            )
            ax.set_yticks(range(n_feat))
            if j % ncols == 0:
                ax.set_yticklabels(sorted_names, fontsize=7)
                row_start = (j // ncols) * ncols
                row_end = min(row_start + ncols, K)
                for rank, fi in enumerate(alpha_order):
                    any_sig = np.any(sig[fi, row_start:row_end])
                    ax.get_yticklabels()[rank].set_fontweight(
                        "bold" if any_sig else "normal"
                    )
            ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        for j in range(K, len(axes.flat)):
            axes.flat[j].set_visible(False)

        fig.supxlabel("Posterior mean (standardized)")

        stem = f"{IMG_DIR}/figS4b_nmf_phoible_forest"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        print(f"Saved {stem}.pdf")
        plt.show()


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)

    # Fig. 1b
    P_sorted, labels_sorted = load_nmf()
    print(f"Loaded Q matrix: {len(labels_sorted)} languages x {K} components")
    plot_structure(P_sorted, labels_sorted)

    # Supp Fig. 3
    df_ce = load_ce()
    print(f"Loaded cross-entropy for K={df_ce['K'].min()}..{df_ce['K'].max()}")
    plot_ce(df_ce)

    # Supp Fig. 4
    feat_labels, coef, ci_lo, ci_hi, sig, r2 = load_brms()
    print(f"Loaded brms: {len(feat_labels)} features x {K} components")
    plot_phoible_heatmap(feat_labels, coef, sig)
    plot_phoible_forest(feat_labels, coef, ci_lo, ci_hi, sig, r2)
