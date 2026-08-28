"""Diagnostic: correlation matrix of terminal branch rate, pre-training hours, within-language variance."""

import json

import dendropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from plots._config import DEFAULT_STYLE
from src._config import _FLEURS_LANG_TO_ID

TREES_FILE = "data/trees/contraband/v7zqs3lv/input_v16f_42.trees"
POP_TRAITS_FILE = "data/trees/continuous/gelu/v7zqs3lv/population_traits.pt"
POP_LABELS_FILE = "data/trees/continuous/gelu/v7zqs3lv/population_labels.json"
HOURS_FILE = "tmp/dataset_hours_comparison.csv"
BURNIN_FRAC = 0.10
MAX_TREES = 2500
IMG_DIR = "img_v3/fig2"

# FLEURS name -> ISO 639-1
NAME_TO_ISO = {name: code for name, code in _FLEURS_LANG_TO_ID.items()}
# FLEURS uses nb (Bokmål) but the hours CSV uses no (Norwegian macro)
_ISO_ALIAS = {"nb": "no"}


def _select_tree_indices(n_total, burnin_frac, max_trees, seed=42):
    n_burnin = int(n_total * burnin_frac)
    n_post = n_total - n_burnin
    if max_trees and n_post > max_trees:
        rng = np.random.default_rng(seed)
        keep = set(np.sort(rng.choice(n_post, max_trees, replace=False)) + n_burnin)
    else:
        keep = set(range(n_burnin, n_total))
    return keep


def load_trees(trees_file):
    n_total = 0
    with open(trees_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().lower().startswith("tree "):
                n_total += 1

    keep = _select_tree_indices(n_total, BURNIN_FRAC, MAX_TREES)

    preamble, tree_strings = [], []
    tree_idx, in_trees = 0, False
    with open(trees_file) as f:
        for line in f:
            s = line.strip().lower()
            if s.startswith("begin trees"):
                in_trees = True
                preamble.append(line)
            elif not in_trees:
                preamble.append(line)
            elif s.startswith("tree "):
                if tree_idx in keep:
                    tree_strings.append(line)
                tree_idx += 1
            elif s == "end;":
                break
            else:
                preamble.append(line)

    trees = dendropy.TreeList.get(
        data="".join(preamble) + "".join(tree_strings) + "End;\n",
        schema="nexus",
        preserve_underscores=True,
        extract_comment_metadata=True,
    )
    return trees


def extract_tip_rates(trees):
    taxon_rates = {}
    for tr in trees:
        for leaf in tr.leaf_node_iter():
            label = leaf.taxon.label
            r = leaf.annotations.get_value("rate")
            if r is None and hasattr(leaf, "edge"):
                r = leaf.edge.annotations.get_value("rate")
            if r is not None:
                taxon_rates.setdefault(label, []).append(float(r))
    return {k: np.median(v) for k, v in taxon_rates.items()}


def compute_within_lang_variance(pop_traits_file, pop_labels_file):
    X = torch.load(pop_traits_file, weights_only=True).numpy()
    with open(pop_labels_file) as f:
        labels = json.load(f)
    label_arr = np.array(labels)
    variances = {}
    for lang in sorted(set(labels)):
        mask = label_arr == lang
        variances[lang] = np.mean(np.var(X[mask], axis=0, ddof=1))
    return variances


def load_pretrain_hours():
    df = pd.read_csv(HOURS_FILE)
    return dict(zip(df["code"], df["hours_xlsr_pretrain"].fillna(0)))


if __name__ == "__main__":
    print("Loading trees...")
    trees = load_trees(TREES_FILE)
    tip_rates = extract_tip_rates(trees)
    within_var = compute_within_lang_variance(POP_TRAITS_FILE, POP_LABELS_FILE)
    pretrain_hours = load_pretrain_hours()

    rows = []
    for lang in sorted(tip_rates):
        iso = NAME_TO_ISO.get(lang)
        if iso is None:
            continue
        hours = pretrain_hours.get(_ISO_ALIAS.get(iso, iso), 0.0)
        rows.append({
            "language": lang,
            "rate": tip_rates[lang],
            "within_var": within_var.get(lang, np.nan),
            "pretrain_hours": hours,
            "log_pretrain_hours": np.log1p(hours),
        })

    df = pd.DataFrame(rows).dropna()
    print(f"{len(df)} languages matched\n")

    cols = ["rate", "within_var", "log_pretrain_hours"]
    labels = ["Terminal branch\nrate (median)", "Within-language\nvariance", "log(1 + pre-training\nhours)"]

    n = len(cols)
    rho_mat = np.ones((n, n))
    p_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, p = stats.spearmanr(df[cols[i]], df[cols[j]])
            rho_mat[i, j] = rho_mat[j, i] = r
            p_mat[i, j] = p_mat[j, i] = p

    print("Spearman correlation matrix:")
    for i in range(n):
        for j in range(n):
            sig = "*" if p_mat[i, j] < 0.05 and i != j else " "
            print(f"  {rho_mat[i,j]:+.3f}{sig}", end="")
        print(f"  {labels[i].replace(chr(10), ' ')}")
    print()

    print("Per-pair details:")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"  {cols[i]} vs {cols[j]}: rho={rho_mat[i,j]:+.3f}, p={p_mat[i,j]:.4f}")

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(rho_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        for i in range(n):
            for j in range(n):
                sig = "" if i == j else ("*" if p_mat[i, j] < 0.05 else "")
                ax.text(j, i, f"{rho_mat[i,j]:+.2f}{sig}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Spearman $\\rho$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        plt.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/diag_corr_matrix.{ext}", dpi=300, bbox_inches="tight")
        print(f"\nSaved to {IMG_DIR}/diag_corr_matrix.{{pdf,svg}}")
        plt.show()
