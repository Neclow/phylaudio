"""Diagnostic: per-language terminal branch rate vs within-language embedding variance."""

import json

import dendropy
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from plots._config import DEFAULT_STYLE

TREES_FILE = "data/trees/contraband/v7zqs3lv/input_v16f_42.trees"
POP_TRAITS_FILE = "data/trees/continuous/gelu/v7zqs3lv/population_traits.pt"
POP_LABELS_FILE = "data/trees/continuous/gelu/v7zqs3lv/population_labels.json"
BURNIN_FRAC = 0.10
MAX_TREES = 2500
IMG_DIR = "img_v3/fig2"


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
    print(f"Reading {trees_file}...")
    n_total = 0
    with open(trees_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().lower().startswith("tree "):
                n_total += 1

    keep = _select_tree_indices(n_total, BURNIN_FRAC, MAX_TREES)
    print(f"  {n_total} trees, selecting {len(keep)} (burn-in {int(n_total * BURNIN_FRAC)})")

    preamble_lines, tree_strings = [], []
    tree_idx, in_trees_block = 0, False
    with open(trees_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip().lower()
            if stripped.startswith("begin trees"):
                in_trees_block = True
                preamble_lines.append(line)
            elif not in_trees_block:
                preamble_lines.append(line)
            elif stripped.startswith("tree "):
                if tree_idx in keep:
                    tree_strings.append(line)
                tree_idx += 1
            elif stripped == "end;":
                break
            else:
                preamble_lines.append(line)

    nexus_str = "".join(preamble_lines) + "".join(tree_strings) + "End;\n"
    print(f"  Parsing {len(tree_strings)} trees...")
    trees = dendropy.TreeList.get(
        data=nexus_str,
        schema="nexus",
        preserve_underscores=True,
        extract_comment_metadata=True,
    )
    print(f"  Parsed {len(trees)} trees")
    return trees


def extract_tip_rates(trees):
    """Extract per-taxon terminal branch rate across posterior trees."""
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
    """Compute mean within-language variance across 256 dims."""
    X = torch.load(pop_traits_file, weights_only=True).numpy()
    with open(pop_labels_file, "r", encoding="utf-8") as f:
        labels = json.load(f)

    unique_langs = sorted(set(labels))
    label_arr = np.array(labels)
    variances = {}
    for lang in unique_langs:
        mask = label_arr == lang
        variances[lang] = np.mean(np.var(X[mask], axis=0, ddof=1))
    return variances


if __name__ == "__main__":
    trees = load_trees(TREES_FILE)
    tip_rates = extract_tip_rates(trees)
    within_var = compute_within_lang_variance(POP_TRAITS_FILE, POP_LABELS_FILE)

    shared = sorted(set(tip_rates) & set(within_var))
    print(f"\n{len(shared)} shared taxa")

    rates = np.array([tip_rates[t] for t in shared])
    variances = np.array([within_var[t] for t in shared])

    rho, pval = stats.spearmanr(variances, rates)
    print(f"Spearman rho = {rho:.3f}, p = {pval:.4f}")

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(variances, rates, s=20, c="#555555", edgecolors="none", zorder=4)
        for i, t in enumerate(shared):
            ax.annotate(
                t, (variances[i], rates[i]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=5, color="#333333", clip_on=True,
            )
        ax.set_xlabel("Mean within-language variance (256 dims)")
        ax.set_ylabel("Median terminal branch rate")
        ax.text(
            0.98, 0.02,
            f"Spearman $\\rho$ = {rho:.3f}, p = {pval:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, style="italic", color="#333333",
        )
        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/diag_tip_rate_vs_popvar.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved to {IMG_DIR}/diag_tip_rate_vs_popvar.{{pdf,svg}}")
        plt.show()
