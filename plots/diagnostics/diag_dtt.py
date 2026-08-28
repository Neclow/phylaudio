"""Disparity-through-time (DTT) plot for the continuous BM data."""

import dendropy
import matplotlib.pyplot as plt
import numpy as np
import torch

from plots._config import DEFAULT_STYLE, FIG2_COLOR_SPEECH

MCC_FILE = "data/trees/contraband/v7zqs3lv/input_v16f_42.mcc.tree"
TRAITS_FILE = "data/trees/continuous/gelu/v7zqs3lv/mean_embeddings.pt"
LABELS_FILE = "data/trees/contraband/v7zqs3lv/input_v16f.xml"
IMG_DIR = "img_v3/fig2"


def get_taxon_order_from_tree(tree):
    """Return taxon labels in tree tip order."""
    return [leaf.taxon.label for leaf in tree.leaf_node_iter()]


def get_taxon_order_from_xml(xml_path):
    """Extract taxon order from the XML alignment (sequence element ids)."""
    import re

    taxa = []
    with open(xml_path) as f:
        for line in f:
            m = re.search(r'<sequence id="seq_([^"]+)"', line)
            if m:
                taxa.append(m.group(1))
            if len(taxa) >= 60:
                break
    return taxa


def subclade_disparity(traits, indices):
    """Mean pairwise Euclidean distance within a set of tip indices."""
    sub = traits[indices]
    n = len(sub)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.sum((sub[i] - sub[j]) ** 2))
    return np.mean(dists)


def compute_dtt(tree, traits, taxon_to_idx):
    """Compute DTT: mean relative subclade disparity at each node depth."""
    tree.calc_node_ages()
    root_age = float(tree.seed_node.age)

    total_disp = subclade_disparity(traits, list(range(len(traits))))

    node_depths = []
    node_disparities = []

    for nd in tree.preorder_node_iter():
        if nd.is_leaf():
            continue
        leaves = [leaf.taxon.label for leaf in nd.leaf_iter()]
        indices = [taxon_to_idx[l] for l in leaves if l in taxon_to_idx]
        if len(indices) < 2:
            continue

        relative_time = float(nd.age) / root_age
        disp = subclade_disparity(traits, indices)
        node_depths.append(relative_time)
        node_disparities.append(disp / total_disp if total_disp > 0 else 0)

    return np.array(node_depths), np.array(node_disparities)


def dtt_expected_bm(n_tips, n_points=200):
    """Expected DTT under constant-rate BM: linearly declining."""
    t = np.linspace(0, 1, n_points)
    return t, t


if __name__ == "__main__":
    tree = dendropy.Tree.get(
        path=MCC_FILE,
        schema="nexus",
        preserve_underscores=True,
        extract_comment_metadata=True,
    )

    traits = torch.load(TRAITS_FILE, weights_only=True, map_location="cpu").numpy()
    xml_taxa = get_taxon_order_from_xml(LABELS_FILE)
    tree_taxa = get_taxon_order_from_tree(tree)

    print(f"Traits shape: {traits.shape}")
    print(f"XML taxa: {len(xml_taxa)}, Tree taxa: {len(tree_taxa)}")

    taxon_to_trait_idx = {name: i for i, name in enumerate(xml_taxa)}
    taxon_to_idx = {}
    for name in tree_taxa:
        if name in taxon_to_trait_idx:
            taxon_to_idx[name] = taxon_to_trait_idx[name]
    print(f"Matched: {len(taxon_to_idx)} taxa")

    node_depths, node_disps = compute_dtt(tree, traits, taxon_to_idx)

    # Sort and compute running mean for smooth DTT line
    order = np.argsort(node_depths)
    nd_sorted = node_depths[order]
    disp_sorted = node_disps[order]

    t_bm, dtt_bm = dtt_expected_bm(len(tree_taxa))

    tree.calc_node_ages()
    root_age = float(tree.seed_node.age)
    print(f"Root age: {root_age:.2f} ka")

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        ax.plot(
            t_bm, dtt_bm,
            ls="--", color="grey", lw=1, label="Expected (BM)", zorder=1,
        )

        ax.scatter(
            nd_sorted, disp_sorted,
            s=15, c=FIG2_COLOR_SPEECH, edgecolors="none", alpha=0.6, zorder=3,
        )

        # LOESS-like smoothing via rolling mean
        window = max(3, len(nd_sorted) // 8)
        if len(nd_sorted) >= window:
            cumsum = np.cumsum(np.insert(disp_sorted, 0, 0))
            smooth = (cumsum[window:] - cumsum[:-window]) / window
            t_smooth = nd_sorted[window // 2 : window // 2 + len(smooth)]
            ax.plot(
                t_smooth, smooth,
                color=FIG2_COLOR_SPEECH, lw=2, label="Observed DTT", zorder=4,
            )

        ax.set_xlabel("Relative time (root = 1, tips = 0)")
        ax.set_ylabel("Mean relative subclade disparity")
        ax.set_xlim(1.05, -0.05)
        ax.set_ylim(-0.05, 1.3)
        ax.legend(fontsize=7, loc="upper right", frameon=False)

        # Add absolute time axis on top
        ax2 = ax.twiny()
        ticks_rel = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ticks_rel)
        ax2.set_xticklabels([f"{t * root_age:.1f}" for t in ticks_rel], fontsize=7)
        ax2.set_xlabel("Age (ka BP)", fontsize=8)

        plt.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/diag_dtt.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved to {IMG_DIR}/diag_dtt.{{pdf,svg}}")
        plt.show()
