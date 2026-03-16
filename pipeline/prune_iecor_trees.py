#!/usr/bin/env python3
"""Prune historical (non-modern) taxa from IECoR posterior trees.

Reads the combined IECoR posterior tree file, identifies taxa whose
tip-to-root distance differs from the maximum (i.e. historical/ancient
languages), removes them, and writes a pruned tree file.

Ported from post_beast/speech_phylo_v4/final_tree_pruning.ipynb.
"""

import os
import concurrent.futures
from functools import partial

import dendropy


# ─── Paths ────────────────────────────────────────────────────────────────────
INPUT_PATH = "data/trees/beast/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined.trees"
OUTPUT_PATH = "data/trees/beast/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined_prunedtomodern.trees"


def prune_one_tree(tr, labels):
    """Worker function to prune a single tree."""
    pruned_tr = tr.clone(depth=2)
    pruned_tr.retain_taxa_with_labels(labels)
    pruned_tr.suppress_unifurcations()
    return pruned_tr


def main():
    print(f"Loading trees from {INPUT_PATH} (this may take >10 minutes)...")
    trees = dendropy.TreeList.get(
        path=INPUT_PATH,
        schema="nexus",
        preserve_underscores=True,
    )
    print(f"  Loaded {len(trees)} trees.")

    # Identify historical taxa from reference tree
    ref_tree = trees[0]
    leaf_distances = {
        leaf.taxon.label: leaf.distance_from_root()
        for leaf in ref_tree.leaf_node_iter()
    }
    max_dist = max(leaf_distances.values())

    historical_labels = {
        label for label, dist in leaf_distances.items()
        if (max_dist - dist) > 1e-6
    }
    labels_to_keep = set(leaf_distances.keys()) - historical_labels

    print(f"Removing {len(historical_labels)} historical taxa. Keeping {len(labels_to_keep)} modern taxa.")

    # Parallel pruning
    num_workers = min(64, os.cpu_count() - 4)
    print(f"Pruning with {num_workers} worker processes...")

    task = partial(prune_one_tree, labels=labels_to_keep)
    n_trees = len(trees)
    pruned_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, result in enumerate(executor.map(task, trees, chunksize=100)):
            pruned_list.append(result)
            if (i + 1) % 500 == 0 or (i + 1) == n_trees:
                print(f"  {i + 1}/{n_trees} trees pruned", flush=True)

    trees_pruned = dendropy.TreeList(pruned_list)
    print(f"Successfully pruned to {len(trees_pruned)} trees.")

    trees_pruned.write(path=OUTPUT_PATH, schema="nexus")
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
