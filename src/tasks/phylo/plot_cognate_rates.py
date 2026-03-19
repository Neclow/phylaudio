#!/usr/bin/env python3
"""Cognate rate over time with credible intervals from BEAST posterior trees.

Same style as plot_figure2_rates.py (speech), but for the cognate tree.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba
import matplotlib.cm as cm

# ── Configuration ──────────────────────────────────────────────────────────────
TREES_FILE = "data/trees/beast/iecor/prunedtomodern.trees"
OUTPUT_DIR = "data/phyloregression/figures"
BURNIN = 1000
NTIMES = 200
MAX_TREES = 2000

# ── Import shared utilities from plot_figure2_rates ────────────────────────────
from src.tasks.phylo.plot_figure2_rates import (
    smooth_nan_1d, interp_nan_1d, _extract_segments,
    _rates_over_time_slices, _alpha_cmap, _add_alpha_band,
)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import dendropy

    print("Reading cognate trees (this may take several minutes)...")
    trees = dendropy.TreeList.get(
        path=TREES_FILE, schema="nexus",
        preserve_underscores=True, extract_comment_metadata=True,
    )
    print(f"  Read {len(trees)} trees")

    trees_post = trees[BURNIN:]
    print(f"  After burn-in: {len(trees_post)} trees")

    if MAX_TREES and len(trees_post) > MAX_TREES:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(trees_post), MAX_TREES, replace=False))
        trees_post = dendropy.TreeList([trees_post[i] for i in idx])
        print(f"  Subsampled to {len(trees_post)} trees")

    tmax = 0.0
    for tr in trees_post:
        tr.calc_node_ages()
        tmax = max(tmax, float(tr.seed_node.age))
    print(f"  Max root age: {tmax:.2f} ka")

    t_grid = np.linspace(0.05, tmax, NTIMES)
    print("Computing rates over time slices...")
    raw_rates = _rates_over_time_slices(trees_post, t_grid)

    ref_rate = raw_rates[:, 0:1]
    pct = (raw_rates - ref_rate) / ref_rate * 100

    norm_rates = (raw_rates - np.nanmean(raw_rates, axis=1, keepdims=True)) / \
                 np.nanstd(raw_rates, axis=1, keepdims=True, ddof=1)

    # ── Nature-style plot ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })

    color = "#7ad151"

    # -- Standardized rate plot --
    fig, ax = plt.subplots(figsize=(7.2, 2.5))

    counts = np.sum(np.isfinite(norm_rates), axis=0) / norm_rates.shape[0] * 100
    norm_counts = Normalize(vmin=0, vmax=100)
    ylow, yhigh = -3, 3
    ax.set_ylim(ylow, yhigh)

    _add_alpha_band(ax, t_grid, norm_rates, counts, color, "Cognates",
                    norm_counts, ylow, yhigh, zorder=1)

    ax.axhline(0, color="black", lw=0.5, zorder=0)
    for t in range(1, int(tmax) + 1):
        ax.axvline(t, color="#cccccc", lw=0.3, ls=":", zorder=0)

    # Historical event bars (spans)
    EVENTS = [
        ("Yamnaya",          5.3, 4.6, "#c2945a"),
        ("Corded Ware",      4.9, 4.35, "#8aaa5e"),
        ("Indus Valley Civ.", 5.3, 3.3, "#b07aa1"),
        ("BMAC",             4.4, 3.6, "#d4a06a"),
        ("Chariots",         4.1, 3.5, "#7297b5"),
    ]
    bar_y_top = yhigh
    bar_h = (yhigh - ylow) * 0.035
    bar_gap = bar_h * 0.15
    for i, (name, t_start, t_end, ecolor) in enumerate(EVENTS):
        y_top_i = bar_y_top - i * (bar_h + bar_gap)
        ax.barh(y_top_i - bar_h / 2, width=t_start - t_end, left=t_end,
                height=bar_h, color=ecolor, alpha=0.7, edgecolor=ecolor,
                linewidth=0.5, zorder=5)
        ax.text(t_end - 0.03, y_top_i - bar_h / 2, name,
                ha="right", va="center", fontsize=5, color="black", zorder=6)

    ax.set_xlim(tmax, 0.0)
    ax.set_ylim(ylow, yhigh)
    ax.set_ylabel("Standardised rate (z-score)", fontsize=8)
    ax.set_xlabel("Time (ka BP)", fontsize=8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.spines[["top", "right"]].set_visible(False)

    sm = cm.ScalarMappable(norm=norm_counts, cmap=_alpha_cmap(color))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.8, aspect=20)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("% posterior trees\nat time t", fontsize=6)

    ax.legend(fontsize=6, loc="upper left", frameon=False)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_std = os.path.join(OUTPUT_DIR, "panel_b_rate_standardized_cognate.pdf")
    fig.savefig(out_std, dpi=300, bbox_inches="tight")
    fig.savefig(out_std.replace(".pdf", ".svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_std}")

    # -- Percentage change plot --
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    counts_pct = np.sum(np.isfinite(pct), axis=0) / pct.shape[0] * 100
    ylow_p, yhigh_p = -100, 100

    _add_alpha_band(ax, t_grid, pct, counts_pct, color, "Cognates",
                    norm_counts, ylow_p, yhigh_p, zorder=1)

    ax.axhline(0, color="black", lw=0.5, zorder=0)
    ax.set_xlim(t_grid.max(), t_grid.min())
    ax.set_ylim(ylow_p, yhigh_p)
    ax.set_ylabel("% change in rate\n(relative to present)", fontsize=8)
    ax.set_xlabel("Time (ka BP)", fontsize=8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.spines[["top", "right"]].set_visible(False)

    sm2 = cm.ScalarMappable(norm=norm_counts, cmap=_alpha_cmap(color))
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=ax, pad=0.02, shrink=0.8, aspect=20)
    cbar2.ax.tick_params(labelsize=6)
    cbar2.set_label("% posterior trees\nat time t", fontsize=6)

    ax.legend(fontsize=6, loc="upper left", frameon=False)

    plt.tight_layout()
    out_pct = os.path.join(OUTPUT_DIR, "panel_b_rate_pctchange_cognate.pdf")
    fig.savefig(out_pct, dpi=300, bbox_inches="tight")
    fig.savefig(out_pct.replace(".pdf", ".svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_pct}")

    np.savez(
        os.path.join(OUTPUT_DIR, "panel_b_data_cognate.npz"),
        t_grid=t_grid, raw_rates=raw_rates, norm_rates=norm_rates,
        pct_change=pct, tmax=tmax,
    )
    print("  Saved intermediate data: panel_b_data_cognate.npz")


if __name__ == "__main__":
    main()
