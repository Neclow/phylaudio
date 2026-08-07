"""Figure 2b / Supp Fig S5b / S6: rate over time + speech vs cognate scatter."""

# pylint: disable=redefined-outer-name, invalid-name

import json
import os

import dendropy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from scipy import stats

from src._config import DEFAULT_METADATA_DIR

from ._config import (
    BURNIN_FRAC,
    COGNATE_BEAST_DIR,
    DEFAULT_IMG_DIR,
    DEFAULT_STYLE,
    FIG2_COLOR_COGNATE,
    FIG2_COLOR_SPEECH,
    FIG2_SIZE,
    SPEECH_BEAST_DIR,
)

EVENTS = [
    ("Yamnaya", 5.3, 4.6, "#c2945a"),
    ("Corded Ware", 4.9, 4.35, "#8aaa5e"),
    ("Indus Valley Civ.", 5.3, 3.3, "#b07aa1"),
    ("BMAC", 4.4, 3.6, "#d4a06a"),
    ("Chariots", 4.1, 3.5, "#7297b5"),
]
IMG_DIR = f"{DEFAULT_IMG_DIR}/fig2"
SPEECH_TREES_FILE = f"{SPEECH_BEAST_DIR}/input_v1_101.trees"
COGNATE_TREES_FILE = f"{COGNATE_BEAST_DIR}/prunedtomodern.trees"
NTIMES = 200
MAX_TREES = 2500


def smooth_nan_1d(x, window=15):
    x = np.asarray(x, float)
    if window <= 1:
        return x
    w = int(window) | 1
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return x
    x_filled = np.where(mask, x, 0.0)
    kernel = np.ones(w)
    num = np.convolve(x_filled, kernel, mode="same")
    den = np.convolve(mask.astype(float), kernel, mode="same")
    return num / np.where(den == 0, np.nan, den)


def interp_nan_1d(y):
    y = np.asarray(y, float)
    x = np.arange(len(y))
    m = np.isfinite(y)
    if m.sum() == 0:
        return y
    y2 = y.copy()
    y2[~m] = np.interp(x[~m], x[m], y[m])
    return y2


def _extract_segments(tree):
    tree.calc_node_ages()
    T0, T1, R = [], [], []
    for nd in tree.preorder_node_iter():
        if nd.parent_node is None:
            continue
        tp, tc = float(nd.parent_node.age), float(nd.age)
        if tp <= tc:
            continue
        r = nd.annotations.get_value("rate")
        if r is None and hasattr(nd, "edge"):
            r = nd.edge.annotations.get_value("rate")
        r = float(r) if r is not None else np.nan
        T0.append(tc)
        T1.append(tp)
        R.append(r)
    T0, T1, R = np.asarray(T0), np.asarray(T1), np.asarray(R)
    good = np.isfinite(R) & (T1 > T0)
    return T0[good], T1[good], R[good]


def _rates_over_time_slices(trees, t_grid, min_segments=1):
    rates = np.full((len(trees), len(t_grid)), np.nan)
    for i, tr in enumerate(trees):
        T0, T1, R = _extract_segments(tr)
        for j, t in enumerate(t_grid):
            msk = (T0 < t) & (T1 > t)
            if msk.sum() >= min_segments:
                rates[i, j] = R[msk].mean()
    return rates


def _alpha_cmap(base_color, max_alpha=0.6):
    return LinearSegmentedColormap.from_list(
        f"alpha_{base_color}",
        [to_rgba(base_color, 0.0), to_rgba(base_color, max_alpha)],
    )


def _add_alpha_band(
    ax,
    t_grid,
    mat,
    counts,
    base_color,
    label,
    norm_counts,
    ylow,
    yhigh,
    draw_mean=True,
    zorder=1,
):
    q_lo = np.nanpercentile(mat, 2.5, axis=0)
    q_hi = np.nanpercentile(mat, 97.5, axis=0)

    band = ax.fill_between(t_grid, q_lo, q_hi, color="none", zorder=zorder)

    cnt = smooth_nan_1d(counts.astype(float), window=15)
    cnt = interp_nan_1d(cnt)
    cnt = np.maximum(cnt, 0.0)

    n_rows = 200
    cnt_img = np.tile(cnt, (n_rows, 1))

    im = ax.imshow(
        cnt_img,
        extent=(t_grid.min(), t_grid.max(), ylow, yhigh),
        origin="lower",
        aspect="auto",
        cmap=_alpha_cmap(base_color),
        norm=norm_counts,
        zorder=zorder,
    )
    if len(band.get_paths()) > 0:
        im.set_clip_path(band.get_paths()[0], transform=ax.transData)

    if draw_mean:
        mean = np.nanmean(mat, axis=0)
        mean_valid = np.isfinite(mean)
        t_v = t_grid[mean_valid]
        m_v = mean[mean_valid]
        c_v = cnt[mean_valid]
        c_max = np.nanmax(c_v) if np.nanmax(c_v) > 0 else 1.0
        for k in range(len(t_v) - 1):
            alpha_k = float(np.clip(c_v[k] / c_max, 0.05, 1.0))
            ax.plot(
                t_v[k : k + 2],
                m_v[k : k + 2],
                lw=2.5,
                color=(*to_rgba("white")[:3], alpha_k),
                zorder=zorder + 1.4,
                solid_capstyle="round",
            )
            ax.plot(
                t_v[k : k + 2],
                m_v[k : k + 2],
                lw=1.2,
                color=(*to_rgba(base_color)[:3], alpha_k),
                zorder=zorder + 1.5,
                solid_capstyle="round",
            )
        ax.plot([], [], lw=1.5, color=base_color, label=f"{label}")
    return im


def _select_tree_indices(n_total, burnin_frac, max_trees, seed=42):
    n_burnin = int(n_total * burnin_frac)
    n_post = n_total - n_burnin
    if max_trees and n_post > max_trees:
        rng = np.random.default_rng(seed)
        keep = set(np.sort(rng.choice(n_post, max_trees, replace=False)) + n_burnin)
    else:
        keep = set(range(n_burnin, n_total))
    return keep


def load_rates(trees_file):
    print(f"Reading {trees_file}...")

    # Pass 1: count trees (text scan, no parsing)
    n_total = 0
    with open(trees_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().lower().startswith("tree "):
                n_total += 1

    keep = _select_tree_indices(n_total, BURNIN_FRAC, MAX_TREES)
    print(
        f"  {n_total} trees, selecting {len(keep)} (burn-in {int(n_total * BURNIN_FRAC)})"
    )

    # Pass 2: extract preamble + selected tree strings
    preamble_lines = []
    tree_strings = []
    tree_idx = 0
    in_trees_block = False
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

    tmax = 0.0
    for tr in trees:
        tr.calc_node_ages()
        tmax = max(tmax, float(tr.seed_node.age))
    t_grid = np.linspace(0.05, tmax, NTIMES)
    print("Computing rates over time slices...")
    raw_rates = _rates_over_time_slices(trees, t_grid)

    return t_grid, raw_rates, tmax


def _plot_rates(
    ax, t_grid, raw_rates, tmax, color, label, norm_counts, ylow, yhigh, zorder=1
):
    norm_rates = (raw_rates - np.nanmean(raw_rates, axis=1, keepdims=True)) / np.nanstd(
        raw_rates, axis=1, keepdims=True, ddof=1
    )
    counts = np.sum(np.isfinite(norm_rates), axis=0) / norm_rates.shape[0] * 100
    _add_alpha_band(
        ax,
        t_grid,
        norm_rates,
        counts,
        color,
        label,
        norm_counts,
        ylow,
        yhigh,
        zorder=zorder,
    )

    ax.axhline(0, color="black", lw=0.5, zorder=0)
    for t in range(1, int(tmax) + 1):
        ax.axvline(t, color="#cccccc", lw=0.3, ls=":", zorder=0)


def _plot_events(ax, ylow, yhigh):
    bar_y_top = yhigh
    bar_h = (yhigh - ylow) * 0.035
    bar_gap = bar_h * 0.15
    for i, (name, t_start, t_end, ecolor) in enumerate(EVENTS):
        y_top_i = bar_y_top - i * (bar_h + bar_gap)
        ax.barh(
            y_top_i - bar_h / 2,
            width=t_start - t_end,
            left=t_end,
            height=bar_h,
            color=ecolor,
            alpha=0.7,
            edgecolor=ecolor,
            linewidth=0.5,
            zorder=5,
        )
        ax.text(
            t_end - 0.03,
            y_top_i - bar_h / 2,
            name,
            ha="right",
            va="center",
            fontsize=5,
            color="black",
            zorder=6,
        )


def plot_rates(t_grid, raw_rates, tmax, color, label, output_name):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=FIG2_SIZE)

        norm_counts = Normalize(vmin=0, vmax=100)
        ylow, yhigh = -3, 3
        ax.set_ylim(ylow, yhigh)

        _plot_rates(ax, t_grid, raw_rates, tmax, color, label, norm_counts, ylow, yhigh)
        _plot_events(ax, ylow, yhigh)

        ax.set_xlim(tmax, 0.0)
        ax.set_ylabel("Standardised rate (z-score)", fontsize=8)
        ax.set_xlabel("Age (ka BP)", fontsize=8)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.spines[["top", "right"]].set_visible(False)

        sm = cm.ScalarMappable(norm=norm_counts, cmap=_alpha_cmap(color))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.8, aspect=20)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("% posterior trees\nat time t", fontsize=6)

        ax.legend(fontsize=6, loc="upper left", frameon=False)

        plt.tight_layout()
        for ext in ("pdf", "svg"):
            output_path = f"{IMG_DIR}/{output_name}.{ext}"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {IMG_DIR}/{output_name}.{{pdf,svg}}")
        plt.show()


def plot_rates_overlaid(
    t_grid_s,
    raw_rates_s,
    tmax_s,
    t_grid_c,
    raw_rates_c,
    tmax_c,
    output_name="figS5b_rates",
):
    tmax = max(tmax_s, tmax_c)
    color_s, color_c = FIG2_COLOR_SPEECH, FIG2_COLOR_COGNATE

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=FIG2_SIZE)

        norm_counts = Normalize(vmin=0, vmax=100)
        ylow, yhigh = -3, 3
        ax.set_ylim(ylow, yhigh)

        _plot_rates(
            ax,
            t_grid_c,
            raw_rates_c,
            tmax,
            color_c,
            "Cognates",
            norm_counts,
            ylow,
            yhigh,
            zorder=1,
        )
        _plot_rates(
            ax,
            t_grid_s,
            raw_rates_s,
            tmax,
            color_s,
            "Speech",
            norm_counts,
            ylow,
            yhigh,
            zorder=3,
        )
        _plot_events(ax, ylow, yhigh)

        ax.set_xlim(tmax, 0.0)
        ax.set_ylabel("Standardised rate (z-score)", fontsize=8)
        ax.set_xlabel("Age (ka BP)", fontsize=8)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6, loc="upper left", frameon=False)

        sm = cm.ScalarMappable(norm=norm_counts, cmap=_alpha_cmap(color_s))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.8, aspect=20)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("% posterior trees\nat time t", fontsize=7)

        plt.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/{output_name}.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved figure to {IMG_DIR}/{output_name}.{{pdf,svg}}")
        plt.show()


def plot_rate_scatter(output_name="figS6_speech_vs_cognate_rates"):
    speech = pd.read_csv(f"{SPEECH_BEAST_DIR}/metadata_with_inventory.csv")
    cognate = pd.read_csv(f"{COGNATE_BEAST_DIR}/metadata_with_inventory.csv")

    with open(
        f"{DEFAULT_METADATA_DIR}/fleurs-r/languages.json", "r", encoding="utf-8"
    ) as f:
        langs = json.load(f)
    iecor_to_fleurs = {
        v["iecor"]: v["fleurs"]
        for v in langs.values()
        if "iecor" in v and v["iecor"] != v["fleurs"]
    }
    cognate["language"] = cognate["language"].map(lambda x: iecor_to_fleurs.get(x, x))

    s = speech[["language", "rate_median"]].rename(
        columns={"rate_median": "speech_rate"}
    )
    c = cognate[["language", "rate_median"]].rename(
        columns={"rate_median": "cognate_rate"}
    )
    merged = s.merge(c, on="language")
    rho, pval = stats.spearmanr(merged["speech_rate"], merged["cognate_rate"])

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        ax.scatter(
            merged["speech_rate"],
            merged["cognate_rate"],
            c="#555555",
            s=20,
            edgecolors="none",
            zorder=4,
        )

        for _, row in merged.iterrows():
            ax.annotate(
                row["language"],
                (row["speech_rate"], row["cognate_rate"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="#333333",
                clip_on=True,
            )

        x_vals = merged["speech_rate"].values
        y_vals = merged["cognate_rate"].values
        x_pad = (x_vals.max() - x_vals.min()) * 0.05
        y_pad = (y_vals.max() - y_vals.min()) * 0.06
        ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
        ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

        ax.set_xlabel("Median rate (speech)")
        ax.set_ylabel("Median rate (cognate)")
        ax.spines[["top", "right"]].set_visible(False)

        ax.text(
            0.98,
            0.02,
            f"Spearman $\\rho$ = {rho:.3f}, p = {pval:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            style="italic",
            color="#333333",
        )

        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/{output_name}.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved figure to {IMG_DIR}/{output_name}.{{pdf,svg}}")
        plt.show()


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)

    # Fig 2b: speech rates
    t_grid, raw_rates, tmax = load_rates(SPEECH_TREES_FILE)
    plot_rates(
        t_grid, raw_rates, tmax, FIG2_COLOR_SPEECH, "Speech", "fig2b_rate_standardized"
    )

    # Load cognate rates (reused by S4b and S5)
    t_grid_c, raw_rates_c, tmax_c = load_rates(COGNATE_TREES_FILE)

    # Fig S5b: overlaid speech + cognate rates
    plot_rates_overlaid(t_grid, raw_rates, tmax, t_grid_c, raw_rates_c, tmax_c)

    # Fig S6: speech vs cognate rate scatter
    plot_rate_scatter()
