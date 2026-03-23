#!/usr/bin/env python3
"""Panel B: Rate over time with credible intervals from BEAST posterior trees.

Adapted from final_plots.py — uses dendropy to extract per-branch rates and
computes rates through time with 95% CI bands.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba

# ── Configuration ──────────────────────────────────────────────────────────────
TREES_FILE = "data/trees/beast/input_v12_combined_resampled.trees"
OUTPUT_DIR = "data/phyloregression/figures"
BURNIN = 0  # speech trees have no burn-in
NTIMES = 200
MAX_TREES = 2000  # cap for speed; use None for all post-burnin trees

# Modern taxa to keep (same set as Panel A)
KEEP_TAXA = {
    "Assamese",
    "Bengali",
    "Oriya",
    "Nepali",
    "Gujarati",
    "Marathi",
    "Punjabi",
    "Sindhi",
    "Urdu",
    "Hindi",
    "Pashto",
    "Tajik",
    "PersianTehran",
    "ArmenianEastern",
    "Sorani-Kurdish",
    "Greek",
    "French",
    "Occitan",
    "Galician",
    "Asturian",
    "Spanish",
    "Catalan",
    "Romanian",
    "Italian",
    "Portuguese",
    "Kabuverdianu",
    "English",
    "GaelicIrish",
    "WelshNorth",
    "Dutch",
    "Afrikaans",
    "Luxembourgish",
    "German",
    "Danish",
    "Icelandic",
    "Swedish",
    "NorwegianBokmal",
    "Latvian",
    "Lithuanian",
    "Macedonian",
    "Bulgarian",
    "Slovene",
    "Serbian",
    "Bosnian",
    "Croatian",
    "Slovak",
    "Czech",
    "Polish",
    "Ukrainian",
    "Russian",
    "Belarusian",
}


# ── Utility functions (from final_plots.py) ───────────────────────────────────


def smooth_nan_1d(x, window=15):
    """NaN-safe moving average."""
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
    """Linearly interpolate across NaNs."""
    y = np.asarray(y, float)
    x = np.arange(len(y))
    m = np.isfinite(y)
    if m.sum() == 0:
        return y
    y2 = y.copy()
    y2[~m] = np.interp(x[~m], x[m], y[m])
    return y2


def _extract_segments(tree):
    """Extract per-branch [child_age, parent_age, rate]."""
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
    """Mean rate of branches spanning each timepoint, per tree."""
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
    """95% CI band with alpha encoding sample density."""
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
        from matplotlib.colors import to_rgba

        mean = np.nanmean(mat, axis=0)
        mean_valid = np.isfinite(mean)
        # Draw mean line as segments that fade with sample density
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
        # Invisible line for legend
        ax.plot([], [], lw=1.5, color=base_color, label=f"{label}")
    return im


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    import dendropy

    print("Reading trees (this may take several minutes)...")
    trees = dendropy.TreeList.get(
        path=TREES_FILE,
        schema="nexus",
        preserve_underscores=True,
        extract_comment_metadata=True,
    )
    print(f"  Read {len(trees)} trees")

    # Remove burn-in
    trees_post = trees[BURNIN:]
    print(f"  After burn-in: {len(trees_post)} trees")

    # Subsample if needed
    if MAX_TREES and len(trees_post) > MAX_TREES:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(trees_post), MAX_TREES, replace=False))
        trees_post = dendropy.TreeList([trees_post[i] for i in idx])
        print(f"  Subsampled to {len(trees_post)} trees")

    # Get tmax
    tmax = 0.0
    for tr in trees_post:
        tr.calc_node_ages()
        tmax = max(tmax, float(tr.seed_node.age))
    print(f"  Max root age: {tmax:.2f} ka")

    # Compute rates over time
    t_grid = np.linspace(0.05, tmax, NTIMES)
    print("Computing rates over time slices...")
    raw_rates = _rates_over_time_slices(trees_post, t_grid)

    # ── Panel B option 1: Percentage change relative to present ────────────
    ref_rate = raw_rates[:, 0:1]
    pct = (raw_rates - ref_rate) / ref_rate * 100

    # ── Panel B option 2: Standardized rate ────────────────────────────────
    norm_rates = (raw_rates - np.nanmean(raw_rates, axis=1, keepdims=True)) / np.nanstd(
        raw_rates, axis=1, keepdims=True, ddof=1
    )

    # ── Nature-style plot ──────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )

    color = "#414487"

    # -- Standardized rate plot --
    fig, ax = plt.subplots(figsize=(7.2, 2.5))

    counts = np.sum(np.isfinite(norm_rates), axis=0) / norm_rates.shape[0] * 100
    norm_counts = Normalize(vmin=0, vmax=100)
    ylow, yhigh = -3, 3
    ax.set_ylim(ylow, yhigh)

    _add_alpha_band(
        ax,
        t_grid,
        norm_rates,
        counts,
        color,
        "Speech",
        norm_counts,
        ylow,
        yhigh,
        zorder=1,
    )

    ax.axhline(0, color="black", lw=0.5, zorder=0)
    # Vertical guide lines at each ka
    for t in range(1, int(tmax) + 1):
        ax.axvline(t, color="#cccccc", lw=0.3, ls=":", zorder=0)

    # Historical event bars (spans)
    EVENTS = [
        ("Yamnaya", 5.3, 4.6, "#c2945a"),
        ("Corded Ware", 4.9, 4.35, "#8aaa5e"),
        ("Indus Valley Civ.", 5.3, 3.3, "#b07aa1"),
        ("BMAC", 4.4, 3.6, "#d4a06a"),
        ("Chariots", 4.1, 3.5, "#7297b5"),
    ]
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

    ax.set_xlim(tmax, 0.0)
    ax.set_ylim(ylow, yhigh)
    ax.set_ylabel("Standardised rate (z-score)", fontsize=8)
    ax.set_xlabel("Time (ka BP)", fontsize=8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.spines[["top", "right"]].set_visible(False)

    # Colorbar for sample density
    sm = cm.ScalarMappable(norm=norm_counts, cmap=_alpha_cmap(color))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.8, aspect=20)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("% posterior trees\nat time t", fontsize=6)

    legend = ax.legend(fontsize=6, loc="upper left", frameon=False)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_std = os.path.join(OUTPUT_DIR, "panel_b_rate_standardized.pdf")
    fig.savefig(out_std, dpi=300, bbox_inches="tight")
    fig.savefig(out_std.replace(".pdf", ".svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_std}")

    # -- Percentage change plot --
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    counts_pct = np.sum(np.isfinite(pct), axis=0) / pct.shape[0] * 100
    ylow_p, yhigh_p = -100, 100

    _add_alpha_band(
        ax,
        t_grid,
        pct,
        counts_pct,
        color,
        "Speech",
        norm_counts,
        ylow_p,
        yhigh_p,
        zorder=1,
    )

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
    out_pct = os.path.join(OUTPUT_DIR, "panel_b_rate_pctchange.pdf")
    fig.savefig(out_pct, dpi=300, bbox_inches="tight")
    fig.savefig(out_pct.replace(".pdf", ".svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_pct}")

    # Save data for combine script
    np.savez(
        os.path.join(OUTPUT_DIR, "panel_b_data.npz"),
        t_grid=t_grid,
        raw_rates=raw_rates,
        norm_rates=norm_rates,
        pct_change=pct,
        tmax=tmax,
    )
    print("  Saved intermediate data: panel_b_data.npz")


if __name__ == "__main__":
    main()
