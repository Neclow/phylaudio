#!/usr/bin/env python3
"""
make_figure3_geo.py  –  Publication Figure 3 (linear_geo model)

Font note: Arial is used because matplotlib can locate Arial Bold as a
separate TTF file (/System/Library/Fonts/Supplemental/Arial Bold.ttf),
whereas Helvetica / HelveticaNeue ship as .ttc collections that matplotlib
resolves to the same file for all weights, making bold unavailable.
"""

import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from scipy import stats
from shapely.geometry import Point
from shapely.geometry import box as shapely_box
from shapely.prepared import prep

from src.tasks.phylo.constants import EXCLUDE_LANGUAGES, GEOJSON_EXPANSION

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
BASE = HERE.parent  # src/tasks/phylo -> repo root
OUT_DIR = BASE / "data/phyloregression/figures"

RESULTS_DIR = BASE / "data/phyloregression/with_inventory"
DATA_DIR = BASE / "data"
GEOJSON_PATH = DATA_DIR / "metadata/fleurs-r/language_polygons.geojson"

# ─── Model config ─────────────────────────────────────────────────────────────
SPEECH_TREE = "input_v12_combined_resampled"
COGNATE_TREE = "heggarty2024_raw"
MODEL = "linear_geo"
SCMAP = "viridis"  # panel c dots
MAP_CMAP = "magma"  # panel d surface + dots
RATE_CMAP = "plasma"  # unused (legacy)


def load_language_polygons(filepath):
    gdf = gpd.read_file(filepath)
    merged = [
        {"name": n, "geometry": gdf[gdf["name"] == n].geometry.union_all()}
        for n in gdf["name"].unique()
    ]
    gdf = gpd.GeoDataFrame(merged, crs=gdf.crs)
    expand_mask = gdf["name"].isin(GEOJSON_EXPANSION)
    expanded = []
    for _, row in gdf[expand_mask].iterrows():
        for new_name in GEOJSON_EXPANSION[row["name"]]:
            r = row.copy()
            r["name"] = new_name
            expanded.append(r)
    if expanded:
        gdf = pd.concat(
            [gdf[~expand_mask], gpd.GeoDataFrame(expanded, crs=gdf.crs)],
            ignore_index=True,
        )
    return gpd.GeoDataFrame(
        [
            {"name": n, "geometry": gdf[gdf["name"] == n].geometry.union_all()}
            for n in gdf["name"].unique()
        ],
        crs=gdf.crs,
    )


# ─── Bold font (Arial has a proper bold TTF on macOS) ─────────────────────────
_PANEL_FP = FontProperties(family="sans-serif", weight="bold", size=30)
_AXIS_FP = FontProperties(family="sans-serif", weight="normal", size=15)

# ─── Display-name helpers ─────────────────────────────────────────────────────
_VAR_DISPLAY = {
    "longitude": "Longitude",
    "latitude": "Latitude",
    "log_n_speakers": "log_n_speakers",
    "n_phonemes": "n_phonemes",
    "delta": "delta",
}


def _coef_suffix_to_display(suffix):
    known = _VAR_DISPLAY
    if suffix in known:
        return known[suffix]
    for v1 in sorted(known, key=len, reverse=True):
        if suffix.startswith(v1 + "_"):
            rest = suffix[len(v1) + 1 :]
            if rest in known:
                return f"{known[v1]}×{known[rest]}"
    return None


def _extract_coefficient_data(summ_df):
    rows = []
    for _, row in summ_df.iterrows():
        for col in summ_df.columns:
            if not col.startswith("coef_"):
                continue
            suffix = col[5:]
            if suffix == "Intercept":
                continue
            lo_col, hi_col = f"ci_lower_{suffix}", f"ci_upper_{suffix}"
            if lo_col not in summ_df.columns or hi_col not in summ_df.columns:
                continue
            display = _coef_suffix_to_display(suffix)
            if display is None:
                continue
            est, lo, hi = float(row[col]), float(row[lo_col]), float(row[hi_col])
            if pd.notna(est):
                rows.append(
                    dict(
                        coefficient=display,
                        estimate=est,
                        ci_lower=lo,
                        ci_upper=hi,
                        _n_vars=1 if "×" not in display else 2,
                    )
                )
    return pd.DataFrame(rows)


# ─── Variance decomposition ───────────────────────────────────────────────────
VARIANCE_COMPONENT_ORDER = [
    "Longitude",
    "Latitude",
    "log_n_speakers",
    "n_phonemes",
    "delta",
    "Longitude×Latitude",
    "Longitude×log_n_speakers",
    "Longitude×delta",
    "Latitude×log_n_speakers",
    "Latitude×delta",
    "Phylogenetic",
    "Cov(Fixed, Phylo)",
    "Residual",
]
VARIANCE_COLORS = {
    "Longitude": "#1f77b4",
    "Latitude": "#ff7f0e",
    "log_n_speakers": "#2ca02c",
    "n_phonemes": "#17becf",
    "delta": "#bcbd22",
    "Longitude×Latitude": "#d62728",
    "Longitude×log_n_speakers": "#9467bd",
    "Longitude×delta": "#ff9896",
    "Latitude×log_n_speakers": "#e377c2",
    "Latitude×delta": "#ffbb78",
    "Phylogenetic": "#8c564b",
    "Cov(Fixed, Phylo)": "#c8b8b0",
    "Residual": "#7f7f7f",
}
CSV_TO_DISPLAY = {
    "shapley_longitude_norm_mean": "Longitude",
    "shapley_latitude_norm_mean": "Latitude",
    "shapley_log_n_speakers_norm_mean": "log_n_speakers",
    "shapley_n_phonemes_norm_mean": "n_phonemes",
    "shapley_delta_norm_mean": "delta",
    "shapley_longitude_norm_latitude_norm_mean": "Longitude×Latitude",
    "shapley_longitude_norm_log_n_speakers_norm_mean": "Longitude×log_n_speakers",
    "shapley_longitude_norm_delta_norm_mean": "Longitude×delta",
    "shapley_latitude_norm_log_n_speakers_norm_mean": "Latitude×log_n_speakers",
    "shapley_latitude_norm_delta_norm_mean": "Latitude×delta",
    "prop_phylo_mean": "Phylogenetic",
    "prop_cov_fix_phy_mean": "Cov(Fixed, Phylo)",
    "prop_residual_mean": "Residual",
}
# "delta" renders as "Network signal" everywhere in the figure
PUBLICATION_LABELS = {
    "Longitude": "Longitude",
    "Latitude": "Latitude",
    "log_n_speakers": "Log(speakers)",
    "n_phonemes": "Inventory size",
    "delta": "Network signal (\u03b4)",
    "Longitude×Latitude": "Longitude \u00d7 Latitude",
    "Longitude×log_n_speakers": "Longitude \u00d7 Log(speakers)",
    "Longitude×delta": "Longitude \u00d7 Network signal",
    "Latitude×log_n_speakers": "Latitude \u00d7 Log(speakers)",
    "Latitude×delta": "Latitude \u00d7 Network signal",
}


def load_results(tree):
    suffix = f"{MODEL}_{tree}"
    gp_suffix = f"gp_geo_{tree}"
    meta_file = (
        "speech_metadata_with_inventory.csv"
        if tree == SPEECH_TREE
        else "cognate_metadata_with_inventory.csv"
    )
    result = dict(
        meta=pd.read_csv(DATA_DIR / "phyloregression" / meta_file),
        coef=pd.read_csv(RESULTS_DIR / f"coef_samples_{suffix}.csv"),
        variance=pd.read_csv(RESULTS_DIR / f"variance_samples_{suffix}.csv"),
        summary=pd.read_csv(RESULTS_DIR / f"phylolm_{suffix}.csv"),
        gp_variance=pd.read_csv(RESULTS_DIR / f"variance_samples_{gp_suffix}.csv"),
    )
    return result


# ─── Style ────────────────────────────────────────────────────────────────────
def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "axes.labelsize": 17,
            "axes.labelpad": 10,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cccccc",
            "figure.dpi": 150,
        }
    )


def _bw_box(ax):
    """ggplot2 theme_bw: all four spines, inward ticks, white background."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.9)
    ax.set_facecolor("white")
    ax.tick_params(
        top=True, right=True, which="both", direction="in", length=4.5, width=0.7
    )
    ax.grid(True, color="#dddddd", linewidth=0.5, zorder=0)


def _panel_title(ax, letter):
    """Guaranteed-bold panel label using Arial Bold FontProperties."""
    ax.set_title(letter, loc="left", fontproperties=_PANEL_FP, pad=20)


def _inset_cbar(ax, sm, label, loc="lower right"):
    """Compact horizontal colorbar inset inside ax."""
    rect = (
        [0.55, 0.10, 0.42, 0.06] if loc == "lower right" else [0.03, 0.10, 0.42, 0.06]
    )
    cax = ax.inset_axes(rect)
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=12, labelpad=3)
    cbar.ax.tick_params(labelsize=11, length=3, pad=3)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cax.set_zorder(10)
    return cbar


# =============================================================================
# Shared component order (used by both panels a and b)
# =============================================================================
def _shared_component_order(dat):
    """Return a list of display names present in the data, mains then interactions."""
    eff_df = _extract_coefficient_data(dat["summary"])
    mains = sorted(eff_df[eff_df["_n_vars"] == 1]["coefficient"].unique())
    inters = sorted(eff_df[eff_df["_n_vars"] == 2]["coefficient"].unique())
    return mains + inters


# =============================================================================
# Panel a — Posterior violins + forest plot
# =============================================================================
def make_panel_a(ax, dat, comp_order):
    coef_df = dat["coef"]
    n = len(comp_order)
    display_to_y = {d: i for i, d in enumerate(comp_order)}
    term_colors = {d: VARIANCE_COLORS.get(d, "#888888") for d in comp_order}

    def col_to_display(col):
        return _coef_suffix_to_display(col.replace("_norm", "").replace(".", "_"))

    for col in coef_df.columns:
        if col in ("sample_id", "tree", "Intercept"):
            continue
        disp = col_to_display(col)
        if disp not in display_to_y:
            continue
        samples = coef_df[col].dropna().values
        if len(samples) < 10:
            continue
        y_base = display_to_y[disp]
        kde = stats.gaussian_kde(samples, bw_method="scott")
        xs = np.linspace(np.percentile(samples, 0.5), np.percentile(samples, 99.5), 300)
        dens = kde(xs)
        dens = dens / dens.max() * 0.40
        color = term_colors[disp]
        ax.fill_between(
            xs, y_base - dens, y_base + dens, color=color, alpha=0.50, linewidth=0
        )
        ax.plot(xs, y_base - dens, color=color, alpha=0.55, linewidth=0.4)
        ax.plot(xs, y_base + dens, color=color, alpha=0.55, linewidth=0.4)

    eff_df = _extract_coefficient_data(dat["summary"])
    eff = eff_df.set_index("coefficient")
    for disp, y in display_to_y.items():
        if disp not in eff.index:
            continue
        est, lo, hi = eff.loc[disp, ["estimate", "ci_lower", "ci_upper"]]
        ax.plot(
            [lo, hi],
            [y, y],
            color="#1a1a1a",
            linewidth=1.3,
            solid_capstyle="round",
            zorder=4,
        )
        ax.plot(
            est,
            y,
            "o",
            color="white",
            markersize=6,
            markeredgecolor="#1a1a1a",
            markeredgewidth=1.0,
            zorder=5,
        )

    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([PUBLICATION_LABELS.get(c, c) for c in comp_order])
    ax.set_ylim(-0.7, n - 0.3)
    ax.margins(x=0.07)
    ax.set_xlabel("Standardized effect size")
    _bw_box(ax)
    _panel_title(ax, "a")


# =============================================================================
# Panel b — GP Shapley variance attribution violins
# =============================================================================
_GP_SHAPLEY_COLS = {
    "shapley_longitude_norm": "Longitude",
    "shapley_latitude_norm": "Latitude",
    "shapley_log_n_speakers_norm": "log_n_speakers",
    "shapley_n_phonemes_norm": "n_phonemes",
    "shapley_delta_norm": "delta",
    "shapley_longitude_norm_latitude_norm": "Longitude×Latitude",
    "shapley_longitude_norm_log_n_speakers_norm": "Longitude×log_n_speakers",
    "shapley_longitude_norm_delta_norm": "Longitude×delta",
    "shapley_latitude_norm_log_n_speakers_norm": "Latitude×log_n_speakers",
    "shapley_latitude_norm_delta_norm": "Latitude×delta",
}


def make_panel_b_violin(ax, dat, comp_order):
    gp_var = dat["gp_variance"]
    col_for = {d: c for c, d in _GP_SHAPLEY_COLS.items()}
    n = len(comp_order)
    display_to_y = {d: i for i, d in enumerate(comp_order)}

    for disp in comp_order:
        col = col_for.get(disp)
        if col is None or col not in gp_var.columns:
            continue
        samples = gp_var[col].dropna().values
        if len(samples) < 10:
            continue
        y_base = display_to_y[disp]
        kde = stats.gaussian_kde(samples, bw_method="scott")
        xs = np.linspace(np.percentile(samples, 0.5), np.percentile(samples, 99.5), 300)
        dens = kde(xs)
        dens = dens / dens.max() * 0.40
        color = VARIANCE_COLORS.get(disp, "#888888")
        ax.fill_between(
            xs, y_base - dens, y_base + dens, color=color, alpha=0.50, linewidth=0
        )
        ax.plot(xs, y_base - dens, color=color, alpha=0.55, linewidth=0.4)
        ax.plot(xs, y_base + dens, color=color, alpha=0.55, linewidth=0.4)

        # Median + 95% CI line
        med = np.median(samples)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        ax.plot(
            [lo, hi],
            [y_base, y_base],
            color="#1a1a1a",
            linewidth=1.3,
            solid_capstyle="round",
            zorder=4,
        )
        ax.plot(
            med,
            y_base,
            "o",
            color="white",
            markersize=6,
            markeredgecolor="#1a1a1a",
            markeredgewidth=1.0,
            zorder=5,
        )

    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([])
    ax.set_ylim(-0.7, n - 0.3)
    ax.margins(x=0.07)
    ax.set_xlabel("Contribution to fixed-effect variance\n(Shapley attribution)")
    _bw_box(ax)
    _panel_title(ax, "b")


# =============================================================================
# Panel c — log(rate) vs Network signal; grey CrI band, language labels,
#            dots coloured by GP geographic rate (viridis inset scale)
# =============================================================================
def make_panel_c(ax, dat, gp_result):
    meta = dat["meta"][~dat["meta"]["language"].isin(EXCLUDE_LANGUAGES)].copy()
    coef_df = dat["coef"]
    meta["log_rate"] = np.log(meta["rate_median"])
    meta = meta.set_index("language")

    # Regression line from posterior delta coefficient
    mu_delta = meta["delta"].mean()
    sd_delta = meta["delta"].std(ddof=1)
    b_int = coef_df["Intercept"].values
    b_del = coef_df["delta_norm"].values
    delta_seq = np.linspace(meta["delta"].min() - 0.02, meta["delta"].max() + 0.02, 300)
    delta_n = (delta_seq - mu_delta) / sd_delta
    preds = b_int[:, None] + b_del[:, None] * delta_n[None, :]
    med = np.median(preds, axis=0)
    lo95 = np.percentile(preds, 2.5, axis=0)
    hi95 = np.percentile(preds, 97.5, axis=0)

    ax.fill_between(
        delta_seq, lo95, hi95, color="#aaaaaa", alpha=0.30, linewidth=0, zorder=2
    )
    ax.plot(delta_seq, med, color="#555555", linewidth=1.4, zorder=3)

    # Colour dots by mean fitted log rate from linear model (posterior mean of X @ beta)
    langs = list(meta.index)

    # Build z-scored design matrix for each language
    def _z(col):
        v = meta.loc[langs, col].values
        return (v - meta[col].mean()) / meta[col].std(ddof=1)

    lon_n = _z("longitude")
    lat_n = _z("latitude")
    lns_n = (
        _z("log_n_speakers")
        if "log_n_speakers" in meta.columns
        else (
            np.log(meta.loc[langs, "n_speakers"].values)
            - np.log(meta["n_speakers"]).mean()
        )
        / np.log(meta["n_speakers"]).std(ddof=1)
    )
    del_n = _z("delta")
    pho_n = _z("n_phonemes") if "n_phonemes" in meta.columns else None

    # Build X matrix matching coef_df column order
    X_dict = {
        "Intercept": np.ones(len(langs)),
        "longitude_norm": lon_n,
        "latitude_norm": lat_n,
        "log_n_speakers_norm": lns_n,
        "delta_norm": del_n,
        "longitude_norm.latitude_norm": lon_n * lat_n,
        "longitude_norm.log_n_speakers_norm": lon_n * lns_n,
        "longitude_norm.delta_norm": lon_n * del_n,
        "latitude_norm.log_n_speakers_norm": lat_n * lns_n,
        "latitude_norm.delta_norm": lat_n * del_n,
    }
    if pho_n is not None and "n_phonemes_norm" in coef_df.columns:
        X_dict["n_phonemes_norm"] = pho_n

    # Only use columns present in coef_df
    cols = [
        c for c in coef_df.columns if c in X_dict and c != "sample_id" and c != "tree"
    ]
    X = np.column_stack([X_dict[c] for c in cols])  # (n_langs, p)
    B = coef_df[cols].values  # (n_samples, p)
    fitted = (B.mean(axis=0)[None, :] * X).sum(axis=1)  # posterior-mean fitted log rate

    dot_norm = Normalize(vmin=np.nanmin(fitted), vmax=np.nanmax(fitted))

    ax.scatter(
        meta.loc[langs, "delta"],
        meta.loc[langs, "log_rate"],
        c=fitted,
        cmap=SCMAP,
        norm=dot_norm,
        s=130,
        edgecolors="none",
        zorder=4,
    )

    # Label only non-overlapping points: keep those far from neighbours
    from scipy.spatial import cKDTree

    xs = meta.loc[langs, "delta"].values
    ys = meta.loc[langs, "log_rate"].values

    # Normalise to axis range so x/y distances are comparable
    x_range = xs.max() - xs.min() or 1
    y_range = ys.max() - ys.min() or 1
    coords = np.column_stack([xs / x_range, ys / y_range])
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)  # k=2: nearest neighbour (not self)
    nn_dist = dists[:, 1]

    # Label threshold: top ~40% most isolated points
    threshold = np.percentile(nn_dist, 60)
    label_mask = nn_dist >= threshold

    # Place labels with simple greedy repulsion to avoid overlaps
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    placed_boxes = []  # list of (x0, y0, x1, y1) in data coords

    def _get_bbox_data(txt):
        """Get text bounding box in data coordinates."""
        bb = txt.get_window_extent(renderer=renderer)
        (dx0, dy0), (dx1, dy1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return (dx0, dy0, dx1, dy1)

    def _overlaps(box):
        for pb in placed_boxes:
            if box[0] < pb[2] and box[2] > pb[0] and box[1] < pb[3] and box[3] > pb[1]:
                return True
        return False

    # Try 8 offset directions to find non-overlapping placement
    offsets_pt = [(6, 6), (-6, 6), (6, -10), (-6, -10),
                  (12, 0), (-12, 0), (0, 10), (0, -14)]

    for i, lang in enumerate(langs):
        if not label_mask[i]:
            continue
        best_txt = None
        for dx, dy in offsets_pt:
            txt = ax.annotate(
                lang, (xs[i], ys[i]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=11, color="#333333", clip_on=True,
            )
            box = _get_bbox_data(txt)
            if not _overlaps(box):
                placed_boxes.append(box)
                best_txt = txt
                break
            txt.remove()
        if best_txt is None:
            # All positions overlap; place at first offset anyway
            txt = ax.annotate(
                lang, (xs[i], ys[i]),
                xytext=offsets_pt[0], textcoords="offset points",
                fontsize=11, color="#333333", clip_on=True,
            )
            placed_boxes.append(_get_bbox_data(txt))

    # Inset colorbar
    sm_dot = ScalarMappable(norm=dot_norm, cmap=SCMAP)
    sm_dot.set_array([])
    _inset_cbar(ax, sm_dot, "Fitted log rate\n(linear model)", loc="lower right")

    # Tight axis limits with small padding
    x_vals = meta.loc[langs, "delta"].values
    y_vals = meta.loc[langs, "log_rate"].values
    x_span = x_vals.max() - x_vals.min()
    y_span = y_vals.max() - y_vals.min()
    ax.set_xlim(x_vals.min() - x_span * 0.08, x_vals.max() + x_span * 0.18)
    ax.set_ylim(y_vals.min() - y_span * 0.08, y_vals.max() + y_span * 0.08)

    ax.set_xlabel("Network signal (\u03b4)")
    ax.set_ylabel("Log Median Bayesian Phylogenetic Rate")
    _bw_box(ax)
    _panel_title(ax, "c")


# =============================================================================
# Panel d — GP regression surface map (replaces old spline surface)
# =============================================================================
_GP_MAP_CONSTANTS = dict(
    GRID_N_LON_TRAIN=150,
    GRID_N_LAT_TRAIN=100,
    GRID_N_LON_PRED=600,
    GRID_N_LAT_PRED=400,
    FIXED_NOISE=1e-2,
    PAD_DEG=15,
)


def _fit_gp_surface(meta, geojson_path):
    """Fit a gpflow GP to polygon-gridded training data and return prediction arrays."""
    import gpflow
    from shapely.strtree import STRtree
    from sklearn.preprocessing import StandardScaler

    c = _GP_MAP_CONSTANTS

    gdf_language = load_language_polygons(str(geojson_path))

    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    if "NAME" in world.columns and "name" not in world.columns:
        world = world.rename(columns={"NAME": "name"})
    ire = world.loc[world["name"] == "Ireland"].total_bounds
    bgd = world.loc[world["name"] == "Bangladesh"].total_bounds
    pad = c["PAD_DEG"]
    roi_minx = min(ire[0], bgd[0]) - pad
    roi_maxx = max(ire[2], bgd[2]) + pad
    roi_miny = min(ire[1], bgd[1]) - pad
    roi_maxy = max(ire[3], bgd[3]) + pad
    roi_box = shapely_box(roi_minx, roi_miny, roi_maxx, roi_maxy)

    # Filter metadata
    meta_idx = meta.set_index("language") if "language" in meta.columns else meta
    meta_idx = meta_idx.loc[~meta_idx.index.isin(EXCLUDE_LANGUAGES)].copy()

    # Clip polygons to ROI
    gdf_clipped = gpd.clip(gdf_language, roi_box)
    gdf_clipped = gdf_clipped[gdf_clipped["name"].isin(meta_idx.index)].copy()
    gdf_clipped["rate"] = gdf_clipped["name"].map(meta_idx["rate_median"])
    gdf_clipped = gdf_clipped.dropna(subset=["rate"]).reset_index(drop=True)

    # Build training grid
    from collections import Counter, defaultdict

    lon_grid = np.linspace(roi_box.bounds[0], roi_box.bounds[2], c["GRID_N_LON_TRAIN"])
    lat_grid = np.linspace(roi_box.bounds[1], roi_box.bounds[3], c["GRID_N_LAT_TRAIN"])
    LON_tr, LAT_tr = np.meshgrid(lon_grid, lat_grid)
    grid_points = np.column_stack([LON_tr.ravel(), LAT_tr.ravel()])

    geometries = gdf_clipped.geometry.values
    tree = STRtree(geometries)
    point_rates = defaultdict(list)
    point_labels = defaultdict(list)
    n_min = 20
    for lon, lat in grid_points:
        pt = Point(lon, lat)
        for idx in tree.query(pt):
            if geometries[idx].contains(pt):
                point_rates[(lon, lat)].append(gdf_clipped.iloc[idx]["rate"])
                point_labels[(lon, lat)].append(gdf_clipped.iloc[idx]["name"])

    # Adaptive upsampling for under-represented languages
    lang_count = Counter(l for labels in point_labels.values() for l in labels)
    for _, row in gdf_clipped.iterrows():
        lang = row["name"]
        if lang_count.get(lang, 0) >= n_min:
            continue
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        bbox_area = max((maxx - minx) * (maxy - miny), 1e-6)
        fill_rate = max(geom.area / bbox_area, 0.01)
        n_needed = n_min - lang_count.get(lang, 0)
        target = max(int(np.ceil(n_needed / fill_rate * 3)), 100)
        ratio = (maxx - minx) / max(maxy - miny, 1e-6)
        n_lo = max(int(np.ceil(np.sqrt(target * ratio))), 5)
        n_la = max(int(np.ceil(np.sqrt(target / max(ratio, 1e-6)))), 5)
        for lo in np.linspace(minx, maxx, n_lo):
            for la in np.linspace(miny, maxy, n_la):
                key = (lo, la)
                if key in point_rates:
                    continue
                pt = Point(lo, la)
                for other_idx in tree.query(pt):
                    if geometries[other_idx].contains(pt):
                        point_rates[key].append(gdf_clipped.iloc[other_idx]["rate"])
                        point_labels[key].append(gdf_clipped.iloc[other_idx]["name"])
                if lang in point_labels[key]:
                    lang_count[lang] = lang_count.get(lang, 0) + 1

    if len(point_rates) == 0:
        return None

    X_train = np.array(list(point_rates.keys()))
    y_train = np.array([np.mean(v) for v in point_rates.values()])

    # Fit GP
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    kernel = gpflow.kernels.Matern32(lengthscales=1.0, variance=1.0)
    model = gpflow.models.GPR(
        data=(X_scaled, y_scaled.reshape(-1, 1)),
        kernel=kernel,
        mean_function=None,
    )
    model.likelihood.variance.assign(c["FIXED_NOISE"])
    gpflow.set_trainable(model.likelihood.variance, False)
    gpflow.optimizers.Scipy().minimize(
        model.training_loss,
        variables=model.trainable_variables,
        options=dict(maxiter=500),
    )

    # Prediction grid
    lon_pred = np.linspace(roi_minx, roi_maxx, c["GRID_N_LON_PRED"])
    lat_pred = np.linspace(roi_miny, roi_maxy, c["GRID_N_LAT_PRED"])
    LON, LAT = np.meshgrid(lon_pred, lat_pred)
    grid_pred = np.column_stack([LON.ravel(), LAT.ravel()])

    language_union = gdf_clipped.unary_union
    language_union_prep = prep(language_union)
    in_mask = np.array(
        [language_union_prep.contains(Point(p[0], p[1])) for p in grid_pred]
    )

    grid_inside = grid_pred[in_mask]
    # Predict in batches
    mean_all = []
    for i in range(0, len(grid_inside), 5000):
        batch = grid_inside[i : i + 5000]
        Xg = scaler_X.transform(batch)
        m, _ = model.predict_f(Xg)
        mean_all.append(m.numpy().ravel())
    mean_inside = np.concatenate(mean_all)

    Z = np.full(len(grid_pred), np.nan, dtype=float)
    Z[in_mask] = scaler_y.inverse_transform(mean_inside.reshape(-1, 1)).ravel()
    Z = Z.reshape(LON.shape)

    # Obs predictions (inverse-transform back to raw rate space)
    X_obs = meta_idx[["longitude", "latitude"]].to_numpy()
    m_obs, _ = model.predict_f(scaler_X.transform(X_obs))
    meta_idx["rate_gp"] = scaler_y.inverse_transform(
        m_obs.numpy().reshape(-1, 1)
    ).ravel()

    # Land for coastlines
    try:
        import geodatasets

        _land = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    except Exception:
        _land = gpd.read_file(
            "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
        )
    land = gpd.clip(_land, roi_box)

    return dict(
        LON=LON,
        LAT=LAT,
        Z=Z,
        meta=meta_idx,
        land=land,
        roi=(roi_minx, roi_maxx, roi_miny, roi_maxy),
    )


def make_panel_d(ax, gp_result):
    """GP regression surface (viridis); dots = observed rate (same scale)."""
    if gp_result is None:
        ax.text(
            0.5,
            0.5,
            "GP surface unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#888",
        )
        _panel_title(ax, "d")
        return

    LON, LAT, Z = gp_result["LON"], gp_result["LAT"], gp_result["Z"]
    meta = gp_result["meta"]
    land = gp_result["land"]
    roi_minx, roi_maxx, roi_miny, roi_maxy = gp_result["roi"]

    # Shared norm from surface + GP predictions at obs points
    vals = []
    if np.isfinite(Z).any():
        vals.append(Z[np.isfinite(Z)].ravel())
    vals.append(meta["rate_gp"].to_numpy())
    vals = np.concatenate([v[np.isfinite(v)] for v in vals])
    shared_norm = Normalize(
        vmin=np.quantile(vals, 0.02) if vals.size > 0 else 0,
        vmax=np.quantile(vals, 0.98) if vals.size > 0 else 1,
    )

    ax.set_facecolor("#d5e9ff")
    land.plot(ax=ax, color="#f0f0f0", edgecolor="#999999", linewidth=0.45, zorder=1)

    if np.isfinite(Z).any():
        ax.pcolormesh(
            LON, LAT, Z, shading="auto", cmap=MAP_CMAP, norm=shared_norm, zorder=2
        )

    # Dots: GP prediction at obs points, same cmap/norm as surface
    ax.scatter(
        meta["longitude"].values,
        meta["latitude"].values,
        c=meta["rate_gp"].values,
        cmap=MAP_CMAP,
        norm=shared_norm,
        s=90,
        marker="o",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )

    sm = ScalarMappable(norm=shared_norm, cmap=MAP_CMAP)
    sm.set_array([])
    _inset_cbar(ax, sm, "Median\n Bayesian Phylogenetic Rate", loc="lower left")

    ax.set_xlim(roi_minx, roi_maxx)
    ax.set_ylim(roi_miny, roi_maxy)
    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.grid(True, color="white", linewidth=0.35, alpha=0.6, zorder=3)
    _panel_title(ax, "d")


# =============================================================================
# Assemble and save
# =============================================================================
def make_figure(dat, out_stem, tree, geojson_path=None):
    apply_style()

    comp_order = _shared_component_order(dat)

    if geojson_path is None:
        geojson_path = GEOJSON_PATH

    # Fit GP surface for panels c/d
    print("  Fitting GP surface for map...")
    gp_result = _fit_gp_surface(dat["meta"], geojson_path)

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        hspace=0.26,
        wspace=0.38,
        left=0.16,
        right=0.96,
        top=0.95,
        bottom=0.08,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    print("  Panel a...")
    make_panel_a(ax_a, dat, comp_order)
    print("  Panel b...")
    make_panel_b_violin(ax_b, dat, comp_order)
    print("  Panel c...")
    make_panel_c(ax_c, dat, gp_result)
    print("  Panel d...")
    make_panel_d(ax_d, gp_result)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg", "png"):
        path = OUT_DIR / f"{out_stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("── Figure 3 geo (main) — speech tree ───────────────────────────────")
    dat_speech = load_results(SPEECH_TREE)
    make_figure(dat_speech, "figure3_geo_speech", SPEECH_TREE)

    print("── Figure 3 geo (supp) — cognate tree ──────────────────────────────")
    dat_cognate = load_results(COGNATE_TREE)
    make_figure(dat_cognate, "figure3_geo_cognate_supp", COGNATE_TREE)

    print("Done.")
