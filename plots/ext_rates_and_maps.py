#!/usr/bin/env python3
"""Final plots for speech phylogenetic analysis.

Sections:
  1. Bayesian effects forest plot
  2. Variance decomposition stacked bars
  3. Rate vs longitude scatter
  4. Root age comparison (dendropy, slow)
  5. Rate-over-time: standardized & % change (dendropy, slow)
  8. Continuous GP maps (tensorflow/gpflow)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tasks.phylo.constants import (
    COGNATE_TO_SPEECH,
    EXCLUDE_LANGUAGES,
    GEOJSON_EXPANSION,
)

# only need CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ─── Configuration ───────────────────────────────────────────────────────────

RESULTS_DIR = "data/phyloregression"
OUTPUT_DIR = "data/phyloregression/figures"
DATA_DIR = "data"
REGRESSION_DIR = "data/phyloregression"

# Metadata CSV paths keyed by variant then tree name
TREE_META_PATHS = {
    "with_inventory": {
        "input_v12_combined_resampled": f"{REGRESSION_DIR}/speech_metadata_with_inventory.csv",
        "heggarty2024_raw": f"{REGRESSION_DIR}/cognate_metadata_with_inventory.csv",
    },
    "no_inventory": {
        "input_v12_combined_resampled": f"{REGRESSION_DIR}/speech_metadata.csv",
        "heggarty2024_raw": f"{REGRESSION_DIR}/cognate_metadata.csv",
    },
}

# Mapping from tree identifiers in CSVs to display names
TREE_DISPLAY = {
    "heggarty2024_raw": "Cognates",
    "input_v12_combined_resampled": "Speech",
}

# Canonical family lookup (used by delta and continuous-map plots)
FAMILY_LOOKUP = {
    "Asturian": "latinofaliscan",
    "Icelandic": "germanic",
    "Luxembourgish": "germanic",
    "Occitan": "latinofaliscan",
    "Sindhi": "indoaryan",
    "Tajik": "iranian",
    "Assamese": "indoaryan",
    "Afrikaans": "germanic",
    "Bosnian": "slavic",
    "Belarusian": "slavic",
    "Bulgarian": "slavic",
    "Bengali": "indoaryan",
    "Catalan": "latinofaliscan",
    "Czech": "slavic",
    "Danish": "germanic",
    "German": "germanic",
    "Greek": "greek",
    "English": "germanic",
    "Galician": "latinofaliscan",
    "Gujarati": "indoaryan",
    "Spanish": "latinofaliscan",
    "PersianTehran": "iranian",
    "French": "latinofaliscan",
    "Hindi": "indoaryan",
    "Serbian": "slavic",
    "Croatian": "slavic",
    "ArmenianEastern": "armenian",
    "Armenian": "armenian",
    "Italian": "latinofaliscan",
    "Lithuanian": "baltic",
    "Latvian": "baltic",
    "Macedonian": "slavic",
    "Marathi": "indoaryan",
    "NorwegianBokmal": "germanic",
    "Nepali": "indoaryan",
    "Oriya": "indoaryan",
    "Dutch": "germanic",
    "Punjabi": "indoaryan",
    "Polish": "slavic",
    "Pashto": "iranian",
    "Portuguese": "latinofaliscan",
    "Romanian": "latinofaliscan",
    "Russian": "slavic",
    "Slovak": "slavic",
    "Slovenian": "slavic",
    "Slovene": "slavic",
    "Swedish": "germanic",
    "Ukrainian": "slavic",
    "Urdu": "indoaryan",
    "KurdishCJafi": "iranian",
    "Sorani-Kurdish": "iranian",
    "Persian": "iranian",
    "SerboCroatian": "slavic",
    "NorwegianBokmal": "germanic",
    "Norwegian": "germanic",
    "GaelicIrish": "celtic",
    "WelshNorth": "celtic",
    "Welsh": "celtic",
    "Irish": "celtic",
    "Kabuverdianu": "latinofaliscan",
}

FAMILY_MARKERS = {
    "germanic": "s",
    "slavic": "^",
    "baltic": "D",
    "indoaryan": "o",
    "latinofaliscan": "v",
    "iranian": "P",
    "armenian": "X",
    "greek": "h",
    "celtic": "*",
    "other": "o",
}

FAMILY_COLORS = {
    "latinofaliscan": "#e5efd7ff",
    "indoaryan": "#dacbd1ff",
    "germanic": "#d1d8ccff",
    "slavic": "#f1ecdaff",
    "baltic": "#e9e4ccff",
    "armenian": "#cfcde1ff",
    "iranian": "#e8d6dcff",
    "greek": "#d8e3f1ff",
    "celtic": "#d4e8d4ff",
}


# ─── Shared Utilities ────────────────────────────────────────────────────────


def tidy_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Clean metadata CSV: split 'lo,hi' columns, drop extras, set language index."""
    meta = meta.rename(columns={"level_0": "language_true"})
    for col in list(meta.columns):
        if (
            pd.api.types.is_string_dtype(meta[col])
            and meta[col].astype(str).str.contains(",", na=False).any()
        ):
            new_cols = meta[col].astype(str).str.split(",", expand=True)
            meta[f"{col}_lo"] = pd.to_numeric(new_cols[0], errors="coerce")
            meta[f"{col}_hi"] = pd.to_numeric(new_cols[1], errors="coerce")
            meta.drop(columns=[col], inplace=True)
    meta.drop(columns=meta.columns[1:7], errors="ignore", inplace=True)
    meta.rename(columns={"language_true": "language"}, inplace=True)
    meta["language"] = meta["language"].astype(str).str.strip()
    return meta.set_index("language")


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


def load_language_polygons(filepath):
    """Load and preprocess language polygon GeoJSON.

    Applies GEOJSON_EXPANSION to rename and/or duplicate polygons so each row
    matches the language name used in the metadata CSVs.  Entries with multiple
    target names (e.g. "Slovene" → ["Slovene", "Slovenian"]) are duplicated so
    the same polygon matches both Speech and Cognate metadata files.
    """
    import geopandas as gpd

    gdf = gpd.read_file(filepath)

    # First pass: merge multi-row geometries with the same geojson name
    merged_rows = []
    for name in gdf["name"].unique():
        poly = gdf[gdf["name"] == name].geometry.unary_union
        merged_rows.append({"name": name, "geometry": poly})
    gdf = gpd.GeoDataFrame(merged_rows, crs=gdf.crs)

    # Expand entries according to GEOJSON_EXPANSION
    expand_mask = gdf["name"].isin(GEOJSON_EXPANSION)
    gdf_to_expand = gdf[expand_mask]
    gdf_rest = gdf[~expand_mask]
    expanded = []
    for _, row in gdf_to_expand.iterrows():
        for new_name in GEOJSON_EXPANSION[row["name"]]:
            new_row = row.copy()
            new_row["name"] = new_name
            expanded.append(new_row)
    if expanded:
        gdf_exp = gpd.GeoDataFrame(expanded, crs=gdf.crs)
        gdf = pd.concat([gdf_rest, gdf_exp], ignore_index=True)

    # Second pass: re-merge any rows that ended up with the same name
    # (e.g. "Netherlandic" → "Dutch" gets merged with the original "Dutch")
    merged_rows = []
    for name in gdf["name"].unique():
        poly = gdf[gdf["name"] == name].geometry.unary_union
        merged_rows.append({"name": name, "geometry": poly})
    gdf = gpd.GeoDataFrame(merged_rows, crs=gdf.crs)

    return gdf


def _get_world_gdf():
    """Load world country boundaries (110m Natural Earth)."""
    import geopandas as gpd

    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    world = gpd.read_file(url)
    # Normalise to lowercase 'name' used throughout the code
    if "NAME" in world.columns and "name" not in world.columns:
        world = world.rename(columns={"NAME": "name"})
    return world


def _get_land_gdf():
    """Load Natural Earth land polygons (coastlines only, no country borders)."""
    import geopandas as gpd

    url = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
    return gpd.read_file(url)


# ═════════════════════════════════════════════════════════════════════════════
# Rate vs Longitude Scatter
# ═════════════════════════════════════════════════════════════════════════════


def plot_rate_vs_longitude(output_dir=OUTPUT_DIR, meta_paths=None):
    if meta_paths is None:
        meta_paths = TREE_META_PATHS["with_inventory"]
    for tree_name in ["heggarty2024_raw", "input_v12_combined_resampled"]:
        meta_path = meta_paths.get(tree_name)
        if meta_path is None or not os.path.exists(meta_path):
            print(f"  Skipping {tree_name}: metadata not found")
            continue

        meta = pd.read_csv(meta_path).set_index("language")

        y = meta["rate_median"].to_numpy().reshape(-1, 1)

        fig = plt.figure(figsize=(10, 6))
        sc = plt.scatter(
            meta["longitude"], y, c=meta["n_speakers"], cmap="viridis", s=50
        )

        for lang, row in meta.iterrows():
            plt.annotate(
                lang,
                (row["longitude"], row["rate_median"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color="black",
                alpha=0.5,
            )

        plt.yscale("log")
        plt.colorbar(sc, label="Log Number of Speakers", pad=0.12)
        plt.xlabel("Longitude", fontsize=22)
        plt.ylabel("Log Rate", fontsize=22)
        plt.tick_params(axis="x", labelsize=22)
        plt.tick_params(axis="y", labelsize=22)
        plt.grid(True)

        os.makedirs(output_dir, exist_ok=True)
        out_path = f"{output_dir}/rate_vs_longitude_{tree_name}.pdf"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Root Age Comparison
# ═════════════════════════════════════════════════════════════════════════════


def _read_log_tree_height(log_path, burnin=0):
    """Read TreeHeight.t:tree column from a BEAST log file, skipping burnin rows."""
    df = pd.read_csv(log_path, sep="\t", comment="#")
    col = "TreeHeight.t:tree"
    heights = df[col].values
    if burnin > 0:
        heights = heights[burnin:]
    return heights


def plot_root_age_comparison(
    speech_log=f"{DATA_DIR}/trees/beast/speech/0.01_brsupport/input_combined_resampled.log",
    cognate_log=f"{DATA_DIR}/trees/beast/iecor/raw.log",
    speech_prior_log=f"{DATA_DIR}/trees/beast/speech/0.01_brsupport/prior_1.log",
    cognate_prior_log=f"{DATA_DIR}/trees/beast/iecor/prior/raw.log",
    burnin=1000,
    output_dir=OUTPUT_DIR,
):
    from scipy import stats

    speech_post = _read_log_tree_height(speech_log, burnin=0)
    cognate_post = _read_log_tree_height(cognate_log, burnin=burnin)
    speech_prior = _read_log_tree_height(speech_prior_log, burnin=0)
    cognate_prior = _read_log_tree_height(cognate_prior_log, burnin=0)

    for label, arr in [
        ("Speech posterior", speech_post),
        ("Cognate posterior", cognate_post),
        ("Speech prior", speech_prior),
        ("Cognate prior", cognate_prior),
    ]:
        print(
            f"  {label}: mean={arr.mean():.3f} kya, "
            f"95% CI=[{np.percentile(arr, 2.5):.3f}, {np.percentile(arr, 97.5):.3f}]"
        )

    # Match cognate rates plot x-range (0–10 ka) and width so 1 ka has
    # the same physical length across figures.
    all_ages = np.concatenate([speech_post, cognate_post, speech_prior, cognate_prior])
    xmax = np.ceil(all_ages.max() + 0.5)  # round up to nearest ka
    fig, ax = plt.subplots(figsize=(7.2, 4))

    def plot_density(data, color, label, filled=True):
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min() - 0.2, data.max() + 0.2, 500)
        density = kde(x_range)
        if filled:
            ax.fill_between(x_range, density, alpha=0.45, color=color)
            ax.plot(x_range, density, color=color, linewidth=1.8, label=label)
        else:
            ax.plot(
                x_range,
                density,
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
                label=label,
            )

    plot_density(speech_post, "#414487", "Speech posterior", filled=True)
    plot_density(cognate_post, "#7ad151", "Cognates posterior", filled=True)
    plot_density(speech_prior, "#414487", "Speech prior", filled=False)
    plot_density(cognate_prior, "#7ad151", "Cognates prior", filled=False)

    ax.set_xlim(xmax, 0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_xlabel("Thousand years before present", fontsize=22)
    ax.set_ylabel("Density", fontsize=22)
    ax.legend(loc="upper left", fontsize=16)
    ax.set_ylim(0, None)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/root_age_comparison.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # save as svg
    out_path_svg = f"{output_dir}/root_age_comparison.svg"
    plt.savefig(out_path_svg, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path_svg}")


# ═════════════════════════════════════════════════════════════════════════════
# Rate Over Time (standardized & % change)
# ═════════════════════════════════════════════════════════════════════════════


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


def _global_tmax(*groups):
    tmax = 0.0
    for group in groups:
        for tr in group:
            tr.calc_node_ages()
            tmax = max(tmax, float(tr.seed_node.age))
    return tmax


def _alpha_cmap(base_color, max_alpha=0.6):
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

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


def plot_rate_over_time_normalized(
    trees_cognates, trees_speech, burnin=1000, ntimes=200, output_dir=OUTPUT_DIR
):
    """Standardized rate over time with alpha-density bands."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    cog = trees_cognates[burnin:]
    # spe = trees_speech[burnin:]

    # no burn in for speech
    spe = trees_speech

    tmax = _global_tmax(cog, spe)
    t_grid = np.linspace(0.0, tmax, ntimes)

    rates_A = _rates_over_time_slices(cog, t_grid)
    rates_B = _rates_over_time_slices(spe, t_grid)

    norm_A = (rates_A - np.nanmean(rates_A, axis=1, keepdims=True)) / np.nanstd(
        rates_A, axis=1, keepdims=True, ddof=1
    )
    norm_B = (rates_B - np.nanmean(rates_B, axis=1, keepdims=True)) / np.nanstd(
        rates_B, axis=1, keepdims=True, ddof=1
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    color_A, color_B = "#7ad151", "#414487"

    counts_A = np.sum(np.isfinite(norm_A), axis=0) / norm_A.shape[0] * 100
    counts_B = np.sum(np.isfinite(norm_B), axis=0) / norm_B.shape[0] * 100
    norm_counts = Normalize(vmin=0, vmax=100)
    ylow, yhigh = -3, 3
    ax.set_ylim(ylow, yhigh)

    _add_alpha_band(
        ax,
        t_grid,
        norm_A,
        counts_A,
        color_A,
        "Cognates",
        norm_counts,
        ylow,
        yhigh,
        zorder=1,
    )
    _add_alpha_band(
        ax,
        t_grid,
        norm_B,
        counts_B,
        color_B,
        "Speech",
        norm_counts,
        ylow,
        yhigh,
        zorder=3,
    )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlim(tmax, 0.0)
    ax.set_ylim(ylow, yhigh)
    ax.set_ylabel("Standardized Rate")
    ax.set_xlabel("ka BP")
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.legend()

    sm = cm.ScalarMappable(norm=norm_counts, cmap=_alpha_cmap(color_A))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("% MCMC samples (trees) at time t")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/rate_over_time_normalized.pdf"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")

    return t_grid, norm_A, norm_B


def plot_pct_change_over_time(
    trees_cognates, trees_speech, burnin=1000, ntimes=200, output_dir=OUTPUT_DIR
):
    """% change in rate relative to present, with alpha-density bands."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    cog = trees_cognates[burnin:]

    # spe = trees_speech[burnin:]
    # no burn in for speech
    spe = trees_speech

    tmax = _global_tmax(cog, spe)
    t_grid = np.linspace(0.05, tmax, ntimes)

    raw_cog = _rates_over_time_slices(cog, t_grid)
    raw_speech = _rates_over_time_slices(spe, t_grid)

    def pct_change(rates):
        ref_rate = rates[:, 0:1]
        return (rates - ref_rate) / ref_rate * 100

    pct_cog = pct_change(raw_cog)
    pct_speech = pct_change(raw_speech)

    fig, ax = plt.subplots(figsize=(8, 4))
    color_cog, color_speech = "#7ad151", "#414487"

    counts_cog = np.sum(np.isfinite(pct_cog), axis=0) / pct_cog.shape[0] * 100
    counts_speech = np.sum(np.isfinite(pct_speech), axis=0) / pct_speech.shape[0] * 100
    norm_counts = Normalize(vmin=0, vmax=100)
    ylow, yhigh = -100, 100

    _add_alpha_band(
        ax,
        t_grid,
        pct_cog,
        counts_cog,
        color_cog,
        "Cognates",
        norm_counts,
        ylow,
        yhigh,
        zorder=1,
    )
    _add_alpha_band(
        ax,
        t_grid,
        pct_speech,
        counts_speech,
        color_speech,
        "Speech",
        norm_counts,
        ylow,
        yhigh,
        zorder=3,
    )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlim(t_grid.max(), t_grid.min())
    ax.set_ylim(ylow, yhigh)
    ax.set_ylabel("% change in rate (relative to present)")
    ax.set_xlabel("ka BP")
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    sm = cm.ScalarMappable(norm=norm_counts, cmap=_alpha_cmap(color_cog))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("% MCMC samples at time t")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/pct_change_over_time_alpha.pdf"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Continuous GP Maps (grid-trained on polygons)
# ═════════════════════════════════════════════════════════════════════════════


def plot_continuous_map_grid(
    results_dir=RESULTS_DIR, output_dir=OUTPUT_DIR, geojson_path=None, meta_paths=None
):
    """GP-interpolated rate map trained on polygon-interior grid points."""
    if meta_paths is None:
        meta_paths = TREE_META_PATHS["with_inventory"]
    import geopandas as gpd
    import gpflow
    from shapely.geometry import Point, box
    from shapely.prepared import prep
    from shapely.strtree import STRtree
    from sklearn.preprocessing import StandardScaler

    if geojson_path is None:
        geojson_path = f"{DATA_DIR}/metadata/fleurs-r/language_polygons.geojson"

    CMAP = "magma"
    GRID_N_LON_TRAIN, GRID_N_LAT_TRAIN = 150, 100
    GRID_N_LON_PRED, GRID_N_LAT_PRED = 600, 400
    FIXED_NOISE = 1e-2
    PAD_DEG = 15

    tree_names = ["heggarty2024_raw", "input_v12_combined_resampled"]

    world = _get_world_gdf()
    ire = world.loc[world["name"] == "Ireland"].total_bounds
    bgd = world.loc[world["name"] == "Bangladesh"].total_bounds
    ROI_MINX = min(ire[0], bgd[0]) - PAD_DEG
    ROI_MAXX = max(ire[2], bgd[2]) + PAD_DEG
    ROI_MINY = min(ire[1], bgd[1]) - PAD_DEG
    ROI_MAXY = max(ire[3], bgd[3]) + PAD_DEG
    ROI_BOX = box(ROI_MINX, ROI_MINY, ROI_MAXX, ROI_MAXY)

    gdf_language = load_language_polygons(geojson_path)

    def create_polygon_grid_training_data(
        gdf_lang, meta, roi_box, n_lon, n_lat, n_min_per_lang=20
    ):
        gdf_clipped = gpd.clip(gdf_lang, roi_box)
        gdf_clipped = gdf_clipped[gdf_clipped["name"].isin(meta.index)].copy()
        gdf_clipped["rate"] = gdf_clipped["name"].map(meta["rate_median"])
        gdf_clipped = gdf_clipped.dropna(subset=["rate"]).reset_index(drop=True)

        lon_grid = np.linspace(roi_box.bounds[0], roi_box.bounds[2], n_lon)
        lat_grid = np.linspace(roi_box.bounds[1], roi_box.bounds[3], n_lat)
        LON, LAT = np.meshgrid(lon_grid, lat_grid)
        grid_points = np.column_stack([LON.ravel(), LAT.ravel()])

        geometries = gdf_clipped.geometry.values
        tree = STRtree(geometries)

        from collections import Counter, defaultdict

        point_rates = defaultdict(list)
        point_labels = defaultdict(list)
        for lon, lat in grid_points:
            pt = Point(lon, lat)
            for idx in tree.query(pt):
                if geometries[idx].contains(pt):
                    point_rates[(lon, lat)].append(gdf_clipped.iloc[idx]["rate"])
                    point_labels[(lon, lat)].append(gdf_clipped.iloc[idx]["name"])

        # Adaptive upsampling: languages covered by fewer than n_min_per_lang
        # training points get a dense local grid over their polygon bounding box.
        # Candidate count is scaled by the inverse fill rate so that low-fill
        # polygons (e.g. fragmented or narrow) still hit the target.
        lang_count = Counter(l for labels in point_labels.values() for l in labels)
        for _, row in gdf_clipped.iterrows():
            lang = row["name"]
            if lang_count.get(lang, 0) >= n_min_per_lang:
                continue
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds
            bbox_area = max((maxx - minx) * (maxy - miny), 1e-6)
            fill_rate = max(geom.area / bbox_area, 0.01)  # cap at 1 % floor
            n_needed = n_min_per_lang - lang_count.get(lang, 0)
            # candidates = needed / fill_rate * safety factor of 3
            target = max(int(np.ceil(n_needed / fill_rate * 3)), 100)
            ratio = (maxx - minx) / max(maxy - miny, 1e-6)
            n_lo = max(int(np.ceil(np.sqrt(target * ratio))), 5)
            n_la = max(int(np.ceil(np.sqrt(target / max(ratio, 1e-6)))), 5)
            for lo in np.linspace(minx, maxx, n_lo):
                for la in np.linspace(miny, maxy, n_la):
                    key = (lo, la)
                    if key in point_rates:
                        continue  # global grid already sampled this location
                    pt = Point(lo, la)
                    # Check all polygons (not just the target) so overlapping
                    # language areas are averaged correctly.
                    for other_idx in tree.query(pt):
                        if geometries[other_idx].contains(pt):
                            other_lang = gdf_clipped.iloc[other_idx]["name"]
                            point_rates[key].append(gdf_clipped.iloc[other_idx]["rate"])
                            point_labels[key].append(other_lang)
                    if lang in point_labels[key]:
                        lang_count[lang] = lang_count.get(lang, 0) + 1

        X_train = np.array(list(point_rates.keys()))
        y_train = np.array([np.mean(v) for v in point_rates.values()])
        lang_labels = [
            labels[0] if len(labels) == 1 else f"overlap({','.join(labels)})"
            for labels in point_labels.values()
        ]
        return X_train, y_train, lang_labels, gdf_clipped

    def predict_in_batches(model, scaler_X, grid_points, batch_size=5000):
        mean_all, var_all = [], []
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points[i : i + batch_size]
            Xg = scaler_X.transform(batch)
            m, v = model.predict_f(Xg)
            mean_all.append(m.numpy().ravel())
            var_all.append(v.numpy().ravel())
        return np.concatenate(mean_all), np.concatenate(var_all)

    for tree_name in tree_names:
        print(f"  [{tree_name}] Processing...")
        meta_path = meta_paths.get(tree_name)
        if meta_path is None or not os.path.exists(meta_path):
            print(f"    metadata not found; skipping.")
            continue

        meta = pd.read_csv(meta_path).set_index("language")
        meta = meta.loc[~meta.index.isin(EXCLUDE_LANGUAGES)].copy()
        if meta.empty:
            continue

        # Training data from polygon grid
        print("    Creating polygon-based training grid...")
        X_train, y_train, _, gdf_clipped = create_polygon_grid_training_data(
            gdf_language, meta, ROI_BOX, GRID_N_LON_TRAIN, GRID_N_LAT_TRAIN
        )
        if len(X_train) == 0:
            print("    No training points generated; skipping.")
            continue

        # Fit GP
        print(f"    Fitting GP with {X_train.shape[0]} training points...")
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
        model.likelihood.variance.assign(FIXED_NOISE)
        gpflow.set_trainable(model.likelihood.variance, False)
        gpflow.optimizers.Scipy().minimize(
            model.training_loss,
            variables=model.trainable_variables,
            options=dict(maxiter=500),
        )

        # Prediction grid
        lon_pred = np.linspace(ROI_MINX, ROI_MAXX, GRID_N_LON_PRED)
        lat_pred = np.linspace(ROI_MINY, ROI_MAXY, GRID_N_LAT_PRED)
        LON, LAT = np.meshgrid(lon_pred, lat_pred)
        grid_pred = np.column_stack([LON.ravel(), LAT.ravel()])

        language_union = gdf_clipped.unary_union
        language_union_prep = prep(language_union)

        print("    Creating prediction mask...")
        in_language_mask = np.array(
            [language_union_prep.contains(Point(p[0], p[1])) for p in grid_pred]
        )

        grid_pred_inside = grid_pred[in_language_mask]
        print(f"    Predicting on {len(grid_pred_inside)} points...")
        mean_inside, _ = predict_in_batches(model, scaler_X, grid_pred_inside)

        Z_eff = np.full(len(grid_pred), np.nan, dtype=float)
        Z_eff[in_language_mask] = scaler_y.inverse_transform(
            mean_inside.reshape(-1, 1)
        ).ravel()
        Z_eff = Z_eff.reshape(LON.shape)

        # Observation-point predictions (inverse-transform back to raw rate space)
        X_obs = meta[["longitude", "latitude"]].to_numpy()
        m_obs, _ = model.predict_f(scaler_X.transform(X_obs))
        meta["rate_mean"] = scaler_y.inverse_transform(
            m_obs.numpy().reshape(-1, 1)
        ).ravel()
        meta["rate_eff"] = meta["rate_mean"]

        # Color norm
        vals = []
        if np.isfinite(Z_eff).any():
            vals.append(Z_eff[np.isfinite(Z_eff)].ravel())
        vals.append(meta["rate_mean"].to_numpy())
        vals = np.concatenate([v[np.isfinite(v)] for v in vals])
        vmin, vmax = (
            (np.quantile(vals, 0.02), np.quantile(vals, 0.98))
            if vals.size > 0
            else (0, 1)
        )

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        cmap = plt.get_cmap(CMAP)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_facecolor("#d5e9ff")

        # Continental land (coastlines, no country borders)
        land = gpd.clip(_get_land_gdf(), ROI_BOX)
        land.plot(ax=ax, color="#f0f0f0", edgecolor="#999999", linewidth=0.5, zorder=0)

        in_language_mask_2d = in_language_mask.reshape(LON.shape)
        if np.isfinite(Z_eff).any():
            ax.pcolormesh(
                LON, LAT, Z_eff, shading="auto", cmap=cmap, norm=norm, zorder=1
            )

        ax.scatter(
            meta["longitude"],
            meta["latitude"],
            s=60,
            c=meta["rate_mean"],
            cmap=cmap,
            norm=norm,
            marker="o",
            edgecolor="white",
            linewidth=0.6,
            zorder=10,
        )

        threshold = np.percentile(meta["rate_mean"], 90)
        for lang, row in meta.iterrows():
            if row["rate_mean"] > threshold:
                ax.annotate(
                    lang,
                    (row["longitude"], row["latitude"]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.8,
                        linewidth=0.5,
                    ),
                    zorder=11,
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.3",
                        color="black",
                        lw=0.5,
                        alpha=0.6,
                    ),
                )

        ax.set_xlim(ROI_MINX, ROI_MAXX)
        ax.set_ylim(ROI_MINY, ROI_MAXY)
        ax.set_xlabel("Longitude", fontsize=22)
        ax.set_ylabel("Latitude", fontsize=22)
        ax.tick_params(axis="both", labelsize=22)
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.6)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.9)
        cbar.ax.tick_params(labelsize=22)
        cbar.set_label("GP Posterior Mean", fontsize=22, fontweight="bold")

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(
            output_dir, f"language_polygons_gridtrain_{tree_name}.pdf"
        )
        fig.savefig(fig_path, dpi=300)
        plt.close()
        print(f"    Saved: {fig_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Speech vs Cognate Rate Scatter (Extended Data Fig. 6)
# ═════════════════════════════════════════════════════════════════════════════


def plot_speech_vs_cognate_rates(output_dir=OUTPUT_DIR):
    """Scatter of speech vs cognate median rates for matched languages."""
    import matplotlib as mpl
    from scipy import stats

    # Apply figure3-style rcParams
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 15,
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

    speech = pd.read_csv(f"{REGRESSION_DIR}/speech_metadata_with_inventory.csv")
    cognate = pd.read_csv(f"{REGRESSION_DIR}/cognate_metadata_with_inventory.csv")

    cognate["language"] = cognate["language"].map(lambda x: COGNATE_TO_SPEECH.get(x, x))

    s = speech[["language", "rate_median"]].rename(
        columns={"rate_median": "speech_rate"}
    )
    c = cognate[["language", "rate_median"]].rename(
        columns={"rate_median": "cognate_rate"}
    )
    merged = s.merge(c, on="language")
    rho, pval = stats.spearmanr(merged["speech_rate"], merged["cognate_rate"])

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        merged["speech_rate"],
        merged["cognate_rate"],
        c="#555555",
        s=50,
        edgecolors="none",
        zorder=4,
    )

    for _, row in merged.iterrows():
        ax.annotate(
            row["language"],
            (row["speech_rate"], row["cognate_rate"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=10,
            color="#333333",
            clip_on=False,
        )

    # Tight axis limits with small padding
    x_vals = merged["speech_rate"].values
    y_vals = merged["cognate_rate"].values
    x_pad = (x_vals.max() - x_vals.min()) * 0.05
    y_pad = (y_vals.max() - y_vals.min()) * 0.06
    ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
    ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

    ax.set_xlabel("Median Bayesian Phylogenetic Rate (Speech)")
    ax.set_ylabel("Median Bayesian Phylogenetic Rate (Cognate)")
    ax.text(
        0.98,
        0.02,
        f"Spearman $\\rho$ = {rho:.3f}, p = {pval:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
        style="italic",
        color="#333333",
    )

    # bw_box style: all four spines, inward ticks, white bg, grid
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.9)
    ax.set_facecolor("white")
    ax.tick_params(
        top=True, right=True, which="both", direction="in", length=4.5, width=0.7
    )
    ax.grid(True, color="#dddddd", linewidth=0.5, zorder=0)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/speech_vs_cognate_rates.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tree_order = ["input_v12_combined_resampled", "heggarty2024_raw"]
    tree_labels = {
        "input_v12_combined_resampled": "Speech",
        "heggarty2024_raw": "Cognates",
    }

    # ── Pre-load GP libraries once (used in loop below) ──────────────────────
    _gpflow_ok = False
    try:
        import gpflow  # noqa: F811
        import tensorflow  # noqa: F401

        _gpflow_ok = True
    except ImportError as e:
        print(f"  gpflow/tensorflow not available — GP maps will be skipped: {e}")

    geojson = f"{DATA_DIR}/metadata/fleurs-r/language_polygons.geojson"

    for variant, mpaths in TREE_META_PATHS.items():
        out_dir = os.path.join(OUTPUT_DIR, variant)
        os.makedirs(out_dir, exist_ok=True)
        # Regression CSVs live in a variant-specific subdir of RESULTS_DIR
        results_dir_v = os.path.join(RESULTS_DIR, variant)
        if not os.path.isdir(results_dir_v):
            results_dir_v = RESULTS_DIR

        print("=" * 60)
        print(f"Rate vs longitude scatter  [{variant}]")
        print("=" * 60)
        plot_rate_vs_longitude(output_dir=out_dir, meta_paths=mpaths)

        print()
        print("=" * 60)
        print(f"Continuous GP maps  [{variant}]")
        print("=" * 60)
        if _gpflow_ok and os.path.exists(geojson):
            print("  Grid-trained GP maps...")
            plot_continuous_map_grid(
                output_dir=out_dir, geojson_path=geojson, meta_paths=mpaths
            )
        elif not _gpflow_ok:
            print("  Skipping (gpflow unavailable).")
        else:
            print(f"  GeoJSON not found: {geojson}")

        print()

    # ── Speech vs Cognate rate scatter ──────────────────────────────────────
    print("=" * 60)
    print("Speech vs Cognate rate scatter")
    print("=" * 60)
    plot_speech_vs_cognate_rates(output_dir=OUTPUT_DIR)

    # ── Root age comparison — same for both variants ───────────────────────
    print("=" * 60)
    print("Root age comparison")
    print("=" * 60)
    plot_root_age_comparison(output_dir=OUTPUT_DIR)

    # ── Rate-over-time — same for both variants ────────────────────────────
    speech_trees_path = f"{DATA_DIR}/trees/beast/speech/0.01_brsupport/input_combined_resampled.trees"
    cognate_trees_path = f"{DATA_DIR}/trees/beast/iecor/prunedtomodern.trees"
    if os.path.exists(speech_trees_path) and os.path.exists(cognate_trees_path):
        try:
            import dendropy

            trees_cognates_rated = dendropy.TreeList.get(
                path=cognate_trees_path,
                schema="nexus",
                preserve_underscores=True,
                extract_comment_metadata=True,
            )
            trees_speech_rated = dendropy.TreeList.get(
                path=speech_trees_path,
                schema="nexus",
                preserve_underscores=True,
                extract_comment_metadata=True,
            )
            plot_rate_over_time_normalized(
                trees_cognates_rated, trees_speech_rated, output_dir=OUTPUT_DIR
            )
            plot_pct_change_over_time(
                trees_cognates_rated, trees_speech_rated, output_dir=OUTPUT_DIR
            )
        except ImportError:
            print("  dendropy not installed; skipping.")

    print()
    print("Done.")
