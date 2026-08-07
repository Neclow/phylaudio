"""Figure 3: geographic regression panels (linear_geo model)."""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats

from src._config import EXCLUDE_LANGUAGES

from ._config import (
    COGNATE_BEAST_DIR,
    DEFAULT_IMG_DIR,
    DEFAULT_STYLE,
    SPEECH_BEAST_DIR,
)

IMG_DIR = f"{DEFAULT_IMG_DIR}/fig3"
FIGSIZE = (5, 4)
SCMAP = "viridis"
MAP_CMAP = "magma"


_VAR_DISPLAY = {
    "longitude": "Longitude",
    "latitude": "Latitude",
    "log_n_speakers": "log_n_speakers",
    "n_phonemes": "n_phonemes",
    "delta": "delta",
}

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
PUBLICATION_LABELS = {
    "Longitude": "Longitude",
    "Latitude": "Latitude",
    "log_n_speakers": "Log(speakers)",
    "n_phonemes": "Inventory size",
    "delta": "Network signal (δ)",
    "Longitude×Latitude": "Longitude × Latitude",
    "Longitude×log_n_speakers": "Longitude × Log(speakers)",
    "Longitude×delta": "Longitude × Network signal",
    "Latitude×log_n_speakers": "Latitude × Log(speakers)",
    "Latitude×delta": "Latitude × Network signal",
}

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


# Display-name helpers
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


# Helpers
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


def _inset_cbar(ax, sm, label, loc="lower right"):
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


def _shared_component_order(dat):
    eff_df = _extract_coefficient_data(dat["summary"])
    mains = sorted(eff_df[eff_df["_n_vars"] == 1]["coefficient"].unique())
    inters = sorted(eff_df[eff_df["_n_vars"] == 2]["coefficient"].unique())
    return mains + inters


# Data loading
def load_results(beast_dir):
    regression_dir = f"{beast_dir}/phyloregression/with_inventory"
    return dict(
        meta=pd.read_csv(f"{beast_dir}/metadata_with_inventory.csv"),
        coef=pd.read_csv(f"{regression_dir}/coef_samples_linear_geo.csv"),
        variance=pd.read_csv(f"{regression_dir}/variance_samples_linear_geo.csv"),
        summary=pd.read_csv(f"{regression_dir}/phylolm_linear_geo.csv"),
        gp_variance=pd.read_csv(f"{regression_dir}/variance_samples_gp_geo.csv"),
    )


def load_gp_surface(beast_dir):
    cache_dir = f"{beast_dir}/gp_cache"
    grid = np.load(f"{cache_dir}/gp_grid.npz")
    obs = pd.read_csv(f"{cache_dir}/gp_obs.csv", index_col=0)
    land = gpd.read_file(f"{cache_dir}/land_clipped.geojson")
    return dict(
        LON=grid["LON"],
        LAT=grid["LAT"],
        Z=grid["Z"],
        roi=tuple(grid["roi"]),
        obs=obs,
        land=land,
    )


# Panel a: posterior coefficient distributions
def plot_panel_a(dat, comp_order, output_stem):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        coef_df = dat["coef"]
        n = len(comp_order)
        display_to_y = {d: i for i, d in enumerate(comp_order)}
        term_colors = {d: VARIANCE_COLORS.get(d, "#888888") for d in comp_order}

        def col_to_display(col):
            return _coef_suffix_to_display(col.replace("_norm", "").replace(".", "_"))

        for col in coef_df.columns:
            if col in ("sample_id", "beast_dir", "Intercept"):
                continue
            disp = col_to_display(col)
            if disp not in display_to_y:
                continue
            samples = coef_df[col].dropna().values
            if len(samples) < 10:
                continue
            y_base = display_to_y[disp]
            kde = stats.gaussian_kde(samples, bw_method="scott")
            xs = np.linspace(
                np.percentile(samples, 0.5), np.percentile(samples, 99.5), 300
            )
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
        ax.set_xlabel("Effect on acoustic evolutionary rate")

        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/{output_stem}.{ext}", bbox_inches="tight")
        print(f"Saved {IMG_DIR}/{output_stem}.{{pdf,svg}}")
        plt.close(fig)


# Panel b: Shapley variance attribution
def plot_panel_b(dat, comp_order, output_stem):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)

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
            xs = np.linspace(
                np.percentile(samples, 0.5), np.percentile(samples, 99.5), 300
            )
            dens = kde(xs)
            dens = dens / dens.max() * 0.40
            color = VARIANCE_COLORS.get(disp, "#888888")
            ax.fill_between(
                xs, y_base - dens, y_base + dens, color=color, alpha=0.50, linewidth=0
            )
            ax.plot(xs, y_base - dens, color=color, alpha=0.55, linewidth=0.4)
            ax.plot(xs, y_base + dens, color=color, alpha=0.55, linewidth=0.4)

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
        ax.set_yticklabels([PUBLICATION_LABELS.get(c, c) for c in comp_order])
        ax.set_ylim(-0.7, n - 0.3)
        ax.margins(x=0.07)
        ax.set_xlabel(
            "Contribution to variance of acoustic\nevolutionary rate (Shapley attribution)"
        )

        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/{output_stem}.{ext}", bbox_inches="tight")
        print(f"Saved {IMG_DIR}/{output_stem}.{{pdf,svg}}")
        plt.close(fig)


# Panel c: delta vs rate scatter
def plot_panel_c(dat, output_stem):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        meta = dat["meta"][~dat["meta"]["language"].isin(EXCLUDE_LANGUAGES)].copy()
        coef_df = dat["coef"]
        meta["log_rate"] = np.log(meta["rate_median"])
        meta = meta.set_index("language")

        mu_delta = meta["delta"].mean()
        sd_delta = meta["delta"].std(ddof=1)
        b_int = coef_df["Intercept"].values
        b_del = coef_df["delta_norm"].values
        delta_seq = np.linspace(
            meta["delta"].min() - 0.02, meta["delta"].max() + 0.02, 300
        )
        delta_n = (delta_seq - mu_delta) / sd_delta
        preds = b_int[:, None] + b_del[:, None] * delta_n[None, :]
        med = np.median(preds, axis=0)
        lo95 = np.percentile(preds, 2.5, axis=0)
        hi95 = np.percentile(preds, 97.5, axis=0)

        ax.fill_between(
            delta_seq, lo95, hi95, color="#aaaaaa", alpha=0.30, linewidth=0, zorder=2
        )
        ax.plot(delta_seq, med, color="#555555", linewidth=1.4, zorder=3)

        langs = list(meta.index)

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

        cols = [
            c
            for c in coef_df.columns
            if c in X_dict and c != "sample_id" and c != "beast_dir"
        ]
        X = np.column_stack([X_dict[c] for c in cols])
        B = coef_df[cols].values
        fitted = (B.mean(axis=0)[None, :] * X).sum(axis=1)

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

        xs = meta.loc[langs, "delta"].values
        ys = meta.loc[langs, "log_rate"].values

        lo95_at_x = np.interp(xs, delta_seq, lo95)
        hi95_at_x = np.interp(xs, delta_seq, hi95)
        label_mask = (ys < lo95_at_x) | (ys > hi95_at_x)

        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        placed_boxes = []

        def _get_bbox_data(txt):
            bb = txt.get_window_extent(renderer=renderer)
            (dx0, dy0), (dx1, dy1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            return (dx0, dy0, dx1, dy1)

        def _overlaps(box):
            for pb in placed_boxes:
                if (
                    box[0] < pb[2]
                    and box[2] > pb[0]
                    and box[1] < pb[3]
                    and box[3] > pb[1]
                ):
                    return True
            return False

        offsets_pt = [
            (6, 6),
            (-6, 6),
            (6, -10),
            (-6, -10),
            (12, 0),
            (-12, 0),
            (0, 10),
            (0, -14),
        ]

        for i, lang in enumerate(langs):
            if not label_mask[i]:
                continue
            best_txt = None
            for dx, dy in offsets_pt:
                txt = ax.annotate(
                    lang,
                    (xs[i], ys[i]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    color="#333333",
                    clip_on=True,
                )
                box = _get_bbox_data(txt)
                if not _overlaps(box):
                    placed_boxes.append(box)
                    best_txt = txt
                    break
                txt.remove()
            if best_txt is None:
                txt = ax.annotate(
                    lang,
                    (xs[i], ys[i]),
                    xytext=offsets_pt[0],
                    textcoords="offset points",
                    color="#333333",
                    clip_on=True,
                )
                placed_boxes.append(_get_bbox_data(txt))

        sm_dot = ScalarMappable(norm=dot_norm, cmap=SCMAP)
        sm_dot.set_array([])
        _inset_cbar(ax, sm_dot, "Fitted log rate\n(linear model)", loc="lower right")

        x_vals = meta.loc[langs, "delta"].values
        y_vals = meta.loc[langs, "log_rate"].values
        x_span = x_vals.max() - x_vals.min()
        y_span = y_vals.max() - y_vals.min()
        ax.set_xlim(x_vals.min() - x_span * 0.08, x_vals.max() + x_span * 0.18)
        ax.set_ylim(y_vals.min() - y_span * 0.08, y_vals.max() + y_span * 0.08)

        ax.set_xlabel("Network signal (δ)")
        ax.set_ylabel("Log Median Bayesian Phylogenetic Rate")

        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/{output_stem}.{ext}", bbox_inches="tight")
        print(f"Saved {IMG_DIR}/{output_stem}.{{pdf,svg}}")
        plt.close(fig)


# Panel d: GP regression surface map
def plot_panel_d(gp_result, output_stem):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        LON, LAT, Z = gp_result["LON"], gp_result["LAT"], gp_result["Z"]
        obs = gp_result["obs"]
        land = gp_result["land"]
        roi_minx, roi_maxx, roi_miny, roi_maxy = gp_result["roi"]

        vals = []
        if np.isfinite(Z).any():
            vals.append(Z[np.isfinite(Z)].ravel())
        vals.append(obs["rate_gp"].to_numpy())
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

        ax.scatter(
            obs["longitude"].values,
            obs["latitude"].values,
            c=obs["rate_gp"].values,
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
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(True, color="white", linewidth=0.35, alpha=0.6, zorder=3)

        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/{output_stem}.{ext}", bbox_inches="tight")
        print(f"Saved {IMG_DIR}/{output_stem}.{{pdf,svg}}")
        plt.close(fig)


# Main
if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)

    for label, beast_dir in [
        ("speech", SPEECH_BEAST_DIR),
        ("cognate", COGNATE_BEAST_DIR),
    ]:
        is_main = label == "speech"
        prefix = "fig3" if is_main else "figS6"

        print(f"Plotting {label}...")
        dat = load_results(beast_dir)
        comp_order = _shared_component_order(dat)

        plot_panel_a(dat, comp_order, f"{prefix}a_coef_{label}")
        plot_panel_b(dat, comp_order, f"{prefix}b_shapley_{label}")
        plot_panel_c(dat, f"{prefix}c_delta_vs_rate_{label}")

        gp = load_gp_surface(beast_dir)
        plot_panel_d(gp, f"{prefix}d_gp_map_{label}")

    print("Done.")
