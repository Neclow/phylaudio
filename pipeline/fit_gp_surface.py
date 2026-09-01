"""Fit a GPflow GP surface for the geographic regression map.

Inputs:  a BEAST run ID and subdirectory
Options: --cognate_beast_dir, --variant
Flow:    Load regression metadata and language polygons
                    |
                    v
         Fit Matern-3/2 GP over language polygon centroids
                    |
                    v
         Predict on a dense geographic grid
Outputs: gp_cache/gp_grid.npz, gp_obs.csv, land_clipped.geojson in the BEAST run directory
"""

import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from collections import Counter, defaultdict

import geopandas as gpd
import gpflow
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import box as shapely_box
from shapely.prepared import prep
from shapely.strtree import STRtree
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src._config import (
    COGNATE_BEAST_DIR,
    EXCLUDE_LANGUAGES,
    GEOJSON_EXPANSION,
    GEOJSON_PATH,
    NE_COUNTRIES_PATH,
    NE_LAND_PATH,
)
from src.tasks.phylo.beast import resolve_beast_dir

GRID_N_LON_TRAIN = 150
GRID_N_LAT_TRAIN = 100
GRID_N_LON_PRED = 600
GRID_N_LAT_PRED = 400
FIXED_NOISE = 1e-2
MAXITER = 500
PAD_DEG = 15


def load_language_polygons(filepath=GEOJSON_PATH):
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


def _compute_roi(countries_path=NE_COUNTRIES_PATH):
    world = gpd.read_file(countries_path)
    if "NAME" in world.columns and "name" not in world.columns:
        world = world.rename(columns={"NAME": "name"})
    ire = world.loc[world["name"] == "Ireland"].total_bounds
    bgd = world.loc[world["name"] == "Bangladesh"].total_bounds
    roi_minx = min(ire[0], bgd[0]) - PAD_DEG
    roi_maxx = max(ire[2], bgd[2]) + PAD_DEG
    roi_miny = min(ire[1], bgd[1]) - PAD_DEG
    roi_maxy = max(ire[3], bgd[3]) + PAD_DEG
    return roi_minx, roi_maxx, roi_miny, roi_maxy


def _build_training_data(gdf_clipped, roi_box):
    lon_grid = np.linspace(roi_box.bounds[0], roi_box.bounds[2], GRID_N_LON_TRAIN)
    lat_grid = np.linspace(roi_box.bounds[1], roi_box.bounds[3], GRID_N_LAT_TRAIN)
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
        return None, None
    X_train = np.array(list(point_rates.keys()))
    y_train = np.array([np.mean(v) for v in point_rates.values()])
    return X_train, y_train


def fit_gp_surface(meta_path, out_dir):
    meta = pd.read_csv(meta_path)
    meta_idx = meta.set_index("language")
    meta_idx = meta_idx.loc[~meta_idx.index.isin(EXCLUDE_LANGUAGES)].copy()

    roi = _compute_roi()
    roi_minx, roi_maxx, roi_miny, roi_maxy = roi
    roi_box = shapely_box(roi_minx, roi_miny, roi_maxx, roi_maxy)

    gdf_language = load_language_polygons()
    gdf_clipped = gpd.clip(gdf_language, roi_box)
    gdf_clipped = gdf_clipped[gdf_clipped["name"].isin(meta_idx.index)].copy()
    gdf_clipped["rate"] = gdf_clipped["name"].map(meta_idx["rate_median"])
    gdf_clipped = gdf_clipped.dropna(subset=["rate"]).reset_index(drop=True)

    X_train, y_train = _build_training_data(gdf_clipped, roi_box)
    if X_train is None:
        print("  No training data — skipping.")
        return

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
    pbar = tqdm(desc="Optimizing GP")

    # pylint: disable=unused-argument
    def _step_callback(step, variables, values):
        pbar.update(1)

    # pylint: enable=unused-argument

    gpflow.optimizers.Scipy().minimize(
        model.training_loss,
        variables=model.trainable_variables,
        step_callback=_step_callback,
        options=dict(maxiter=MAXITER),
    )
    pbar.close()

    lon_pred = np.linspace(roi_minx, roi_maxx, GRID_N_LON_PRED)
    lat_pred = np.linspace(roi_miny, roi_maxy, GRID_N_LAT_PRED)
    LON, LAT = np.meshgrid(lon_pred, lat_pred)
    grid_pred = np.column_stack([LON.ravel(), LAT.ravel()])

    language_union = gdf_clipped.union_all()
    language_union_prep = prep(language_union)
    in_mask = np.array(
        [
            language_union_prep.contains(Point(p[0], p[1]))
            for p in tqdm(grid_pred, desc="Masking grid points")
        ]
    )

    grid_inside = grid_pred[in_mask]
    mean_all = []
    batch_iter = range(0, len(grid_inside), 5000)
    for i in tqdm(batch_iter, desc="Predicting GP surface"):
        batch = grid_inside[i : i + 5000]
        Xg = scaler_X.transform(batch)
        m, _ = model.predict_f(Xg)
        mean_all.append(m.numpy().ravel())
    mean_inside = np.concatenate(mean_all)

    Z = np.full(len(grid_pred), np.nan, dtype=float)
    Z[in_mask] = scaler_y.inverse_transform(mean_inside.reshape(-1, 1)).ravel()
    Z = Z.reshape(LON.shape)

    X_obs = meta_idx[["longitude", "latitude"]].to_numpy()
    m_obs, _ = model.predict_f(scaler_X.transform(X_obs))
    meta_idx["rate_gp"] = scaler_y.inverse_transform(
        m_obs.numpy().reshape(-1, 1)
    ).ravel()

    land = gpd.read_file(NE_LAND_PATH)
    land_clipped = gpd.clip(land, roi_box)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "gp_grid.npz"),
        LON=LON,
        LAT=LAT,
        Z=Z,
        roi=np.array(roi),
    )
    meta_idx[["longitude", "latitude", "rate_gp"]].to_csv(
        os.path.join(out_dir, "gp_obs.csv")
    )
    land_clipped.to_file(
        os.path.join(out_dir, "land_clipped.geojson"), driver="GeoJSON"
    )
    print(f"  Saved GP results to {out_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit GP surface for geographic regression map."
    )
    parser.add_argument("run_id", help="BEAST run UUID, prefix, or full path")
    parser.add_argument("subdir", help="Subdirectory name or prefix within the run")
    parser.add_argument(
        "--cognate_beast_dir",
        default=COGNATE_BEAST_DIR,
        help="BEAST directory for cognates (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    speech_beast_dir = resolve_beast_dir(args.run_id, args.subdir)
    beast_dirs = {"speech": speech_beast_dir, "cognate": args.cognate_beast_dir}

    for stem, beast_dir in beast_dirs.items():
        meta_path = f"{beast_dir}/metadata_with_inventory.csv"
        out_dir = f"{beast_dir}/gp_cache"
        print(f"Fitting GP surface for {stem}...")
        fit_gp_surface(meta_path, out_dir)
