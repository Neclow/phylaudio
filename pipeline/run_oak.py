import json
import os
from argparse import ArgumentParser
from pathlib import Path

import gpflow
import numpy as np
import pandas as pd

from src._config import DEFAULT_BEAST_DIR
from src.tasks.oak import PhyloKernel, oak_model, save_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--tree-names",
        nargs="+",
        type=str,
        help="Input file paths",
        default=[
            # IECOR
            "iecor/raw_CCD.nex",
            # Ours
            "eab44e7f-54cc-4469-87d1-282cc81e02c2/0.25/long_v3_44_CCD.nex",
        ],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="fleurs-r",
        help="Dataset. Example: `fleurs`",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ─── Configuration ───────────────────────────────────────────────────────────
    np.random.seed(args.seed)

    for tree_name in args.tree_names:
        tree_path = f"{DEFAULT_BEAST_DIR}/{tree_name}"
        tree_dir = os.path.dirname(tree_path)
        tree_stem = Path(tree_path).stem
        # ─── 1) Load V_raw ───────────────────────────────────────────────────────
        vcv_df = pd.read_csv(f"{tree_dir}/{tree_stem}_vcv.csv", index_col=0)
        assert all(vcv_df.index == vcv_df.columns)
        languages = vcv_df.index.tolist()
        vcv = vcv_df.values.astype(np.float64)

        # ─── 2) Load metadata ────────────────────────────────────────────────────
        meta = pd.read_csv(
            f"{tree_dir}/{tree_stem}_metadata.csv", sep=","
        ).reset_index()
        meta = meta.rename(columns={"level_0": "language"})
        if "language" not in meta.columns:
            raise ValueError(
                "Your metadata CSV does not have a column named 'language'"
            )
        meta["language"] = meta["language"].astype(str).str.strip()
        meta = meta.set_index("language").loc[languages]

        # ─── 3) Build X and y ────────────────────────────────────────────────────
        n = len(languages)
        X = np.hstack(
            [
                meta["longitude"].to_numpy().reshape(-1, 1),
                meta["latitude"].to_numpy().reshape(-1, 1),
                np.log(meta["n_speakers"].to_numpy()).reshape(-1, 1),
                np.arange(n).reshape(-1, 1),  # tip index
            ]
        )
        y = np.log(meta["rate_median"].to_numpy().reshape(-1, 1))

        # ─── 4) OAK model setup ──────────────────────────────────────────────────
        active_dims = [[0], [1], [2]]
        variable_names = ["lon", "lat", "log_speakers"]

        oak = oak_model(
            num_inducing=500,
            max_interaction_depth=2,
            use_sparsity_prior=True,
            sparse=False,
            base_kernels=[gpflow.kernels.RBF for _ in range(3)],
            active_dims=active_dims,
            noise_kernel=PhyloKernel(vcv, active_dims=[3]),
            lam_concurvity=0.0,
            empirical_measure=[[0], [1]],
        )

        # ─── 5) Fit ──────────────────────────────────────────────────────────────
        oak.fit(X, y, initialise_inducing_points=True, optimise=True)

        # ─── 6) Save model ───────────────────────────────────────────────────────
        save_model(oak.m, filename=Path(f"{tree_dir}/{tree_stem}_oak_model"))

        # ─── 7) Evaluate ─────────────────────────────────────────────────────────
        y_pred = oak.predict(X).reshape(-1)
        y_true = y.reshape(-1)
        r_squared = 1 - np.sum((y_true - y_pred) ** 2) / np.sum(
            (y_true - np.mean(y)) ** 2
        )
        print(f"R² for tree {tree_name}: {r_squared:.4f}")

        # ─── 8) Sobol indices ────────────────────────────────────────────────────
        sobol_df = oak.sobol_summary(
            covariate_names=variable_names,
            likelihood_variance=True,
            return_variance=True,
        )
        sobol_df["sobol_index"] = sobol_df["sobol_index"].apply(lambda x: f"{x:.8f}")
        sobol_df.to_csv(f"{tree_dir}/{tree_stem}_oak_sobol.csv", index=False)

        # ─── 10) Save metrics ────────────────────────────────────────────────────
        with open(
            f"{tree_dir}/{tree_stem}_oak_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "lambda": oak.m.kernel.noise_kernel.lambda_.numpy().tolist(),
                    "r_squared": r_squared,
                },
                f,
            )
