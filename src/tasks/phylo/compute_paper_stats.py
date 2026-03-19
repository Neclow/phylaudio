#!/usr/bin/env python3
"""Compute key statistics for the paper from posterior samples."""

import numpy as np
import pandas as pd
from scipy import stats

from src.tasks.phylo.constants import COGNATE_TO_SPEECH

RESULTS_DIR = "data/phyloregression/with_inventory"
REGRESSION_DIR = "data/phyloregression"
SPEECH_TREE = "input_v12_combined_resampled"
COGNATE_TREE = "heggarty2024_raw"


def report_coef_ci(tree):
    """Report 95% CI for delta_norm coefficient."""
    path = f"{RESULTS_DIR}/coef_samples_linear_geo_{tree}.csv"
    df = pd.read_csv(path)
    samples = df["delta_norm"].dropna().values
    lo, hi = np.percentile(samples, [2.5, 97.5])
    mean = np.mean(samples)
    median = np.median(samples)
    print(f"\n── delta coefficient ({tree}) ──")
    print(f"  N samples: {len(samples)}")
    print(f"  Mean:   {mean:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  95% CI: [{lo:.3f}, {hi:.3f}]")


def report_variance_decomposition(tree):
    """Report Shapley variance decomposition for all terms."""
    path = f"{RESULTS_DIR}/variance_samples_gp_geo_{tree}.csv"
    df = pd.read_csv(path)

    shapley_cols = [c for c in df.columns if c.startswith("shapley_")]
    print(f"\n── Shapley variance decomposition ({tree}) ──")
    for col in shapley_cols:
        vals = df[col].dropna().values
        name = col.replace("shapley_", "").replace("_norm", "")
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {name:40s}  mean={np.mean(vals):.3f}  "
              f"median={np.median(vals):.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")


def report_speech_vs_cognate_correlation():
    """Spearman correlation between speech and cognate evolutionary rates."""
    speech_path = f"{REGRESSION_DIR}/speech_metadata_with_inventory.csv"
    cognate_path = f"{REGRESSION_DIR}/cognate_metadata_with_inventory.csv"

    speech = pd.read_csv(speech_path)
    cognate = pd.read_csv(cognate_path)

    # Harmonise language names
    cognate["language"] = cognate["language"].map(
        lambda x: COGNATE_TO_SPEECH.get(x, x))

    s = speech[["language", "rate_median"]].rename(
        columns={"rate_median": "speech_rate"})
    c = cognate[["language", "rate_median"]].rename(
        columns={"rate_median": "cognate_rate"})
    merged = s.merge(c, on="language")

    rho, pval = stats.spearmanr(merged["speech_rate"], merged["cognate_rate"])
    r, p_pearson = stats.pearsonr(merged["speech_rate"], merged["cognate_rate"])

    # Bootstrap 95% CI for Spearman
    rng = np.random.default_rng(42)
    n_boot = 10000
    rhos = np.empty(n_boot)
    speech_vals = merged["speech_rate"].values
    cognate_vals = merged["cognate_rate"].values
    n = len(merged)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rhos[i] = stats.spearmanr(speech_vals[idx], cognate_vals[idx])[0]
    lo, hi = np.percentile(rhos, [2.5, 97.5])

    print(f"\n── Speech vs Cognate rate correlation ──")
    print(f"  N languages: {n}")
    print(f"  Spearman rho = {rho:.3f}, p = {pval:.4f}")
    print(f"  Spearman 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
    print(f"  Pearson r = {r:.3f}, p = {p_pearson:.4f}")


if __name__ == "__main__":
    for tree in [SPEECH_TREE, COGNATE_TREE]:
        try:
            report_coef_ci(tree)
        except FileNotFoundError as e:
            print(f"  Skipping coef for {tree}: {e}")
        try:
            report_variance_decomposition(tree)
        except FileNotFoundError as e:
            print(f"  Skipping variance for {tree}: {e}")

    try:
        report_speech_vs_cognate_correlation()
    except FileNotFoundError as e:
        print(f"  Skipping correlation: {e}")
