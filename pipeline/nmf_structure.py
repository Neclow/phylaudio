"""
Run NMF K-sweep on a BEAST binary alignment and produce STRUCTURE-style plots.

Usage:
    python -m pipeline.nmf_structure <run_id> [options]

Example:
    python -m pipeline.nmf_structure c59
"""

import os
import pickle
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    MetavarTypeHelpFormatter,
)
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from src._config import DEFAULT_BEAST_DIR, DEFAULT_MAPPED_FASTA_FILE
from src.tasks.phylo.fasta import to_numpy
from src.tasks.phylo.nmf import (
    choose_k,
    nmf_k_sweep,
    normalize_rows_to_proportions,
    plot_k_diagnostics,
    plot_structure,
)


class MixedFormatter(ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter):
    pass


def parse_args():
    parser = ArgumentParser(
        description="Run NMF K-sweep on a BEAST binary alignment and produce STRUCTURE-style plots",
        formatter_class=MixedFormatter,
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="BEAST run UUID, prefix, or full path",
    )
    parser.add_argument("--k-min", type=int, default=2, help="Minimum K")
    parser.add_argument("--k-max", type=int, default=30, help="Maximum K")
    parser.add_argument("--n-restarts", type=int, default=20, help="NMF restarts per K")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--error-slack",
        type=float,
        default=0.40,
        help="Error slack for K selection",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="kullback-leibler",
        choices=["kullback-leibler", "frobenius"],
        help="NMF loss function",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--dpi", type=int, default=200, help="Plot DPI")
    return parser.parse_args()


def resolve_beast_dir(run_id):
    """Resolve a run ID (UUID, prefix, or full path) to a BEAST run directory."""
    if os.path.isdir(run_id):
        return run_id
    matches = sorted(glob(f"{DEFAULT_BEAST_DIR}/{run_id}*"))
    dirs = [m for m in matches if os.path.isdir(m)]
    if len(dirs) == 1:
        return dirs[0]
    if len(dirs) == 0:
        raise FileNotFoundError(
            f"No BEAST run matching '{run_id}' in {DEFAULT_BEAST_DIR}/"
        )
    raise ValueError(f"Ambiguous run_id '{run_id}': matches {dirs}")


def find_fasta(beast_dir):
    """Glob for the mapped FASTA alignment inside a BEAST run directory."""
    hits = sorted(glob(f"{beast_dir}/**/{DEFAULT_MAPPED_FASTA_FILE}", recursive=True))
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        raise FileNotFoundError(
            f"No {DEFAULT_MAPPED_FASTA_FILE} found in {beast_dir}/"
        )
    raise ValueError(
        f"Multiple {DEFAULT_MAPPED_FASTA_FILE} found in {beast_dir}/: {hits}"
    )


def main():
    args = parse_args()

    beast_dir = resolve_beast_dir(args.run_id)
    fa_path = find_fasta(beast_dir)
    out_dir = f"{os.path.dirname(fa_path)}/nmf"
    os.makedirs(out_dir, exist_ok=True)

    # -- Parse FASTA --
    X, labels = to_numpy(fa_path)
    n, L = X.shape
    print(f"Loaded {n} languages × {L} sites from {fa_path}")
    print(f"Matrix range: [{X.min():.3f}, {X.max():.3f}]")

    # -- NMF K-sweep (with pickle caching) --
    pkl_name = f"nmf_sweep_k{args.k_min}_k{args.k_max}.pkl"
    pkl_path = f"{out_dir}/{pkl_name}"

    if os.path.isfile(pkl_path):
        print(f"Loading cached results from {pkl_path}")
        with open(pkl_path, "rb") as f:
            cached = pickle.load(f)
        results = cached["results"]
        labels = cached["labels"]
    else:
        results = nmf_k_sweep(
            X,
            k_min=args.k_min,
            k_max=args.k_max,
            n_restarts=args.n_restarts,
            seed0=args.seed,
            loss=args.loss,
        )
        with open(pkl_path, "wb") as f:
            pickle.dump({"results": results, "labels": labels}, f)
        print(f"Saved results to {pkl_path}")

    k_star = choose_k(results, error_slack=args.error_slack)
    print(f"\n>>> Chosen K = {k_star}\n")

    # -- Plots --
    ks = sorted(results.keys())

    if not args.no_plots:
        fig = plot_k_diagnostics(results, k_star=k_star)
        diag_path = f"{out_dir}/nmf_diagnostics_k{args.k_min}_k{args.k_max}.png"
        fig.savefig(diag_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {diag_path}")

        for k_val in ks:
            W = results[k_val]["best"]["W"]
            out_path = f"{out_dir}/nmf_structure_K{k_val:02d}.png"
            fig = plot_structure(
                W,
                labels=labels,
                k_val=k_val,
                err=results[k_val]["best"]["err"],
                stability=results[k_val]["stability"],
            )
            fig.savefig(out_path, dpi=args.dpi)
            plt.close(fig)
        print(f"Saved {len(ks)} structure plots to {out_dir}/")

    # -- Print cluster assignments --
    for k_val in ks:
        W = results[k_val]["best"]["W"]
        P = normalize_rows_to_proportions(W)
        print(f"\n── K={k_val} (stab={results[k_val]['stability']:.3f}) ──")
        for comp_id in range(k_val):
            members = [labels[i] for i in range(n) if np.argmax(P[i]) == comp_id]
            if members:
                print(f"  {comp_id + 1}: {', '.join(sorted(members))}")


if __name__ == "__main__":
    main()
