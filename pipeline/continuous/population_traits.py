# pylint: disable=invalid-name

"""Extract per-recording (population) traits for contraband's `includePopVar` path.

Recomputes the per-recording, post-readout representations exactly as
`pipeline/continuous_embeddings.py` does, centres them within language, and writes:

  population_traits.pt    (N x nTraits) float64, within-language centred
  population_labels.json  language name per row
  population_traits.xml   ready-to-paste <populationTraits .../> element
  population_delta.json   Schaefer-Strimmer optimal shrinkage delta

The mean over rows is checked against the saved `mean_embeddings.pt` so the
representation is provably the same one the tree was built from.

Why centre within language: contraband's `populationTraitMatrix` labels rows
generically and `populateTraitValueMatrixEstimatedPopulationVariance` pools *every*
row when computing each trait's variance. Raw per-recording values would therefore
yield the total (between + within language) variance, and the `+1` tip term in
`PruneLikelihoodProcess.pruneNode` would inject a full cross-language variance of
noise at every tip. Centring makes the pooled column variance the within-language
variance, which is what that `+1` is meant to represent. (The Carnivora example
achieves the same thing by supplying several individuals of one species.)

Usage mirrors `pipeline.continuous_embeddings`, e.g.

    pixi run python -m pipeline.population_traits \
        --dataset fleurs-r --ckpt v7zqs3lv --dtype gelu \
        --subsets dev,test --glottocode indo1319 --max-per-lang 200
"""

import json
import os

import numpy as np
import torch

from src._config import DEFAULT_CONTINUOUS_DIR
from src.tasks.feature_extraction.base import prepare_everything

from pipeline.continuous_embeddings import (
    _aggregate_cached,
    _load_language_mapping,
    _resolve_from_wandb,
)
from pipeline.sentence_trees_discrete import groupby_and_agg


def shrinkage_delta(X):
    """Schaefer & Strimmer (2005) optimal shrinkage intensity toward the identity.

    R* = delta*I + (1-delta)*R_hat, delta* = sum_{i!=j} Var(r_ij) / sum_{i!=j} r_ij^2
    """
    n, p = X.shape
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Z = Xc / sd
    R = (Z.T @ Z) / (n - 1)
    off = ~np.eye(p, dtype=bool)

    # Var_hat(r_ij) = n/(n-1)^3 * sum_k (w_kij - w_bar_ij)^2, accumulated in blocks
    # to avoid materialising the n x p x p tensor.
    varR = np.zeros((p, p))
    Wbar = R * (n - 1) / n
    block = max(1, int(2e8 // (p * p * 8)))
    for start in range(0, n, block):
        Zb = Z[start : start + block]
        W = Zb[:, :, None] * Zb[:, None, :]
        varR += ((W - Wbar) ** 2).sum(0)
    varR *= n / (n - 1) ** 3
    return float(np.clip(varR[off].sum() / (R[off] ** 2).sum(), 0.0, 1.0))


def _resolve_cache(args, run_dir):
    """Fill in embeddings_cache/model_id without needing W&B.

    The run's own cfg.json already records both, so prefer it; fall back to the
    W&B config that `continuous_embeddings.main` reads.
    """
    cfg_path = f"{run_dir}/cfg.json"
    if args.embeddings_cache is None and os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        args.embeddings_cache = cfg.get("embeddings_cache")
        if cfg.get("model_id"):
            args.model_id = cfg["model_id"]
        print(f"Resolved from {cfg_path}: model_id={args.model_id}, cache={args.embeddings_cache}")

    if args.embeddings_cache is None and args.ckpt and not os.path.isfile(args.ckpt):
        model_id, cache = _resolve_from_wandb(args.ckpt)
        if model_id is not None:
            args.model_id = model_id
        args.embeddings_cache = cache
        print(f"Resolved from W&B: model_id={model_id}, cache={cache}")

    if args.embeddings_cache is None:
        raise SystemExit(
            "Could not resolve an embeddings cache. Pass --embeddings-cache explicitly; "
            "the live path would re-run feature extraction."
        )


def main():
    args = _cli()
    run_dir = args.out or f"{DEFAULT_CONTINUOUS_DIR}/{args.dtype}/{args.ckpt}"
    _resolve_cache(args, run_dir)

    inputs = prepare_everything(args)

    if inputs.embedding_cache is None:
        raise SystemExit(
            f"prepare_everything did not load the cache at {args.embeddings_cache}; "
            "check the path exists."
        )

    X_emb, y = _aggregate_cached(args, inputs)
    X_emb = X_emb.detach().cpu().double()
    y = y.detach().cpu()
    n_traits = X_emb.shape[1]
    print(f"per-recording matrix: {tuple(X_emb.shape)}")

    # --- identify the same active languages continuous_embeddings.py kept -------------
    means = groupby_and_agg(X_emb.float(), y, inputs.num_classes, agg="mean")
    active_mask = means.abs().sum(dim=1) > 0
    active_idx = torch.where(active_mask)[0]
    label_map = _load_language_mapping(args.dataset)
    names = [label_map.get(inputs.labels[i], inputs.labels[i]) for i in active_idx.tolist()]

    ref_path = f"{run_dir}/mean_embeddings.pt"
    if os.path.isfile(ref_path):
        ref = torch.load(ref_path, map_location="cpu", weights_only=False).double().cpu()
        got = means[active_mask].double()
        if ref.shape == got.shape and torch.allclose(ref, got, atol=1e-4, rtol=1e-3):
            print(f"OK  reproduces {ref_path} (max abs diff {(ref-got).abs().max():.2e})")
        else:
            raise SystemExit(
                f"MISMATCH against {ref_path} (shapes {tuple(ref.shape)} vs {tuple(got.shape)}, "
                f"max abs diff {(ref-got).abs().max() if ref.shape==got.shape else float('nan'):.2e}). "
                "Refusing to write population traits that do not match the tree's representation."
            )
    else:
        print(f"WARNING: {ref_path} not found — cannot verify against the saved means.")

    # --- restrict to active languages, subsample, centre within language --------------
    keep = torch.isin(y, active_idx)
    X_emb, y = X_emb[keep], y[keep]

    rng = np.random.default_rng(args.seed)
    rows, row_names = [], []
    per_lang = {}
    for pos, li in enumerate(active_idx.tolist()):
        sel = torch.where(y == li)[0].numpy()
        per_lang[names[pos]] = len(sel)
        if args.max_per_lang and len(sel) > args.max_per_lang:
            sel = rng.choice(sel, size=args.max_per_lang, replace=False)
        block = X_emb[torch.from_numpy(np.sort(sel))]
        if args.center != "none":
            block = block - block.mean(0, keepdim=True)  # within-language centring
        rows.append(block)
        row_names.extend([names[pos]] * block.shape[0])

    P = torch.cat(rows, 0)
    counts = np.array(list(per_lang.values()))
    print(
        f"languages: {len(names)}   recordings/lang: min={counts.min()} "
        f"median={int(np.median(counts))} max={counts.max()}"
    )
    print(f"population matrix: {tuple(P.shape)}  (centring: {args.center})")

    # --- shrinkage delta: from the population matrix vs from the 51 language means ----
    Pnp = P.numpy()
    d_pop = shrinkage_delta(Pnp)
    d_mean = shrinkage_delta(means[active_mask].double().numpy())
    print(f"\nSchaefer-Strimmer delta from population matrix ({P.shape[0]} rows) = {d_pop:.6f}")
    print(f"Schaefer-Strimmer delta from language means   ({len(names)} rows) = {d_mean:.6f}")
    print("  -> set delta= on BMPruneShrinkageLikelihood to the population value")

    # --- write outputs ----------------------------------------------------------------
    os.makedirs(run_dir, exist_ok=True)
    torch.save(P, f"{run_dir}/population_traits.pt")
    with open(f"{run_dir}/population_labels.json", "w", encoding="utf-8") as f:
        json.dump(row_names, f)
    with open(f"{run_dir}/population_delta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "delta_population": d_pop,
                "delta_language_means": d_mean,
                "n_rows": int(P.shape[0]),
                "n_traits": int(n_traits),
                "max_per_lang": args.max_per_lang,
                "center": args.center,
                "recordings_per_language": per_lang,
            },
            f,
            indent=2,
        )

    keys = " ".join(f"rec_{i}" for i in range(P.shape[0]))
    vals = " ".join(f"{v:.6g}" for v in Pnp.ravel())
    xml = (
        f'          <populationTraits id="popTraits" spec="parameter.RealParameter"\n'
        f'              minordimension="{n_traits}" keys="{keys}">\n'
        f"            {vals}\n"
        f"          </populationTraits>\n"
    )
    with open(f"{run_dir}/population_traits.xml", "w", encoding="utf-8") as f:
        f.write(xml)

    size_mb = os.path.getsize(f"{run_dir}/population_traits.xml") / 1e6
    print(f"\nwrote {run_dir}/population_traits.{{pt,xml}}, population_labels.json, population_delta.json")
    print(f"  XML fragment: {size_mb:.1f} MB — paste inside <distribution id=\"PCMLikelihood\" ...>")
    print(f'  and set includePopVar="true" deltaVar="0.0" delta="{d_pop:.6f}"')


def _cli():
    from src.tasks.feature_extraction.base import get_fleurs_parallel_args

    p = get_fleurs_parallel_args(with_common_args=True)
    p.add_argument(
        "--subsets",
        type=str,
        default=None,
        help="Comma-separated data subsets to include (train,dev,test). Cache only.",
    )
    p.add_argument("--dtype", required=True, help="Output subdirectory under continuous/")
    p.add_argument("--out", default=None, help="Output dir (default: continuous/<dtype>/<ckpt>)")
    p.add_argument(
        "--max-per-lang",
        type=int,
        default=200,
        help="Subsample this many recordings per language (0 = keep all)",
    )
    p.add_argument(
        "--center",
        choices=["within", "none"],
        default="within",
        help="'within' subtracts each language's own mean (recommended); 'none' keeps raw values",
    )
    return p.parse_args()  # --seed comes from the base parser and seeds the subsampling


if __name__ == "__main__":
    main()
