"""PCA-reduced continuous embeddings for contraband BEAST runs.

Fits PCA on the train split of GELU 256-dim embeddings, retaining components
that explain a target fraction of variance.  Transforms dev+test, then produces
per-language means and within-language-centred population traits.

Usage:
    pixi run python -m pipeline.pca_continuous \
        --ckpt v7zqs3lv --dataset fleurs-r --dtype gelu \
        --glottocode indo1319 --var-explained 0.95 \
        --max-per-lang 200 --out data/trees/continuous/gelu_pca95/v7zqs3lv
"""

import json
import os

import numpy as np
import torch
from sklearn.decomposition import PCA

from src._config import DEFAULT_CONTINUOUS_DIR, DEFAULT_METADATA_DIR, DEFAULT_METADATA_KEY
from src.tasks.feature_extraction.base import (
    get_fleurs_parallel_args,
    post_process_embeddings,
    prepare_everything,
)
from pipeline.continuous_embeddings import _load_language_mapping, _resolve_from_wandb
from pipeline.population_traits import shrinkage_delta
from pipeline.sentence_trees_discrete import groupby_and_agg


def parse_args():
    p = get_fleurs_parallel_args(with_common_args=True)
    p.add_argument("--dtype", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--var-explained", type=float, default=0.95)
    p.add_argument("--max-per-lang", type=int, default=200)
    p.add_argument("--subsets-pca", type=str, default="train",
                   help="Subset(s) to fit PCA on (default: train)")
    p.add_argument("--subsets-infer", type=str, default="dev,test",
                   help="Subset(s) for the phylo data (default: dev,test)")
    return p.parse_args()


def _load_and_project(args):
    """Load the embedding cache, apply GELU projector, return (X, y, meta)."""
    # Resolve W&B run
    if args.ckpt is not None and not os.path.isfile(args.ckpt):
        model_id, cache = _resolve_from_wandb(args.ckpt)
        if model_id is not None:
            args.model_id = model_id
        if cache is not None and args.embeddings_cache is None:
            args.embeddings_cache = cache

    inputs = prepare_everything(args)
    embeddings, meta = inputs.embedding_cache

    y_all = torch.as_tensor(meta["label"].to_numpy(), device="cpu")
    embeddings = embeddings.to("cpu")

    with torch.no_grad():
        X_all, y_all = post_process_embeddings(inputs, embeddings, y_all)

    X_all = X_all.detach().cpu().float()
    y_all = y_all.detach().cpu()

    return X_all, y_all, meta, inputs


def main():
    args = parse_args()
    args.device = "cpu"

    X_all, y_all, meta, inputs = _load_and_project(args)
    print(f"Full projected matrix: {tuple(X_all.shape)}")

    subsets_all = meta["subset"].to_numpy()

    # --- Fit PCA on train split ---
    train_mask = np.isin(subsets_all, args.subsets_pca.split(","))
    X_train = X_all[torch.from_numpy(train_mask)].numpy()
    print(f"PCA fit on {args.subsets_pca}: {X_train.shape[0]} recordings")

    pca = PCA(n_components=args.var_explained, svd_solver="full")
    pca.fit(X_train)
    n_components = int(pca.n_components_)
    cum_var = pca.explained_variance_ratio_.cumsum()
    print(f"PCA: {n_components} components explain {cum_var[-1]*100:.1f}% of variance")

    # --- Transform dev+test ---
    infer_mask = np.isin(subsets_all, args.subsets_infer.split(","))

    X_infer = pca.transform(X_all[torch.from_numpy(infer_mask)].numpy())
    y_infer = y_all[torch.from_numpy(infer_mask)]
    X_infer = torch.from_numpy(X_infer).float()
    print(f"Infer split ({args.subsets_infer}): {X_infer.shape}")

    # --- Per-language means ---
    means = groupby_and_agg(X_infer, y_infer, inputs.num_classes, agg="mean")
    active_mask = means.abs().sum(dim=1) > 0
    active_idx = torch.where(active_mask)[0]
    label_map = _load_language_mapping(args.dataset)
    names = [label_map.get(inputs.labels[i], inputs.labels[i]) for i in active_idx.tolist()]
    mean_active = means[active_mask]
    print(f"Active languages: {len(names)}, traits: {mean_active.shape[1]}")

    # --- Population traits (within-language centred, subsampled) ---
    rng = np.random.default_rng(42)
    rows, row_names = [], []
    per_lang = {}
    for pos, li in enumerate(active_idx.tolist()):
        sel = torch.where(y_infer == li)[0].numpy()
        per_lang[names[pos]] = len(sel)
        if args.max_per_lang and len(sel) > args.max_per_lang:
            sel = rng.choice(sel, size=args.max_per_lang, replace=False)
        block = X_infer[torch.from_numpy(np.sort(sel))].double()
        block = block - block.mean(0, keepdim=True)
        rows.append(block)
        row_names.extend([names[pos]] * block.shape[0])

    P = torch.cat(rows, 0)
    print(f"Population matrix: {tuple(P.shape)}")

    # --- Shrinkage delta ---
    Pnp = P.numpy()
    d_pop = shrinkage_delta(Pnp)
    d_mean = shrinkage_delta(mean_active.double().numpy())
    print(f"Shrinkage delta (population, {P.shape[0]} rows): {d_pop:.6f}")
    print(f"Shrinkage delta (language means, {len(names)} rows): {d_mean:.6f}")

    # --- Write outputs ---
    out_dir = args.out or f"{DEFAULT_CONTINUOUS_DIR}/gelu_pca{int(args.var_explained*100)}/{args.ckpt}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(mean_active, f"{out_dir}/mean_embeddings.pt")
    torch.save(P, f"{out_dir}/population_traits.pt")
    with open(f"{out_dir}/languages.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)
    with open(f"{out_dir}/population_labels.json", "w", encoding="utf-8") as f:
        json.dump(row_names, f)

    # PCA metadata
    with open(f"{out_dir}/pca_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_components": n_components,
            "var_explained_target": args.var_explained,
            "var_explained_actual": float(cum_var[-1]),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "subsets_pca": args.subsets_pca,
            "subsets_infer": args.subsets_infer,
        }, f, indent=2)

    with open(f"{out_dir}/population_delta.json", "w", encoding="utf-8") as f:
        json.dump({
            "delta_population": d_pop,
            "delta_language_means": d_mean,
            "n_rows": int(P.shape[0]),
            "n_traits": n_components,
            "max_per_lang": args.max_per_lang,
            "recordings_per_language": per_lang,
        }, f, indent=2)

    # XML fragments
    # 1. Alignment data
    mean_np = mean_active.numpy()
    lines = ['  <data id="contData" spec="Alignment">']
    ambiguities = " ".join("{" + f"{v:.6g}" + "}" for v in mean_np[0])
    lines.append(f'    <userDataType spec="contraband.app.beauti.ContinuousData" nrOfStates="{n_components}" ambiguities="{ambiguities}"/>')
    for i, name in enumerate(names):
        vals = " ".join("{" + f"{v:.6g}" + "}" for v in mean_np[i])
        lines.append(f'    <sequence id="seq_{name}" taxon="{name}" totalcount="{n_components}" value="{vals}"/>')
    lines.append("  </data>")
    with open(f"{out_dir}/alignment.xml", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 2. Population traits
    keys = " ".join(f"rec_{i}" for i in range(P.shape[0]))
    vals = " ".join(f"{v:.6g}" for v in Pnp.ravel())
    pop_xml = (
        f'          <populationTraits id="popTraits" spec="parameter.RealParameter"\n'
        f'              minordimension="{n_components}" keys="{keys}">\n'
        f"            {vals}\n"
        f"          </populationTraits>\n"
    )
    with open(f"{out_dir}/population_traits.xml", "w", encoding="utf-8") as f:
        f.write(pop_xml)

    print(f"\nOutputs in {out_dir}")
    print(f"  n_components={n_components}, delta={d_pop:.6f}")
    print(f"  To build XML: swap <data> block from alignment.xml,")
    print(f"  swap <populationTraits> from population_traits.xml,")
    print(f'  set nrOfStates="{n_components}" delta="{d_pop:.6f}"')


if __name__ == "__main__":
    main()
