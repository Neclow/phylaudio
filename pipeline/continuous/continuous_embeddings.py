# pylint: disable=invalid-name

"""Aggregate-then-infer phylogenetics for continuous (GELU) embeddings.

Computes per-language mean embeddings across all sentences, saves the
embedding matrix and pairwise distance matrix, and optionally infers a
distance-based tree (FastME / NJ / UPGMA).
"""

import json
import math
import os
from glob import glob

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src._config import (
    DEFAULT_CONTINUOUS_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_METADATA_DIR,
    DEFAULT_METADATA_KEY,
)
from src.tasks.feature_extraction.base import (
    get_fleurs_parallel_args,
    post_process_embeddings,
    prepare_everything,
    save_state,
    sentence_loop,
)
from src.tasks.phylo.tree import make_distance_tree

from pipeline.sentence_trees_discrete import groupby_and_agg


def parse_args():
    parser = get_fleurs_parallel_args(with_common_args=True)

    parser.add_argument(
        "--dtype",
        required=True,
        help="Output subdirectory name under continuous/",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        type=str,
        choices=("euclidean", "sqeuclidean", "mse", "cosine", "angular"),
        help="Distance metric for the pairwise matrix",
    )
    parser.add_argument(
        "--method",
        default="fastme",
        type=str,
        choices=("fastme", "nj", "upgma", "none"),
        help="Distance-based phylogenetic method (none = skip tree inference)",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default=None,
        help="Comma-separated data subsets to include (train,dev,test). Cache only.",
    )

    return parser.parse_args()


def _resolve_from_wandb(run_id):
    """Extract model_id and embeddings-cache from a W&B run's config."""
    configs = sorted(glob(f"{DEFAULT_EVAL_DIR}/wandb/run-*-{run_id}/files/config.yaml"))
    if not configs:
        return None, None
    with open(configs[-1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = None
    try:
        model_id = cfg["model_id"]["value"]
    except (KeyError, TypeError):
        pass

    cache = None
    try:
        for writer in cfg["_wandb"]["value"]["e"].values():
            args_list = writer.get("args", [])
            for i, arg in enumerate(args_list):
                if arg == "--embeddings-cache" and i + 1 < len(args_list):
                    cache = args_list[i + 1]
    except (KeyError, TypeError):
        pass

    return model_id, cache


def _load_language_mapping(dataset):
    """Load dir-code → display-name mapping from metadata."""
    path = f"{DEFAULT_METADATA_DIR}/{dataset}/languages.json"
    with open(path, "r", encoding="utf-8") as f:
        languages = json.load(f)
    return {k: v[DEFAULT_METADATA_KEY] for k, v in languages.items()}


def compute_distance_matrix(mean_embeddings, metric="euclidean"):
    if metric == "euclidean":
        return torch.cdist(mean_embeddings, mean_embeddings, p=2)

    if metric == "sqeuclidean":
        return torch.cdist(mean_embeddings, mean_embeddings, p=2) ** 2

    if metric == "mse":
        p = mean_embeddings.shape[-1]
        return torch.cdist(mean_embeddings, mean_embeddings, p=2) ** 2 / p

    sim = F.cosine_similarity(
        mean_embeddings.unsqueeze(1),
        mean_embeddings.unsqueeze(0),
        dim=-1,
    )
    if metric == "cosine":
        return 1 - sim

    # angular
    return torch.acos(sim.clamp(-1, 1)) / math.pi


def save_phylip(distance_matrix, labels, path):
    n = distance_matrix.shape[0]
    dm = distance_matrix.cpu().numpy()
    dm = (dm + dm.T) / 2
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for i in range(n):
            row_str = "  ".join(f"{dm[i, j]:.6f}" for j in range(n))
            f.write(f"{labels[i]}  {row_str}\n")


def _aggregate_cached(args, inputs):
    embeddings, meta = inputs.embedding_cache

    if args.subsets is not None:
        subsets = set(args.subsets.split(","))
        mask = meta["subset"].isin(subsets).to_numpy()
        embeddings = embeddings[torch.from_numpy(mask)]
        meta = meta[mask].reset_index(drop=True)
        print(f"Filtered to subsets {subsets}: {len(meta)} utterances")

    y = torch.as_tensor(meta["label"].to_numpy(), device=args.device)
    embeddings = embeddings.to(args.device)

    with torch.no_grad():
        X_emb, y = post_process_embeddings(inputs, embeddings, y)

    return X_emb, y


def _aggregate_live(args, inputs, output_folder):
    all_X = []
    all_y = []

    def _accumulate(X_emb, y, sentence_index, args, inputs, output_folder):
        all_X.append(X_emb.cpu())
        all_y.append(y.cpu())

    sentence_loop(args, inputs, output_folder, downstream_func=_accumulate)

    return torch.cat(all_X, dim=0), torch.cat(all_y, dim=0)


def main():
    args = parse_args()

    # Auto-resolve model_id and embeddings-cache from the W&B config
    if args.ckpt is not None and not os.path.isfile(args.ckpt):
        model_id, cache = _resolve_from_wandb(args.ckpt)
        if model_id is not None:
            args.model_id = model_id
        if cache is not None and args.embeddings_cache is None:
            args.embeddings_cache = cache
        print(f"Resolved from W&B: model_id={model_id}, cache={cache}")

    if args.subsets is not None and args.embeddings_cache is None:
        raise ValueError("--subsets requires --embeddings-cache (or a W&B run ID for --ckpt)")

    # Use the ckpt run ID as the output directory name when available
    ckpt_id = None
    if args.ckpt is not None and not os.path.isfile(args.ckpt):
        ckpt_id = args.ckpt

    inputs = prepare_everything(args)

    dtype = args.dtype
    if args.decomposition is not None:
        dtype += f"+{args.decomposition}{args.n_components}"
    continuous_dir = f"{DEFAULT_CONTINUOUS_DIR}/{dtype}"
    os.makedirs(continuous_dir, exist_ok=True)
    run_id = ckpt_id or inputs.run_id
    output_folder = f"{continuous_dir}/{run_id}"
    save_state(inputs, output_folder)

    if inputs.embedding_cache is not None:
        X_emb, y = _aggregate_cached(args, inputs)
    else:
        X_emb, y = _aggregate_live(args, inputs, output_folder)

    mean_embeddings = groupby_and_agg(X_emb, y, inputs.num_classes, agg="mean")

    active_mask = mean_embeddings.abs().sum(dim=1) > 0
    active_indices = torch.where(active_mask)[0]
    active_dir_codes = [inputs.labels[i] for i in active_indices.tolist()]
    mean_active = mean_embeddings[active_mask]

    label_map = _load_language_mapping(args.dataset)
    active_labels = [label_map.get(c, c) for c in active_dir_codes]

    torch.save(mean_active, f"{output_folder}/mean_embeddings.pt")

    with open(f"{output_folder}/languages.json", "w", encoding="utf-8") as f:
        json.dump(active_labels, f, indent=2)

    dm = compute_distance_matrix(mean_active, metric=args.metric)
    torch.save(dm, f"{output_folder}/distance_matrix.pt")
    save_phylip(dm, active_labels, f"{output_folder}/distance_matrix.phy")

    if args.method != "none":
        dm_df = pd.DataFrame(
            dm.cpu().numpy(),
            index=active_labels,
            columns=active_labels,
        )
        newick = make_distance_tree(dm_df, method=args.method)
        with open(f"{output_folder}/tree.nwk", "w", encoding="utf-8") as f:
            f.write(newick)

    print(f"Done. Results at {output_folder}")


if __name__ == "__main__":
    main()
