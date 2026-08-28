# pylint: disable=invalid-name

"""Distance-then-mean variant of continuous_embeddings.py.

Instead of computing the distance matrix on per-language mean embeddings,
computes a pairwise distance matrix for each sentence (parallel across
languages), then averages those per-sentence matrices.
"""

import json
import os

import pandas as pd
import torch
from tqdm import tqdm

from src._config import DEFAULT_CONTINUOUS_DIR
from src.tasks.feature_extraction.base import (
    get_fleurs_parallel_args,
    post_process_embeddings,
    prepare_everything,
    save_state,
)
from src.tasks.phylo.tree import make_distance_tree

from pipeline.continuous_embeddings import (
    _load_language_mapping,
    _resolve_from_wandb,
    compute_distance_matrix,
    save_phylip,
)


def parse_args():
    parser = get_fleurs_parallel_args(with_common_args=True)

    parser.add_argument("--dtype", required=True)
    parser.add_argument(
        "--metric",
        default="euclidean",
        type=str,
        choices=("euclidean", "sqeuclidean", "mse", "cosine", "angular"),
    )
    parser.add_argument(
        "--method",
        default="fastme",
        type=str,
        choices=("fastme", "nj", "upgma", "none"),
    )
    parser.add_argument("--subsets", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.ckpt is not None and not os.path.isfile(args.ckpt):
        model_id, cache = _resolve_from_wandb(args.ckpt)
        if model_id is not None:
            args.model_id = model_id
        if cache is not None and args.embeddings_cache is None:
            args.embeddings_cache = cache
        print(f"Resolved from W&B: model_id={model_id}, cache={cache}")

    if args.subsets is not None and args.embeddings_cache is None:
        raise ValueError("--subsets requires --embeddings-cache")

    ckpt_id = None
    if args.ckpt is not None and not os.path.isfile(args.ckpt):
        ckpt_id = args.ckpt

    inputs = prepare_everything(args)

    # --- Load, filter subsets, project (same as _aggregate_cached) ---
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

    # --- Accumulate per-sentence distance matrices ---
    distance_matrix = torch.zeros(
        (inputs.num_classes, inputs.num_classes), device="cpu", dtype=X_emb.dtype
    )
    count_matrix = torch.zeros_like(distance_matrix, dtype=torch.int64)

    for _, grp in tqdm(meta.groupby("sentence_index"), desc="Sentences"):
        idx = grp.index.to_numpy()
        X_sent = X_emb[idx]
        y_sent = y[idx].cpu()

        dm_sent = compute_distance_matrix(X_sent, metric=args.metric).cpu()

        distance_matrix[y_sent.unsqueeze(1), y_sent.unsqueeze(0)] += dm_sent
        count_matrix[y_sent.unsqueeze(1), y_sent.unsqueeze(0)] += 1

    # Average
    valid = count_matrix > 0
    dm = torch.zeros_like(distance_matrix)
    dm[valid] = distance_matrix[valid] / count_matrix[valid].float()

    # --- Filter to active languages ---
    active_mask = dm.abs().sum(dim=1) > 0
    active_indices = torch.where(active_mask)[0]
    active_dir_codes = [inputs.labels[i] for i in active_indices.tolist()]
    dm_active = dm[active_mask][:, active_mask]

    label_map = _load_language_mapping(args.dataset)
    active_labels = [label_map.get(c, c) for c in active_dir_codes]

    # --- Save ---
    dtype = args.dtype
    continuous_dir = f"{DEFAULT_CONTINUOUS_DIR}/{dtype}"
    os.makedirs(continuous_dir, exist_ok=True)
    run_id = ckpt_id or inputs.run_id
    output_folder = f"{continuous_dir}/{run_id}"
    save_state(inputs, output_folder)

    with open(f"{output_folder}/languages.json", "w", encoding="utf-8") as f:
        json.dump(active_labels, f, indent=2)

    torch.save(dm_active, f"{output_folder}/distance_matrix.pt")
    save_phylip(dm_active, active_labels, f"{output_folder}/distance_matrix.phy")

    if args.method != "none":
        dm_df = pd.DataFrame(
            dm_active.cpu().numpy(),
            index=active_labels,
            columns=active_labels,
        )
        newick = make_distance_tree(dm_df, method=args.method)
        with open(f"{output_folder}/tree.nwk", "w", encoding="utf-8") as f:
            f.write(newick)

    print(f"Done. Results at {output_folder}")


if __name__ == "__main__":
    main()
