# pylint: disable=invalid-name

"""
Maximum likelihood-based phylogenetic tree construction
from the FLEURS dataset after embedding discretization
"""

import numpy as np
import torch
import torch.nn.functional as F

from src._config import DEFAULT_THREADS_NEXUS
from src.tasks.feature_extraction._discretization import (
    DISCRETIZATION_METHODS,
    discretize,
)
from src.tasks.feature_extraction.base import (
    get_fleurs_parallel_args,
    prepare_everything,
    save_state,
    sentence_loop,
)
from src.tasks.phylo.nexus import DiscretePhyloWriter


def parse_args(with_base_args=True):
    """Parse arguments for per-sentence, discrete-featured alignment construction"""

    parser = get_fleurs_parallel_args(with_common_args=with_base_args)

    parser.add_argument(
        "--discretization",
        default="step",
        type=str,
        choices=tuple(DISCRETIZATION_METHODS.keys()),
    )

    parser.add_argument(
        "-q",
        "--q",
        type=int,
        help="Number of discrete states",
    )

    parser.add_argument(
        "--method",
        choices=("iqtree2", "raxml", "pratchet"),
        default="iqtree2",
        help="Phylogenetic inference tool",
    )

    parser.add_argument(
        "--iqtree-model",
        type=str,
        default="MFP",
        help="IQTree substitution model",
    )

    parser.add_argument(
        "--iqtree-bootstrap",
        action="store_true",
        help="If True, run IQTree with bootstrapping",
    )

    parser.add_argument(
        "-nt",
        "--n-threads",
        type=int,
        default=DEFAULT_THREADS_NEXUS,
        help="Number of threads to use for parallel processing of iqtree",
    )

    return parser.parse_args()


def groupby_and_agg(X, y, num_classes, agg="mean"):
    if agg not in ("mean", "sum"):
        raise AttributeError(f"Unknown value for agg: `{agg}`")
    # https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/3
    M = torch.zeros(num_classes, X.shape[0])
    M[y, torch.arange(X.shape[0])] = 1

    # Return type requires .float(): https://github.com/pytorch/pytorch/issues/103054
    if agg == "mean":
        M = F.normalize(M, p=1, dim=1)

    return torch.mm(M.to(dtype=X.dtype, device=X.device), X).float()


def embedding_to_fasta(X_emb, y, sentence_index, args, inputs, output_folder):
    grouped_embeddings = groupby_and_agg(X_emb, y, inputs.num_classes, agg="mean")

    # Step 3: Discretise
    discretised_embeddings = (
        discretize(
            grouped_embeddings, idxs=y.unique(), method=args.discretization, q=args.q
        )
        .cpu()
        .numpy()
    )

    # Write to fasta
    output_file = f"{output_folder}/{sentence_index}_{-1}.fa"

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(discretised_embeddings.shape[0]):
            emb_i = discretised_embeddings[i]
            if np.array_equal(emb_i, np.zeros_like(emb_i)):
                continue
            emb_str = np.array2string(
                emb_i,
                separator="",
                threshold=emb_i.shape[0] + 1,
                max_line_width=emb_i.shape[0] + 2,
            )[1:-1]
            f.write(f">{inputs.labels[i]}\n{emb_str}\n")


def main():
    """Main loop"""

    # Parse arguments
    args = parse_args(with_base_args=True)

    # Prepare inputs
    inputs = prepare_everything(args)

    # Save metadata
    output_folder = f"data/trees/per_sentence/discrete/{inputs.run_id}"
    save_state(inputs, output_folder)

    # Feature extraction loop (applied sentence-wise)
    print("Entering sentence loop...")
    sentence_loop(
        args=args,
        inputs=inputs,
        output_folder=output_folder,
        downstream_func=embedding_to_fasta,
    )

    # Tree inference
    writer = DiscretePhyloWriter(
        inputs.run_id,
        iqtree_method=args.iqtree_model,
        iqtree_bootstrap=args.iqtree_bootstrap,
    )
    writer.write(export_trees=True)

    print(f"View results at {output_folder}")


if __name__ == "__main__":
    main()
