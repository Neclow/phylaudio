# pylint: disable=invalid-name

"""Distance-based phylogenetic tree construction from the FLEURS dataset"""

import torch

from src.tasks.feature_extraction._distance import pairwise_batched
from src.tasks.feature_extraction.base import (
    get_fleurs_parallel_args,
    prepare_everything,
    save_state,
    sentence_loop,
)
from src.tasks.phylo.nexus import DistancePhyloWriter


def parse_args(with_base_args=True):
    """Parse arguments for per-sentence pairwise distance calculation"""

    parser = get_fleurs_parallel_args(with_common_args=with_base_args)

    parser.add_argument(
        "--dbs",
        type=int,
        default=64,
        help="Pairwise distance batch size",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        type=str,
        help="Distance metric",
    )
    parser.add_argument("--soft-dtw", action="store_true", help="Apply Soft-DTW")
    parser.add_argument(
        "--method",
        default="fastme",
        type=str,
        help="Distance-based phylogenetic method",
    )
    parser.add_argument(
        "--layer",
        default=-1,
        type=int,
        help="Layer to use for distance calculation",
    )

    return parser.parse_args()


def embedding_to_pdist(X_emb, y, sentence_index, args, inputs, output_folder):
    distance_matrix, count_matrix = pairwise_batched(
        X=X_emb,
        y=y.cpu(),
        metric=args.metric,
        num_classes=inputs.num_classes,
        batch_size=args.dbs,
        soft_dtw=args.soft_dtw,
        device=args.device,
    )

    # Save distance_matrix and count_matrix for the last layer (-1)
    sentence_data = {
        args.layer: {
            "count_matrix": count_matrix,
            "distance_matrix": distance_matrix,
        }
    }

    # Save data
    sentence_file = f"{output_folder}/{sentence_index}.pt"
    torch.save(sentence_data, sentence_file)


def main():
    """Main loop"""

    # Parse arguments
    args = parse_args(with_base_args=True)

    # Prepare inputs
    inputs = prepare_everything(args)

    # Save metadata
    output_folder = f"data/pdist/{inputs.run_id}"
    save_state(inputs, output_folder)

    # Feature extraction loop (applied sentence-wise)
    print("Entering sentence loop..")
    sentence_loop(
        args,
        inputs,
        output_folder,
        downstream_func=embedding_to_pdist,
    )

    # Tree inference
    writer = DistancePhyloWriter(run_id=inputs.run_id, layer=args.layer)
    writer.write(export_trees=True)

    print(f"View results at {output_folder}")


if __name__ == "__main__":
    main()
