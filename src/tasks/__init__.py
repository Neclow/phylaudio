"""Pipeline task logic.

Subpackages:
    feature_extraction/       — Embedding extraction, discretization (STE, step,
                                quantile, k-means), decomposition (PCA, ICA), and
                                the per-sentence loop that drives tree inference.
    language_identification/  — MLP classifier with optional STE binarization,
                                Lightning training loop, wandb/CSV logging.
    phylo/                    — Tree building (IQ-TREE, FastME, RAxML, parsimony),
                                FASTA/Nexus I/O, BEAST XML utilities, tree metrics
                                (RF, quartet similarity), SplitsTree delta extraction.

Modules:
    common  — Shared CLI argument parsing, model/dataset preparation.
    plot    — Matplotlib axis formatting helpers.
"""
