"""Embedding extraction and transformation for phylogenetic inference.

Modules:
    _decomposition   — Dimensionality reduction: PCA, ICA, kernel PCA.
    _discretization  — Embedding-to-character discretization: STE, step, quantile, k-means.
    _distance        — Pairwise distance metrics: euclidean, cosine, angular, manhattan,
                       normalized euclidean, squared euclidean; batched computation.
    base             — Per-sentence loop, model/dataset preparation, classifier loading,
                       decomposer fitting; FleursParallelInput dataclass.
"""
