# pylint: disable=invalid-name

"""Metrics for pairwise distances and batch distance processing"""

from itertools import combinations

import numba as nb
import torch
import torch.nn.functional as F
from tqdm import tqdm

from extern.pysdtw.soft_dtw_cuda import SoftDTW

from ...utils import batched

MAX_THREADS_PER_BLOCK = 512


def _paired_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise cosine similarity matrix between x1 and x2.
    """
    if x1.ndim == 2:  # assuming x1 and x2 have the same shape
        # pylint: disable=not-callable
        sim = F.cosine_similarity(x1, x2, dim=-1)
        # pylint: enable=not-callable
    else:
        # Adapted from https://discuss.pytorch.org/t/pairwise-cosine-distance/30961/11
        sim = torch.bmm(F.normalize(x1, p=2, dim=-1), F.normalize(x2, p=2, dim=-1).mT)
    return sim


def paired_cosine(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise cosine distance matrix between x1 and x2.
    """
    sim = _paired_cosine_similarity(x1, x2)
    return -sim + 1


def paired_angular(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes the pairwise angular distance matrix between x1 and x2."""
    sim = _paired_cosine_similarity(x1, x2)
    return torch.acos(sim) / torch.pi


def paired_normalized_euclidean(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes the pairwise normalized euclidean distance matrix between x1 and x2."""
    return torch.sqrt(2 * paired_cosine(x1, x2))


def _pairwise_p_norm(x1: torch.Tensor, x2: torch.Tensor, p: int) -> torch.Tensor:
    """Pairwise p-norm distance."""
    # pylint: disable=not-callable
    return F.pairwise_distance(x1, x2, p=p)
    # pylint: enable=not-callable


def paired_manhattan(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes the pairwise Manhattan distance matrix between x1 and x2."""
    return _pairwise_p_norm(x1, x2, p=1)


def paired_euclidean(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes the pairwise euclidean distance matrix between x1 and x2."""
    return _pairwise_p_norm(x1, x2, p=2)


def paired_sqeuclidean(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise squared euclidean distance matrix between x1 and x2.
    """
    if x1.ndim == 2:
        return paired_euclidean(x1, x2) ** 2

    # Computes the pairwise distance matrix between x1 and x2 using the quadratic expansion.
    # This limits the memory cost to the detriment of compute accuracy.
    # Copyright (c) 2020 Mehran Maghoumi
    # Copyright (c) 2022 Antoine Loriette
    # From: https://github.com/toinsson/pysdtw/blob/main/pysdtw/distance.py
    x1_norm = (x1**2).sum(-1).unsqueeze(-1)
    x2_norm = (x2**2).sum(-1).unsqueeze(-2)
    dist = x1_norm + x2_norm - 2.0 * torch.bmm(x1, x2.mT)
    return torch.clamp(dist, 0.0, torch.inf)


PAIRED_DISTANCE_FUNCTIONS = {
    "angular": paired_angular,
    "cosine": paired_cosine,
    "manhattan": paired_manhattan,
    "euclidean": paired_euclidean,
    "sqeuclidean": paired_sqeuclidean,
    "neuclidean": paired_normalized_euclidean,
    "euclidean_normalized": paired_normalized_euclidean,
    "l2": paired_euclidean,
    "l2_squared": paired_sqeuclidean,
    "l2_norm": paired_normalized_euclidean,
}


def load_distance(metric, soft_dtw=False, device="cpu"):
    """Load a distance function based on the metric.

    Parameters
    ----------
    metric : str
        Metric to use for distance computation
    soft_dtw : bool, optional
        Whether to use soft-DTW or not (for 3D embeddings), by default False
    device : str, optional
        Device for Soft-DTW, by default "cpu"

    Returns
    -------
    func : Union[Callable, SoftDTW]
        Distance function
    """
    if metric in PAIRED_DISTANCE_FUNCTIONS:
        func = PAIRED_DISTANCE_FUNCTIONS[metric]
        if soft_dtw:
            use_cuda = "cuda" in device
            nb.cuda.select_device(int(device.split(":")[-1]))
            return SoftDTW(
                use_cuda=use_cuda,
                normalize=True,
                dist_func=func,
            )
        return func

    raise ValueError(
        f"`metric` must be in {PAIRED_DISTANCE_FUNCTIONS.keys()}. Got {metric} instead."
    )


def pairwise_batched(
    X,
    y,
    metric,
    num_classes,
    batch_size=64,
    device="cpu",
    soft_dtw=False,
):
    """Get a pairwise distance matrix in batched fashion.

    Parameters
    ----------
    X : torch.Tensor
        Input data
    y : torch.Tensor
        Input labels
    metric : str
        Metric to use for distance computation
    num_classes : int
        Number of classes
    batch_size : int, optional
        Batch size, by default 64
    device : str, optional
        Device for Soft-DTW, by default "cpu"
    soft_dtw : bool, optional
        Whether to use soft-DTW or not (for 3D embeddings), by default False

    Returns
    -------
    distance_matrix : torch.Tensor
        (Un-normalized) distance matrix
    count_matrix : torch.Tensor
        Occurrence matrix
    """
    dist_fn = load_distance(metric, soft_dtw=soft_dtw, device=device)

    if soft_dtw and X.ndim == 3:
        # embedding: n_sentences x n_chunks x emb_dim
        # NOTE: Need to limit the chunk dimension for pysdtw
        # NOTE: 1024 is too slow, so setting it to MAX_THREADS_PER_BLOCK
        X = X[:, :MAX_THREADS_PER_BLOCK, :]

    distance_matrix = torch.zeros(
        (num_classes, num_classes), device="cpu", dtype=X.dtype
    )

    count_matrix = torch.zeros_like(distance_matrix, device="cpu", dtype=torch.int64)

    # Sample loop
    for idxs in tqdm(
        batched(combinations(range(X.shape[0]), 2), batch_size), leave=False
    ):
        j, k = map(list, zip(*idxs))

        # NOTE: this computes the distances between
        # each index of j with each index of k
        # dist[0] = dist_fn(X[j, 0], X[k, 0])
        # dist[1] = dist_fn(X[j, 1], X[k, 1]) etc.
        dist = dist_fn(X[j], X[k]).cpu().squeeze()

        distance_matrix[y[j], y[k]] += dist
        count_matrix[y[j], y[k]] += 1

        # Symmetrise
        distance_matrix[y[k], y[j]] = distance_matrix[y[j], y[k]]
        count_matrix[y[k], y[j]] = count_matrix[y[j], y[k]]

    return distance_matrix, count_matrix
