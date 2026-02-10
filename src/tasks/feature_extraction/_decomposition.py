# pylint: disable=invalid-name

import torch
from sklearn.decomposition import PCA as sPCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA as kPCA
from torch_pca import PCA as tPCA


def load_decomposer(method, device):
    print("(decomposition) ", end="")
    if method == "pca":
        if torch.device(device).type == "cuda":
            print("Using torch PCA")
            decomposer = tPCA
            decomposer.device = device
        else:
            print("Using sklearn PCA")
            decomposer = sPCA
            decomposer.device = "cpu"
    elif method == "kpca":
        print("Using sklearn Kernel PCA")
        decomposer = kPCA
        decomposer.device = "cpu"
    elif method == "ica":
        print("Using sklearn FastICA")
        decomposer = FastICA
        decomposer.device = "cpu"
    # elif method == "lda":
    #     print("Using sklearn LDA")
    #     decomposer = LDA
    #     decomposer.device = "cpu"
    # elif method == "umap":
    #     from functools import partial
    #     try:
    #         from cuml import UMAP
    #     except ImportError as exc:
    #         raise RuntimeError(
    #             "cuML UMAP is not installed. Are you using the viz environment? Please install cuML or use a different decomposition method."
    #         ) from exc

    #     print("Using cuML UMAP")
    #     decomposer = UMAP
    #     decomposer.device = "cuda"
    #     decomposer = partial(
    #         decomposer, n_neighbors=200, min_dist=0.3, metric="manhattan"
    #     )
    else:
        raise ValueError(f"Unknown decomposition method: {method}")

    return decomposer


def fit_decomposer(X, method, n_components, standardize, device, seed=None):
    decomposer = load_decomposer(method, device)(
        n_components=n_components, random_state=seed
    )

    if standardize:
        decomposer.mean = X.mean(dim=0, keepdim=True)
        decomposer.std = X.std(
            dim=0,
            keepdim=True,
            unbiased=True,
        )

        Z = X.sub(decomposer.mean).div(decomposer.std).to(decomposer.device)
    else:
        decomposer.mean = None
        decomposer.std = None
        Z = X.to(decomposer.device)

    decomposer.fit(Z)

    return decomposer


def decompose(decomposer, X):
    if decomposer.mean is not None and decomposer.std is not None:
        Z = X.sub(decomposer.mean).div(decomposer.std).to(decomposer.device)
    else:
        Z = X.to(decomposer.device)

    return torch.as_tensor(
        decomposer.transform(Z),
        dtype=X.dtype,
        device=X.device,
    )
