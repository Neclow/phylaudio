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
    else:
        raise ValueError(f"Unknown decomposition method: {method}")

    return decomposer


def fit_decomposer(X, method, n_components, device, seed):
    decomposer = load_decomposer(method, device)(
        n_components=n_components, random_state=seed
    )

    decomposer.mean = X.mean(dim=0, keepdim=True)
    decomposer.std = X.std(
        dim=0,
        keepdim=True,
        unbiased=True,
    )

    decomposer.fit(X.sub(decomposer.mean).div(decomposer.std).to(decomposer.device))

    return decomposer


def decompose(decomposer, X):
    return torch.as_tensor(
        decomposer.transform(
            X.sub(decomposer.mean).div(decomposer.std).to(decomposer.device)
        ),
        dtype=X.dtype,
        device=X.device,
    )
