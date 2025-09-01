import numpy as np
import torch
from sklearn.preprocessing import KBinsDiscretizer

from ..._config import RANDOM_STATE


def _qcut(x, q):
    assert q > 0
    qs = torch.quantile(x, torch.linspace(0, 1, q + 1).to(x.device))

    return torch.clamp(torch.bucketize(x, qs) - 1, 0)


qcut = torch.vmap(_qcut, in_dims=(1, None), out_dims=1)


def kmeans_bucketize(x, q):
    discretizer = KBinsDiscretizer(
        n_bins=q, encode="ordinal", strategy="kmeans", random_state=RANDOM_STATE
    )

    return torch.as_tensor(
        discretizer.fit_transform(x.cpu().numpy()),
        dtype=torch.int64,
        device=x.device,
    )


DISCRETIZATION_METHODS = {
    "kmeans": kmeans_bucketize,
    "quantile": qcut,
    "step": lambda x, _: x.gt(0).long(),
}


def load_discretizer(method):
    if method in DISCRETIZATION_METHODS:
        return DISCRETIZATION_METHODS[method]

    raise ValueError(
        f"`method` must be in `{DISCRETIZATION_METHODS.keys()}`. Got {method} instead."
    )


def discretize(x, method, idxs=None, q=None) -> np.ndarray:
    discretization_fn = DISCRETIZATION_METHODS[method]

    output = torch.zeros_like(x, device=x.device, dtype=torch.int64)

    if idxs is None:
        idxs = torch.arange(x.shape[0])

    output[idxs, :] = discretization_fn(x[idxs, :], q)

    return output
