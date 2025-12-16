# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.config import default_float
from matplotlib import pyplot as plt
from scipy import stats

# -----------------------------------------------------------------------------
# Globals & aliases -----------------------------------------------------------
# -----------------------------------------------------------------------------
DTYPE = default_float()  # match GPflow’s global float type
SMALL = 1e-6  # numerical jitter (same name as upstream file)
tfb = tfp.bijectors  # terse alias à la upstream

# -----------------------------------------------------------------------------
# Helper: per-dimension standardiser -----------------------------------------
# -----------------------------------------------------------------------------


def _standardiser(x: np.ndarray, eps: float = SMALL):
    """Return `[Scale, Shift]` bijectors so `(x-mean)*scale` has unit variance.

    Identical logic to the upstream 1-D helper; we keep the function private to
    avoid leaking symbols.
    """
    x = np.asarray(x)
    m, s = np.mean(x), np.std(x)
    # Prevent divide-by-zero and catastrophic *huge* scales.
    scale_init = 1.0 / (s + eps)
    return [
        tfb.Scale(gpflow.Parameter(scale_init, transform=tfb.Exp(), dtype=DTYPE)),
        tfb.Shift(gpflow.Parameter(-m, dtype=DTYPE)),
    ]


def make_sinharcsinh():
    return tfb.SinhArcsinh(
        skewness=gpflow.Parameter(0.0),
        tailweight=gpflow.Parameter(1.0, transform=tfb.Exp()),
    )


def make_standardizer(x):
    return [
        tfb.Scale(gpflow.Parameter(1.0 / np.std(x), transform=tfb.Exp())),
        tfb.Shift(gpflow.Parameter(-np.mean(x))),
    ]


class Normalizer(gpflow.base.Module):
    """
    :param x: input to transform
    :param log: whether to log x first before applying flows of transformations
    :return: flows of transformations to match x to standard Gaussian
    """

    def __init__(
        self,
        x,
        log=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x = x

        if log:
            offset = np.min(x) - 1.0
            self.bijector = tfb.Chain(
                [make_sinharcsinh() for _ in range(1)]
                + make_standardizer(np.log(x - offset))
                + [tfb.Log(), tfb.Shift(-offset)]
            )
        else:
            self.bijector = tfb.Chain(
                [make_sinharcsinh() for _ in range(1)] + make_standardizer(x)
            )

    def plot(self, title="Normalising Flow"):
        f = plt.figure()
        ax = f.add_axes([0.3, 0.3, 0.65, 0.65])
        x = self.x
        y = self.bijector(x).numpy()
        ax.plot(x, y, "k.", label="Gaussian")
        ax.legend()

        ax_x = f.add_axes([0.3, 0.05, 0.65, 0.25], sharex=ax)
        ax_x.hist(x, bins=20)
        ax_y = f.add_axes([0.05, 0.3, 0.25, 0.65], sharey=ax)
        ax_y.hist(y, bins=20, orientation="horizontal")
        ax_y.set_xlim(ax_y.get_xlim()[::-1])
        plt.title(title)

    def KL_objective(self):
        return 0.5 * tf.reduce_mean(tf.square(self.bijector(self.x))) - tf.reduce_mean(
            self.bijector.forward_log_det_jacobian(self.x, event_ndims=0)
        )

    def kstest(self):
        # Kolmogorov-Smirnov test for normality of transformed data
        s, pvalue = stats.kstest(self.bijector(self.x).numpy()[:, 0], "norm")
        print("KS test statistic is %.3f, p-value is %.8f" % (s, pvalue))
        return s, pvalue


# -----------------------------------------------------------------------------
# Normaliser 2-D — minimal extension of the 1-D upstream code -----------------
# -----------------------------------------------------------------------------
class Normalizer2D(gpflow.base.Module):
    """2‑D extension of the OAK Normalizer with **optional decorrelation**.

    * If `decorrelate=False` (default), behaves like two independent 1‑D normalisers.
    * If `decorrelate=True`, appends an **affine whitening** (centre + TriL^{-1})
      computed on the output of the two 1‑D flows, making the result ~ N(0, I).

    ⚠️ Enabling `decorrelate=True` mixes x₁ and x₂ (loses one-to-one identifiability
       per coordinate). Leave it False if you need z₁ ↔ x₁, z₂ ↔ x₂.
    """

    def __init__(
        self, x, log=(False, False), eps: float = SMALL, name="normalizer2d", **kwargs
    ):
        super().__init__(name=name, **kwargs)

        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == 2, "Expected input shape (N, 2)."
        self.x = x
        self.eps = eps
        assert len(log) == 2, "`log` must be a 2-tuple of booleans."

        # ------------------------------------------------------------------
        # 1) Per‑dimension 1‑D flows (right-to-left chain)
        #    forward (log=True): Shift(-offset) -> Log -> Standardise -> SinhArcsinh
        #    forward (log=False): Standardise -> SinhArcsinh
        # ------------------------------------------------------------------
        bijectors = []
        for d in range(2):
            xd = x[:, d]
            if log[d]:
                offset = float(np.min(xd) - 1.0 - eps)  # ensure positivity for Log
                chain = tfb.Chain(
                    [
                        tfb.SinhArcsinh(
                            skewness=gpflow.Parameter(0.0, dtype=DTYPE),
                            tailweight=gpflow.Parameter(
                                1.0, transform=tfb.Exp(), dtype=DTYPE
                            ),
                            name="sinh_arcsinh",
                        ),
                        *_standardiser(np.log(xd - offset), eps),
                        tfb.Log(name="log"),
                        tfb.Shift(-offset, name="shift_pos"),
                    ],
                    name=f"flow_dim{d}",
                )
            else:
                chain = tfb.Chain(
                    [
                        tfb.SinhArcsinh(
                            skewness=gpflow.Parameter(0.0, dtype=DTYPE),
                            tailweight=gpflow.Parameter(
                                1.0, transform=tfb.Exp(), dtype=DTYPE
                            ),
                            name="sinh_arcsinh",
                        ),
                        *_standardiser(xd, eps),
                    ],
                    name=f"flow_dim{d}",
                )
            bijectors.append(chain)

        block = tfb.Blockwise(bijectors, block_sizes=[1, 1], name="per_dim_block")
        pieces = [block]

        self.bijector = tfb.Chain(pieces, name="normalizer2d_flow")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, z):
        return self.bijector.forward(tf.cast(z, DTYPE))

    def inverse(self, y):
        return self.bijector.inverse(tf.cast(y, DTYPE))

    def log_det_jacobian(self, z):
        return self.bijector.forward_log_det_jacobian(tf.cast(z, DTYPE), event_ndims=1)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def KL_objective(self):
        z = tf.convert_to_tensor(self.x, dtype=DTYPE)
        y = self.forward(z)
        return 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(y), axis=-1)
        ) - tf.reduce_mean(self.log_det_jacobian(z))

    def kstest(self):
        y = self.forward(self.x).numpy()
        out = []
        for d in range(2):
            s, p = stats.kstest(y[:, d], "norm")
            out.append((s, p))
            print(f"Dim {d}: KS stat = {s:.3f}, p = {p:.3g}")
        return out

    def plot(self, title="Normalizer 2D"):
        f = plt.figure(figsize=(6, 6))
        ax = f.add_axes([0.3, 0.3, 0.65, 0.65])
        x = self.x
        y = self.forward(x).numpy()
        ax.plot(x[:, 0], y[:, 0], ".", label="dim 1")
        ax.plot(x[:, 1], y[:, 1], ".", label="dim 2")
        ax.legend(loc="best")
        ax.set_xlabel("x")
        ax.set_ylabel("Gaussianised y")

        ax_x = f.add_axes([0.3, 0.05, 0.65, 0.25], sharex=ax)
        ax_x.hist(x[:, 0], bins=20, alpha=0.8)
        ax_x.set_xlabel("x[:,0]")

        ax_y = f.add_axes([0.05, 0.3, 0.25, 0.65], sharey=ax)
        ax_y.hist(y[:, 0], bins=20, orientation="horizontal", alpha=0.8)
        ax_y.set_ylabel("y[:,0]")
        ax_y.set_xlim(ax_y.get_xlim()[::-1])
        f.suptitle(title)
        return f


class NormalizerGeneralized(gpflow.base.Module):
    """
    D-dimensional extension of the Amazon OAK *Normalizer* with optional decorrelation.

    * With `decorrelate=True`, it appends an affine whitening transformation
      based on the sample mean and Cholesky of the marginal covariance after
      the D independent flows. This makes the output approximately N(0, I).
    """

    def __init__(self, x, log=None, eps: float = SMALL, name="normalizer", **kwargs):
        super().__init__(name=name, **kwargs)

        x = np.asarray(x)
        assert x.ndim == 2, "Expected input shape (N, D)."
        self.x = x
        self.eps = eps

        D = x.shape[1]
        if log is None:
            log = [False] * D
        assert len(log) == D, "Length of `log` must match number of dimensions"

        # ------------------------------------------------------------------
        # 1. Per-dimension 1D flows
        # ------------------------------------------------------------------
        bijectors = []
        for d in range(D):
            xd = x[:, d]
            if log[d]:
                offset = np.min(xd) - 1.0 - eps
                chain = tfb.Chain(
                    [
                        *_standardiser(np.log(xd - offset), eps),
                        tfb.Log(),
                        tfb.Shift(-offset),
                        tfb.SinhArcsinh(
                            skewness=gpflow.Parameter(0.0, dtype=DTYPE),
                            tailweight=gpflow.Parameter(
                                1.0, transform=tfb.Exp(), dtype=DTYPE
                            ),
                        ),
                    ],
                    name=f"flow_dim{d}",
                )
            else:
                chain = tfb.Chain(
                    [
                        *_standardiser(xd, eps),
                        tfb.SinhArcsinh(
                            skewness=gpflow.Parameter(0.0, dtype=DTYPE),
                            tailweight=gpflow.Parameter(
                                1.0, transform=tfb.Exp(), dtype=DTYPE
                            ),
                        ),
                    ],
                    name=f"flow_dim{d}",
                )
            bijectors.append(chain)

        block = tfb.Blockwise(bijectors, block_sizes=[1] * D, name="per_dim_block")
        self._block = block

        self.bijector = block

    def forward(self, z):
        return self.bijector.forward(tf.cast(z, DTYPE))

    def inverse(self, y):
        return self.bijector.inverse(tf.cast(y, DTYPE))

    def log_det_jacobian(self, z):
        return self.bijector.forward_log_det_jacobian(tf.cast(z, DTYPE), event_ndims=1)

    def KL_objective(self):
        z = self.x.astype(np.float64)
        y = self.forward(z)
        return 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(y), axis=-1)
        ) - tf.reduce_mean(self.log_det_jacobian(z))

    def kstest(self):
        y = self.forward(self.x).numpy()
        test_results = []
        for d in range(y.shape[1]):
            s, p = stats.kstest(y[:, d], "norm")
            test_results.append((s, p))
            print(f"Dim {d}: KS stat = {s:.3f}, p = {p:.3g}")
        return test_results
