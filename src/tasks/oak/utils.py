# pylint: disable=invalid-name
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import set_trainable
from gpflow.config import default_float, default_jitter
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.models import GPModel
from sklearn.cluster import KMeans

from .input_measures import EmpiricalMeasure, GaussianMeasure, MOGMeasure
from .oak_kernel import KernelComponenent, bounded_param, get_list_representation
from .ortho_binary_kernel import OrthogonalBinary
from .ortho_categorical_kernel import OrthogonalCategorical
from .ortho_rbf_kernel import OrthogonalRBFKernel

# -

opt = gpflow.optimizers.Scipy()
tfd = tfp.distributions
f64 = gpflow.utilities.to_default_float


def model_to_kernel_list(model: GPModel, selected_dims: List):
    # exact list of kernels from the OAK model
    kernel = []
    model_dims = extract_active_dims(model)
    for i in range(len(selected_dims)):
        for j in range(len(model.kernel.kernels) - 1):
            if model_dims[j] == selected_dims[i]:
                kernel.append(model.kernel.kernels[j])
    # append offset kernel
    kernel.append(model.kernel.kernels[-1])
    return kernel


def extract_active_dims(m):
    # exact list of active dimensions from the OAK model m
    active_dims = []
    for i in range(len(m.kernel.kernels) - 1):
        # interaction with product kernel
        if type(m.kernel.kernels[i]) == gpflow.kernels.base.Product:
            sub_m = m.kernel.kernels[i].kernels
            dims = []
            for j in range(len(sub_m)):
                dim = sub_m[j].active_dims
                dims.append(dim[0])
        else:
            dims = m.kernel.kernels[i].active_dims

        active_dims.append(list(dims))
    return active_dims


def grammer_to_kernel(
    selected_dims,
    offset,
    measure=GaussianMeasure(0, 10),
    lengthscales_lo=1e-3,
    lengthscales_hi=100,
    variance_lo=0.01,
    variance_hi=100,
):
    # construct list of kernels
    # selected_dims: list of kernel indices
    selected_kernels = []
    for i in range(len(selected_dims)):
        # loop through depth
        k_list = []
        for j in range(len(selected_dims[i])):

            lengthscales = np.random.uniform(low=lengthscales_lo, high=lengthscales_hi)
            variance = np.random.uniform(low=variance_lo, high=variance_hi)

            dim = selected_dims[i][j] + offset
            if isinstance(measure, EmpiricalMeasure):
                location = measure.location
                k = OrthogonalRBFKernel(
                    gpflow.kernels.RBF(lengthscales=lengthscales, variance=variance),
                    EmpiricalMeasure(np.reshape(location[:, dim], (-1, 1))),
                    active_dims=[dim],
                )
            else:
                k = OrthogonalRBFKernel(
                    gpflow.kernels.RBF(lengthscales=lengthscales, variance=variance),
                    measure,
                    active_dims=[dim],
                )
            k.base_kernel.lengthscales = bounded_param(
                lengthscales_lo, lengthscales_hi, lengthscales
            )
            k.base_kernel.variance = bounded_param(variance_lo, variance_hi, variance)
            if j > 0:
                k.base_kernel.variance.assign(1)
                set_trainable(k.base_kernel.variance, False)

            k_list.append(k)
        k = np.prod(k_list)
        selected_kernels.append(k)

    # add a constant kernel
    k0 = gpflow.kernels.Constant(variance=10)
    selected_kernels.append(k0)

    return selected_kernels


def f1(x, y, lengthscales, delta, mu):
    # eq (44) in Appendix G.1 of paper for calculating Sobol indices
    return (
        lengthscales
        / np.sqrt(lengthscales**2 + 2 * delta**2)
        * np.exp(-((x - y) ** 2) / (4 * lengthscales**2))
        * np.exp(-((mu - (x + y) / 2) ** 2) / (2 * delta**2 + lengthscales**2))
    )


def _pairwise(a):  # (N,) -> (N,N) broadcast helpers
    return a[:, None], a[None, :]


def f2(x, y, lengthscales, delta, mu):
    # eq (45) in Appendix G.1 of paper for calculating Sobol indices
    M = 1 / (lengthscales**2) + 1 / (lengthscales**2 + delta**2)
    m = 1 / M * (mu / (lengthscales**2 + delta**2) + x / lengthscales**2)
    C = x**2 / (lengthscales**2) + mu**2 / (lengthscales**2 + delta**2) - m**2 * M
    return (
        lengthscales
        * np.sqrt((lengthscales**2 + 2 * delta**2) / (delta**2 * M + 1))
        * np.exp(-C / 2)
        / (lengthscales**2 + delta**2)
        * np.exp(-((y - mu) ** 2) / (2 * (lengthscales**2 + delta**2)))
        * np.exp(-((m - mu) ** 2) / (2 * (1 / M + delta**2)))
    )


def f3(x, y, lengthscales, delta, mu):
    # eq (46) in Appendix G.1 of paper for calculating Sobol indices
    # pylint: disable=arguments-out-of-order
    return f2(y, x, lengthscales, delta, mu)
    # pylint: enable=arguments-out-of-order


def f4(x, y, lengthscales, delta, mu):
    # eq (47) in Appendix G.1 of paper for calculating Sobol indices
    return (
        lengthscales**2
        * (lengthscales**2 + 2 * delta**2)
        * np.sqrt((lengthscales**2 + delta**2) / (lengthscales**2 + 3 * delta**2))
        / ((lengthscales**2 + delta**2) ** 2)
        * np.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * (lengthscales**2 + delta**2)))
    )


def get_model_sufficient_statistics(m, get_L=True):
    """
    Compute a vector "alpha" and a matrix "L" which can be used for easy prediction.
    """

    X_data, Y_data = m.data
    if isinstance(m, gpflow.models.SVGP):
        posterior = m.posterior()
        # details of Qinv can be found https://github.com/GPflow/GPflow/blob/develop/gpflow/posteriors.py
        alpha = posterior.alpha
        if get_L:
            L = tf.linalg.cholesky(tf.linalg.inv(posterior.Qinv[0]))
    elif isinstance(m, gpflow.models.SGPR):

        num_inducing = len(m.inducing_variable)
        err = Y_data - m.mean_function(X_data)
        kuf = Kuf(m.inducing_variable, m.kernel, X_data)
        kuu = Kuu(m.inducing_variable, m.kernel, jitter=default_jitter())

        sigma = tf.sqrt(m.likelihood.variance)
        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
            num_inducing, dtype=default_float()
        )
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma

        tmp1 = tf.linalg.solve(tf.transpose(LB), c)
        alpha = tf.linalg.solve(tf.transpose(L), tmp1)

        if get_L:
            # compute the effective L
            LAi = tf.linalg.triangular_solve(L, np.eye(L.shape[0]))
            LBiLAi = tf.linalg.triangular_solve(LB, LAi)
            L = tf.linalg.inv(LAi - LBiLAi)

    elif isinstance(m, gpflow.models.GPR):
        # prepare for prediction
        K = m.kernel(X_data)
        Ktilde = K + np.eye(X_data.shape[0]) * m.likelihood.variance
        L = np.linalg.cholesky(Ktilde)
        alpha = tf.linalg.cholesky_solve(L, Y_data)

    else:
        raise NotImplementedError
    if get_L:
        return alpha, L
    else:
        return alpha


def _pairwise(a):  # you already have this
    return a[:, None], a[None, :]


def compute_L(X, lengthscales, v, dims, delta, mu):
    """
    Builds L_u = ∫ p(x_u) \tilde k(x_u, X_u) \tilde k(x_u, X_u)^T dx_u
    by factoring into per-dimension 1D closed forms (your f1..f4).
    X: (N,d)
    lengthscales: shared scalar lengthscale (keep your name)
    v: kernel var (same name as in your f1..f4)
    dims: iterable[int] coordinates in this block u
    mu, delta: (d,) per-dimension Gaussian params for p(x)
    """
    N = X.shape[0]
    F1 = np.ones((N, N))
    F2 = np.ones((N, N))
    F3 = np.ones((N, N))
    F4 = np.ones((N, N))
    if isinstance(dims, int):
        dims = [dims]
    for i in dims:
        xx = X[:, i]
        x, y = _pairwise(xx)
        scale = v**2

        # multiply per-dimension 1D factors (each already includes sigma**4)
        F1 *= f1(x, y, lengthscales, delta, mu) * scale
        F2 *= f2(x, y, lengthscales, delta, mu) * scale
        F3 *= f3(x, y, lengthscales, delta, mu) * scale
        F4 *= f4(x, y, lengthscales, delta, mu) * scale

    # constrained kernel combination (same signs as your 1D formulas)
    L = F1 - F2 - F3 + F4
    return L  # (N,N)


def compute_L_binary_kernel(
    X: tf.Tensor, p0: float, variance: float, dim: int
) -> np.ndarray:
    """
    Compute L matrix needed for sobol index calculation for orthogonal binary kernels.
    :param X: training input tensor
    :param p0: probability measure for the data distribution (Prob(x=0))
    :param variance: variance parameter for the binary kernel, default is 1
    :param dim: active dimension of the kernel
    :return: sobol value L matrix

    """
    assert 0 <= p0 <= 1

    N = X.shape[0]
    xx = X[:, dim]
    yy = X[:, dim]

    x = np.repeat(xx, N)
    y = np.tile(yy, N)
    p1 = 1 - p0

    L = variance * (
        p0 * (p1**2 * (1 - x) - p0 * p1 * x) * (p1**2 * (1 - y) - p0 * p1 * y)
        + p1 * (-p0 * p1 * (1 - x) + p0**2 * x) * (-p0 * p1 * (1 - y) + p0**2 * y)
    )
    L = np.reshape(L, (N, N))

    return L


def compute_L_categorical_kernel(
    X: tf.Tensor, W: tf.Tensor, kappa: tf.Tensor, p: float, variance: float, dim: int
) -> np.ndarray:
    """
    Compute L matrix needed for sobol index calculation for orthogonal categorical kernels.
    :param X: training input tensor
    :param W: parameter of categorical kernel
    :param kappa: parameter of categorical kernel
    :param p: probability measure for the data distribution (Prob(x=0))
    :param variance: variance parameter for the categorical kernel, default is 1
    :param dim: active dimension of the kernel
    :return: sobol value L matrix

    """
    assert np.abs(p.sum() - 1) < 1e-6

    A = tf.linalg.matmul(W, W, transpose_b=True) + tf.linalg.diag(kappa)
    Ap = tf.linalg.matmul(A, p)
    B = A - tf.linalg.matmul(Ap, Ap, transpose_b=True) / (
        tf.linalg.matmul(p, Ap, transpose_a=True)[0]
    )
    B = B * variance

    xx = tf.range(len(p), dtype=gpflow.config.default_float())

    K = tf.gather(
        tf.transpose(tf.gather(B, tf.cast(X[:, dim], tf.int32))), tf.cast(xx, tf.int32)
    )

    L = tf.linalg.matmul(K, K * p, transpose_a=True)

    return L


@tf.function
def compute_L_empirical_measure(
    x: tf.Tensor, w: tf.Tensor, kernel, z: tf.Tensor
) -> tf.Tensor:
    """
    Compute L matrix needed for sobol index calculation with empirical measure
    :param x: location of empirical measure [M, D_measure]
    :param w: weights of empirical measure [M, 1]
    :param kernel: constrained kernel
    :param z: training data/inducing points [N, D_kernel]
    :return: sobol value L matrix [N, N]
    """

    # Convert to tensors, preserving original dtype
    x = tf.convert_to_tensor(x)
    z = tf.convert_to_tensor(z)
    w = tf.convert_to_tensor(w)

    # Ensure consistent dtype (use the dtype of z as reference)
    target_dtype = z.dtype
    if x.dtype != target_dtype:
        x = tf.cast(x, target_dtype)
    if w.dtype != target_dtype:
        w = tf.cast(w, target_dtype)

    # Get dimensions
    # m = tf.shape(z)[0]  # number of training/inducing points
    n = tf.shape(x)[0]  # number of empirical locations

    # Compute kernel between empirical locations and training points
    kxu = kernel.K(x, z)  # Shape: [n, m]

    # Reshape weights to [1, n] for matrix multiplication
    w = tf.reshape(w, [1, n])

    # Compute L = sum_i w_i * k(x_i, z)^T * k(x_i, z)
    L = tf.matmul(w * tf.transpose(kxu), kxu)

    return L


# def compute_sobol_oak(
#     model: gpflow.models.BayesianModel,
#     delta: float,
#     mu: float,
#     _user_active_dims: List[List[int]] = None,
#     share_var_across_orders: Optional[bool] = True,
# ) -> Tuple[List[List[int]], List[float]]:
#     """
#     Compute sobol indices for Duvenaud model
#     :param model: gpflowm odel
#     :param delta: prior variance of measure p(X)
#     :param mu: prior mean of measure p(x)
#     :param share_var_across_orders: whether to share the same variance across orders,
#            if False, it uses original OrthogonalRBFKernel kernel \prod_i(1+k_i).
#     :return: list of input dimension indices and list of sobol indices
#     """
#     num_dims = model.data[0].shape[1]

#     selected_dims_oak, kernel_list = get_list_representation(
#         model.kernel, num_dims=num_dims, _user_active_dims=_user_active_dims
#     )
#     selected_dims_oak = selected_dims_oak[1:]  # skip constant term
#     if isinstance(model, (gpflow.models.SGPR, gpflow.models.SVGP)):
#         X = model.inducing_variable.Z
#     else:
#         X = model.data[0]
#     N = X.shape[0]
#     alpha = get_model_sufficient_statistics(model, get_L=False)
#     sobol = []
#     L_list = []
#     for kernel in kernel_list:
#         # print(kernel)
#         # assert isinstance(kernel, KernelComponenent)
#         if len(kernel.iComponent_list) == 0:
#             continue  # skip constant term
#         L = np.ones((N, N))
#         n_order = len(kernel.kernels)
#         for j in range(len(kernel.kernels)):
#             if share_var_across_orders:
#                 if j < 1:
#                     v = kernel.oak_kernel.variances[n_order].numpy()
#                 else:
#                     v = 1
#             else:
#                 v = kernel.kernels[j].base_kernel.variance.numpy()

#             dim = kernel.kernels[j].active_dims[0]

#             if isinstance(kernel.kernels[j], OrthogonalRBFKernel):

#                 if isinstance(kernel.kernels[j].base_kernel, gpflow.kernels.RBF) and (
#                     not isinstance(kernel.kernels[j].measure, EmpiricalMeasure)
#                     and (not isinstance(kernel.kernels[j].measure, MOGMeasure))
#                 ):
#                     l = kernel.kernels[j].base_kernel.lengthscales.numpy()
#                     L = L * compute_L(
#                         X,
#                         l,
#                         v,
#                         dim,
#                         delta,
#                         mu,
#                     )

#                 elif isinstance(kernel.kernels[j].measure, EmpiricalMeasure):
#                     L = (
#                         v ** 2
#                         * L
#                         * compute_L_empirical_measure(
#                             kernel.kernels[j].measure.location,
#                             kernel.kernels[j].measure.weights,
#                             kernel.kernels[j],
#                             tf.reshape(X[:, dim], [-1, 1]),
#                         )
#                     )
#                 else:
#                     raise NotImplementedError

#             elif isinstance(kernel.kernels[j], OrthogonalBinary):
#                 p0 = kernel.kernels[j].p0
#                 L = L * compute_L_binary_kernel(X, p0, v, dim)

#             elif isinstance(kernel.kernels[j], OrthogonalCategorical):
#                 p = kernel.kernels[j].p
#                 W = kernel.kernels[j].W
#                 kappa = kernel.kernels[j].kappa
#                 L = L * compute_L_categorical_kernel(X, W, kappa, p, v, dim)

#             else:
#                 raise NotImplementedError
#         L_list.append(L)
#         mean_term = tf.tensordot(
#             tf.tensordot(tf.transpose(alpha), L, axes=1), alpha, axes=1
#         ).numpy()[0][0]
#         sobol.append(mean_term)

#     assert len(selected_dims_oak) == len(sobol)
#     return selected_dims_oak, sobol


def compute_sobol(
    model: GPModel,
    kernel_list: list,
    delta: float,
    mu: float,
    alpha: np.ndarray,
    sparse_gp: bool = True,
):
    # compute Sobol in eq (40) of G.1 of paper
    if sparse_gp:
        X = model.inducing_variable.Z
    else:
        X = model.data[0]
    N = X.shape[0]
    sobol = []
    L_list = []
    for kernel in kernel_list:
        assert not isinstance(
            kernel, KernelComponenent
        ), "should use duvenaud sobol calculation code"
        if isinstance(kernel, gpflow.kernels.base.Product):  # exclude constant term
            L = np.ones((N, N))
            for j in range(len(kernel.kernels)):
                l = kernel.kernels[j].base_kernel.lengthscales.numpy()
                v = kernel.kernels[j].base_kernel.variance.numpy()
                dim = kernel.kernels[j].active_dims[0]
                L = L * compute_L(X, l, v, dim, delta, mu)
            L_list.append(L)
            sobol.append(
                tf.tensordot(
                    tf.tensordot(tf.transpose(alpha), L, axes=1), alpha, axes=1
                ).numpy()[0][0]
            )

        else:
            if type(kernel) != gpflow.kernels.statics.Constant and not isinstance(
                kernel, KernelComponenent
            ):
                l = kernel.base_kernel.lengthscales.numpy()
                v = kernel.base_kernel.variance.numpy()
                dim = kernel.active_dims[0]
                L = compute_L(X, l, v, dim, delta, mu)

                L_list.append(L)
                sobol.append(
                    tf.tensordot(
                        tf.tensordot(tf.transpose(alpha), L, axes=1), alpha, axes=1
                    ).numpy()[0][0]
                )

    return sobol


def get_prediction_component(
    m: gpflow.models.BayesianModel,
    alpha: tf.Tensor,
    X: np.ndarray = None,
    share_var_across_orders: Optional[bool] = True,
) -> list:
    r"""
    Return predictive mean for dataset 1 and 2
    :param m: GP model
    :param X: concatenation of data to make predictions: first half of X are from dataset 1,
              last half of X are from dataset 2. If it is None, then X is set to be the training data.
    :param alpha: statistics used to make predictions, e.g. K^{-1}y
    :param share_var_across_orders: whether to share the same variance across orders,
           if False, it uses original OrthogonalRBFKernel kernel \prod_i(1+k_i)
    :return:  prediction of each kernel component of two datasets (e.g., two different simulation runs), concatenated together
    """
    if X is None:
        X = m.data[0]
    selected_dims, _ = get_list_representation(m.kernel, num_dims=X.shape[1])
    tuple_of_indices = selected_dims[1:]
    out = []
    if isinstance(m, gpflow.models.GPR):
        X_conditioned = m.data[0]
    elif isinstance(m, (gpflow.models.SGPR, gpflow.models.SVGP)):
        X_conditioned = m.inducing_variable.Z
    else:
        raise ValueError(
            (
                "Unsupported model type for prediction component extraction. "
                "Supported types are GPR, SGPR, and SVGP."
            )
        )

    for n in range(len(tuple_of_indices)):
        Kxx = tf.ones([X.shape[0], alpha.shape[0]], dtype=tf.dtypes.float64)
        num_interaction = len(tuple_of_indices[n])
        for ii in range(num_interaction):
            idx = tuple_of_indices[n][ii]
            Kxx *= m.kernel.kernels[idx].K(
                np.reshape(X[:, idx], (-1, 1)), X_conditioned[:, idx : idx + 1]
            )
        if share_var_across_orders:
            Kxx *= m.kernel.variances[num_interaction]

        predictive_component_mean = tf.matmul(Kxx, alpha)
        out.append(predictive_component_mean[:, 0])
    return out


def initialize_kmeans_with_binary(
    X: tf.Tensor,
    binary_index: list,
    continuous_index: Optional[list] = None,
    n_clusters: Optional[int] = 200,
):
    # K-means with combination of continuous and binary feature
    Z = np.zeros([n_clusters, X.shape[1]])

    for index in binary_index:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:, index][:, None])
        Z[:, index] = kmeans.cluster_centers_.astype(int)[:, 0]

    if continuous_index is not None:
        kmeans_continuous = KMeans(n_clusters=n_clusters, random_state=0).fit(
            X[:, continuous_index]
        )
        Z[:, continuous_index] = kmeans_continuous.cluster_centers_

    return Z


def initialize_kmeans_with_categorical(
    X: tf.Tensor,
    binary_index: list,
    categorical_index: list,
    continuous_index: list,
    n_clusters: Optional[int] = 200,
):
    # K-means with combination of continuous and categorical feature
    Z = np.zeros([n_clusters, X.shape[1]])

    for index in binary_index + categorical_index:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:, index][:, None])
        Z[:, index] = kmeans.cluster_centers_.astype(int)[:, 0]

    kmeans_continuous = KMeans(n_clusters=n_clusters, random_state=0).fit(
        X[:, continuous_index]
    )
    Z[:, continuous_index] = kmeans_continuous.cluster_centers_

    return Z


# ──────────────────────────────────────────────────────────────────────────────
#  Helper for safe normalisation
# ──────────────────────────────────────────────────────────────────────────────
def sobol_normalise(sobol_vals: List[float], eps: float = 1e-12) -> List[float]:
    """
    Turn a list of Sobol numerators into first-order indices, guarding
    against a zero or NaN total variance.
    """
    total = float(np.nansum(sobol_vals))
    total = total if total > eps else eps
    return [v / total for v in sobol_vals]


# ──────────────────────────────────────────────────────────────────────────────
#  Main routine
# ──────────────────────────────────────────────────────────────────────────────
def compute_sobol_oak(
    model: "gpflow.models.BayesianModel",
    delta: float,
    mu: float,
    time_point: Optional[float] = None,  # None → unconditional
    time_dim: Optional[int] = None,  # column index of time
    _user_active_dims: Optional[List[List[int]]] = None,
    share_var_across_orders: bool = True,
    # use_noise_kernel=False,
    spatial_block_idx=False,
) -> Tuple[List[List[int]], List[float]]:
    """
    Sobol numerators αᵀ L α for an OAK GP, optionally at a fixed time t*.

    Variance handling:
        • If `share_var_across_orders` is True, the *first* sub-kernel of
          each interaction order gets `v = component.oak_kernel.variances[d]`;
          all remaining sub-kernels use v = 1.       (Same as baseline.)
        • Otherwise `v` is taken from the *sub-kernel* itself:
              - `subk.base_kernel.variance`  (OrthogonalRBFKernel & friends)
              - `subk.variance`              (OrthogonalBinary / Categorical)

    The rest of the routine follows the baseline exactly, with a single
    extension: when `dim == time_dim` **and** `time_point` is given, we
    replace the usual integral factor by
          k(t*, X_t) k(t*, X_t)ᵀ,
    which realises the conditional variance definition.
    """
    # 1.  kernel structure (skip constant term) -------------------------
    num_dims = (
        sum(len(d) for d in _user_active_dims)
        if _user_active_dims
        else model.data[0].shape[1]
    )
    sel, components = get_list_representation(
        model.kernel,
        num_dims=num_dims,
        _user_active_dims=_user_active_dims,
    )
    sel, components = sel[1:], components[1:]
    if spatial_block_idx:
        # remove block is contains spatial_block_idx
        components = [
            components[i] for i, s in enumerate(sel) if spatial_block_idx not in s
        ]
        sel = [s for s in sel if spatial_block_idx not in s]

    # 2.  inputs X and α statistics -------------------------------------
    is_sparse = isinstance(model, (gpflow.models.SGPR, gpflow.models.SVGP))
    X_full = model.inducing_variable.Z.numpy() if is_sparse else model.data[0].numpy()
    alpha_full = get_model_sufficient_statistics(model, get_L=False)  # (N,1)

    if time_point is not None:
        if time_dim is None:
            raise ValueError("`time_dim` must be set when `time_point` is given.")
        if is_sparse:
            X, alpha = X_full, alpha_full
        else:
            mask = np.isclose(X_full[:, time_dim], time_point, atol=1e-9, rtol=1e-6)
            if not np.any(mask):
                raise ValueError(
                    f"No training row matches t*={time_point} in column {time_dim}."
                )
            X, alpha = X_full[mask], alpha_full[mask]
    else:
        X, alpha = X_full, alpha_full

    N = X.shape[0]
    sobol_vals: List[float] = []
    # 3.  component loop ------------------------------------------------
    for comp in components:
        if len(comp.iComponent_list) == 0:
            continue  # skip constant term
        L_np = np.ones((N, N))
        n_order = len(comp.kernels)

        for j, subk in enumerate(comp.kernels):
            # -------- variance selection  (exact copy of baseline) ----
            if share_var_across_orders:
                if j < 1:  # first factor
                    v = comp.oak_kernel.variances[n_order].numpy()
                else:
                    v = 1.0
            else:
                # per-kernel variance
                if hasattr(subk, "variance"):  # Binary / Cat.
                    v = subk.variance.numpy()
                else:  # RBF / others
                    v = subk.base_kernel.variance.numpy()

            dims = subk.active_dims

            # -------- time dimension branch --------------------------
            if time_point is not None and dims == time_dim:
                k_vec = (
                    subk(  # shape (1,N)
                        tf.reshape(tf.constant(time_point, dtype=X.dtype), (1, 1)),
                        tf.reshape(X[:, dims], (-1, 1)),
                    )
                    .numpy()
                    .flatten()
                )  # length N
                L_np *= np.outer(k_vec, k_vec)
                continue

            # -------- continuous RBF --------------------------------
            if isinstance(subk, OrthogonalRBFKernel):
                if isinstance(subk.base_kernel, gpflow.kernels.RBF) and not isinstance(
                    subk.measure, (EmpiricalMeasure, MOGMeasure)
                ):
                    l = subk.base_kernel.lengthscales.numpy()
                    L_np *= compute_L(X, l, v, dims, delta, mu)
                elif isinstance(subk.measure, EmpiricalMeasure):
                    print("DEBUG", "EmpiricalMeasure in Sobol", dims, subk)
                    # assert len(dims) == 1, "EmpiricalMeasure only supports 1D active dims"
                    # L_np *= (
                    #     v**2
                    #     * compute_L_empirical_measure(
                    #         subk.measure.location,
                    #         subk.measure.weights,
                    #         subk,
                    #         tf.reshape(X[:, dims], [-1, 1]),
                    #     ).numpy()
                    # )
                    L_np *= (
                        v**2
                        * compute_L_empirical_measure(
                            subk.measure.location,
                            subk.measure.weights,
                            subk,
                            (
                                X[:, dims]
                                if isinstance(dims, (list, tuple, np.ndarray))
                                else tf.reshape(X[:, dims], [-1, 1])
                            ),
                        ).numpy()
                    )
                else:
                    raise NotImplementedError("Unsupported measure for RBF")

            # -------- binary ----------------------------------------
            elif isinstance(subk, OrthogonalBinary):
                assert len(dims) == 1, "OrthogonalBinary only supports 1D active dims"
                L_np *= compute_L_binary_kernel(X, subk.p0, v, dims)

            # -------- categorical -----------------------------------
            elif isinstance(subk, OrthogonalCategorical):
                assert (
                    len(dims) == 1
                ), "OrthogonalCategorical only supports 1D active dims"
                L_np *= compute_L_categorical_kernel(
                    X,
                    subk.W.numpy(),
                    subk.kappa.numpy(),
                    subk.p,
                    v,
                    dims,
                )
            else:
                raise NotImplementedError(f"Unsupported kernel type: {type(subk)}")

        # 4.  αᵀ L α  -----------------------------------------------
        L_tf = tf.convert_to_tensor(L_np, dtype=alpha.dtype)
        alpha_tf = tf.reshape(alpha, [-1, 1])
        num = tf.squeeze(
            tf.matmul(alpha_tf, tf.matmul(L_tf, alpha_tf), transpose_a=True)
        ).numpy()
        sobol_vals.append(0.0 if np.isnan(num) else max(0.0, num))

    if len(sel) != len(sobol_vals):
        raise RuntimeError("Kernel parsing mismatch.")

    return sel, sobol_vals
