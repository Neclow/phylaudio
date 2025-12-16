# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf

from .input_measures import (
    EmpiricalMeasure,
    GaussianMeasure,
    Measure,
    MOGMeasure,
    UniformMeasure,
)

# -


class OrthogonalRBFKernel(gpflow.kernels.Kernel):
    """
    :param base_kernel: base RBF kernel before applying orthogonality constraint
    :param measure: input measure
    :param active_dims: active dimension
    :return: constrained BRF kernel
    """

    def __init__(
        self, base_kernel: gpflow.kernels.RBF, measure: Measure, active_dims=None
    ):
        super().__init__(active_dims=active_dims)
        self.base_kernel, self.measure = base_kernel, measure
        self.active_dims = self.active_dims
        if not isinstance(base_kernel, gpflow.kernels.RBF):
            raise NotImplementedError
        if not isinstance(
            measure,
            (
                UniformMeasure,
                GaussianMeasure,
                EmpiricalMeasure,
                MOGMeasure,
            ),
        ):
            raise NotImplementedError

        if isinstance(self.measure, UniformMeasure):
            D = len(self.active_dims)

            def _as_lenvec(l, D, dtype):
                l = tf.convert_to_tensor(l, dtype=dtype)
                if l.shape.rank == 0:  # scalar lengthscale -> broadcast to D
                    l = tf.fill([D], l)
                return l

            def cov_X_s(X):
                tf.debugging.assert_shapes([(X, ("N", D))])
                if D == 1:
                    l = self.base_kernel.lengthscales
                    sigma2 = self.base_kernel.variance
                    return (
                        sigma2
                        * l
                        / (self.measure.b - self.measure.a)
                        * np.sqrt(np.pi / 2)
                        * (
                            tf.math.erf((self.measure.b - X) / np.sqrt(2) / l)
                            - tf.math.erf((self.measure.a - X) / np.sqrt(2) / l)
                        )
                    )
                else:
                    X = tf.convert_to_tensor(X)
                    tf.debugging.assert_rank_at_least(X, 2)  # (N, D)
                    D = tf.shape(X)[-1]

                    a = tf.convert_to_tensor(self.measure.a, dtype=X.dtype)  # (D,)
                    b = tf.convert_to_tensor(self.measure.b, dtype=X.dtype)  # (D,)
                    l = _as_lenvec(self.base_kernel.lengthscales, D, X.dtype)  # (D,)
                    sigma2 = tf.cast(self.base_kernel.variance, X.dtype)

                    tf.debugging.assert_shapes(
                        [
                            (X, ("N", "D")),
                            (a, ("D",)),
                            (b, ("D",)),
                            (l, ("D",)),
                        ]
                    )

                    # broadcast (N,D)
                    aN = a[tf.newaxis, :]
                    bN = b[tf.newaxis, :]
                    lN = l[tf.newaxis, :]

                    sqrt_pi_over_2 = tf.sqrt(tf.constant(np.pi, dtype=X.dtype) / 2.0)
                    inv_sqrt2 = 1.0 / tf.sqrt(tf.constant(2.0, dtype=X.dtype))

                    up = (bN - X) * inv_sqrt2 / lN
                    down = (aN - X) * inv_sqrt2 / lN

                    term = (
                        (lN / (bN - aN))
                        * sqrt_pi_over_2
                        * (tf.math.erf(up) - tf.math.erf(down))
                    )  # (N,D)
                    prod = tf.reduce_prod(term, axis=1, keepdims=True)  # (N,1)
                    return sigma2 * prod  # (N,1)

            def var_s():
                if D == 1:
                    l = self.base_kernel.lengthscales
                    sigma2 = self.base_kernel.variance
                    y = (self.measure.b - self.measure.a) / np.sqrt(2) / l
                    return (
                        2.0
                        / ((self.measure.b - self.measure.a) ** 2)
                        * sigma2
                        * l**2
                        * (
                            np.sqrt(np.pi) * y * tf.math.erf(y)
                            + tf.exp(-tf.square(y))
                            - 1.0
                        )
                    )
                else:
                    # E[k(S,S')] for S,S' ~ Uniform([a,b]) with independence across dims
                    a = tf.convert_to_tensor(self.measure.a, dtype=tf.float64)
                    b = tf.convert_to_tensor(self.measure.b, dtype=tf.float64)
                    l = _as_lenvec(
                        self.base_kernel.lengthscales, tf.shape(a)[0], a.dtype
                    )
                    sigma2 = tf.cast(self.base_kernel.variance, a.dtype)

                    Δ = b - a  # (D,)
                    y = Δ / (tf.sqrt(2.0) * l)  # (D,)
                    sqrt_pi = tf.sqrt(tf.constant(np.pi, dtype=a.dtype))

                    # 1D factor: 2/Δ^2 * l^2 * (√π y erf(y) + exp(-y^2) - 1)
                    factor_1d = (
                        (2.0 / (Δ * Δ))
                        * (l * l)
                        * (sqrt_pi * y * tf.math.erf(y) + tf.exp(-y * y) - 1.0)
                    )  # (D,)

                    return sigma2 * tf.reduce_prod(factor_1d)  # scalar

        if isinstance(self.measure, GaussianMeasure):

            # support multi-dimensional Gaussian measure over active_dims
            # new
            try:
                D = len(self.active_dims)
            except TypeError:
                # no explicit list of dims → assume 1D
                D = 1
            # common dtype for kernel params
            dtype = self.base_kernel.lengthscales.dtype
            mu_raw = self.measure.mu
            var_raw = self.measure.var
            mu = tf.cast(mu_raw, dtype)
            var = tf.cast(var_raw, dtype)
            print(f"OrthogonalRBFKernel: mu={mu}, var={var}, D={D}")

            if D == 1:
                # 1D case
                def cov_X_s(X):
                    tf.debugging.assert_shapes([(X, (..., "N", 1))])
                    l = self.base_kernel.lengthscales
                    sigma2 = self.base_kernel.variance
                    return (
                        sigma2
                        * l
                        / tf.sqrt(l**2 + var)
                        * tf.exp(-0.5 * ((X - mu) ** 2) / (l**2 + var))
                    )

                def var_s():
                    l = self.base_kernel.lengthscales
                    sigma2 = self.base_kernel.variance
                    return sigma2 * l / tf.sqrt(l**2 + 2 * var)

            else:
                # D-dimensional isotropic Gaussian
                def cov_X_s(X: tf.Tensor) -> tf.Tensor:
                    tf.debugging.assert_shapes([(X, ("N", D))])
                    l = self.base_kernel.lengthscales
                    sigma2 = self.base_kernel.variance
                    l2 = l**2
                    # denom = ℓ² + var
                    denom = l2 + var
                    # the normalisation prefactor: σ² * (ℓ² / (ℓ² + var))^(D/2)
                    scale = sigma2 * tf.pow(l2 / denom, D / 2)
                    # exponent: –½‖x – μ‖² / (ℓ² + var)
                    diffsq = tf.reduce_sum((X - mu) ** 2, axis=1, keepdims=True)
                    exponent = -0.5 * diffsq / denom
                    return scale * tf.exp(exponent)

                def var_s() -> tf.Tensor:
                    # this is K_{P,P} = ∫∫ K(x,y) dP(x) dP(y)
                    l = self.base_kernel.lengthscales
                    sigma2 = self.base_kernel.variance
                    l2 = l**2
                    # denom2 = ℓ² + 2·var
                    denom2 = l2 + 2 * var
                    # variance of the embedding: σ² * (ℓ² / (ℓ² + 2·var))^(D/2)
                    return sigma2 * tf.pow(l2 / denom2, D / 2)

        if isinstance(self.measure, EmpiricalMeasure):
            print(
                f"OrthogonalRBFKernel: EmpiricalMeasure with {len(self.measure.location)} points"
            )

            def cov_X_s(X):
                location = self.measure.location
                weights = self.measure.weights

                # Convert to tensor and ensure 2D
                X = tf.convert_to_tensor(X)
                if len(tf.shape(X)) == 1:
                    X = tf.reshape(X, [-1, 1])

                location = tf.convert_to_tensor(location)
                if len(tf.shape(location)) == 1:
                    location = tf.reshape(location, [-1, 1])

                # Handle active dimensions
                if hasattr(self, "active_dims") and self.active_dims is not None:
                    active_dims = self.active_dims
                    if isinstance(active_dims, (int, np.integer)):
                        active_dims = [active_dims]

                    # Convert active_dims to tensor with consistent type
                    active_dims = tf.convert_to_tensor(active_dims, dtype=tf.int32)

                    # Check if X already has the right dimensions (already sliced)
                    X_dim = tf.shape(X)[1]
                    expected_dim = tf.shape(active_dims)[
                        0
                    ]  # Number of active dimensions

                    # If X already has the expected number of dimensions, it's likely already sliced
                    if X_dim == expected_dim:
                        # X is already in the right subspace, use it as is
                        pass
                    elif X_dim > expected_dim:
                        # X has more dimensions, need to select the active ones
                        # Validate that all active_dims are within bounds
                        max_dim = tf.cast(
                            X_dim, tf.int32
                        )  # Cast to same type as active_dims

                        # Check bounds for each dimension
                        tf.debugging.assert_less(
                            tf.reduce_max(active_dims),
                            max_dim,
                            message=f"active_dims out of bounds for input dimensions",
                        )

                        X = tf.gather(X, active_dims, axis=1)
                    else:
                        # X has fewer dimensions than expected
                        # This happens when the kernel is called on already-sliced data
                        # Just use X as is
                        pass

                # Ensure location matches X's dimensionality
                if tf.shape(location)[1] != tf.shape(X)[1]:
                    # If location has more dimensions, select the same ones
                    if hasattr(self, "active_dims") and self.active_dims is not None:
                        active_dims_tensor = tf.convert_to_tensor(
                            active_dims, dtype=tf.int32
                        )
                        if tf.shape(location)[1] > tf.shape(active_dims_tensor)[0]:
                            location = tf.gather(location, active_dims_tensor, axis=1)

                # Compute kernel
                K_Xs = self.base_kernel(X, location)  # Shape: [N, M]

                # Weight by empirical measure weights
                return tf.matmul(K_Xs, weights)  # Shape: [N, 1]

            def var_s():
                """
                Compute variance of measure s
                """
                location = self.measure.location
                weights = self.measure.weights

                # Ensure location is at least 2D
                location = tf.convert_to_tensor(location)
                if len(tf.shape(location)) == 1:
                    location = tf.expand_dims(location, axis=-1)

                # Extract active dimensions if specified
                if hasattr(self, "active_dims") and self.active_dims is not None:
                    active_dims = self.active_dims
                    if np.isscalar(active_dims):
                        active_dims = [active_dims]

                    if tf.shape(location)[1] != len(active_dims):
                        location = tf.gather(location, active_dims, axis=1)

                # Compute weighted variance
                K_ss = self.base_kernel(location)  # Shape: [M, M]

                # Compute w^T K w
                return tf.squeeze(
                    tf.matmul(tf.matmul(weights, K_ss, transpose_a=True), weights)
                )

        # if isinstance(self.measure, EmpiricalMeasure):
        #     D = len(self.active_dims)

        #     print(f"OrthogonalRBFKernel: EmpiricalMeasure with {len(self.measure.location)} points")

        #     def cov_X_s(X):
        #         location = self.measure.location
        #         weights = self.measure.weights
        #         tf.debugging.assert_shapes(
        #             [(X, ("N", D)), (location, ("M", D)), (weights, ("M", 1))]
        #         )
        #         return tf.matmul(self.base_kernel(X, location), weights)

        #     def var_s():
        #         location = self.measure.location
        #         weights = self.measure.weights
        #         tf.debugging.assert_shapes([(location, ("M", D)), (weights, ("M", 1))])
        #         return tf.squeeze(
        #             tf.matmul(
        #                 tf.matmul(
        #                     weights, self.base_kernel(location), transpose_a=True
        #                 ),
        #                 weights,
        #             )
        #         )

        if isinstance(self.measure, MOGMeasure):

            def cov_X_s(X):
                tf.debugging.assert_shapes([(X, ("N", 1))])
                l = self.base_kernel.lengthscales
                sigma2 = self.base_kernel.variance
                mu, var, weights = (
                    self.measure.means,
                    self.measure.variances,
                    self.measure.weights,
                )
                tmp = tf.exp(-0.5 * ((X - mu) ** 2) / (l**2 + var)) / tf.sqrt(
                    l**2 + var
                )

                return sigma2 * l * tf.matmul(tmp, tf.reshape(weights, (-1, 1)))

            def var_s():
                l = self.base_kernel.lengthscales

                sigma2 = self.base_kernel.variance
                mu, var, w = (
                    self.measure.means,
                    self.measure.variances,
                    self.measure.weights,
                )
                dists = tf.square(mu[:, None] - mu[None, :])
                scales = tf.square(l) + var[:, None] + var[None, :]
                tmp = sigma2 * l / tf.sqrt(scales) * tf.exp(-0.5 * dists / scales)

                return tf.squeeze(tf.matmul(tf.matmul(w[None, :], tmp), w[:, None]))

        self.cov_X_s = cov_X_s
        self.var_s = var_s

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        :param X: input array X
        :param X2: input array X2, if None, set to X
        :return: kernel matrix K(X,X2)
        """
        cov_X_s = self.cov_X_s(X)
        if X2 is None:
            cov_X2_s = cov_X_s
        else:
            cov_X2_s = self.cov_X_s(X2)
        k = (
            self.base_kernel(X, X2)
            - tf.tensordot(cov_X_s, tf.transpose(cov_X2_s), 1) / self.var_s()
        )
        return k

    def K_diag(self, X):
        cov_X_s = self.cov_X_s(X)
        k = self.base_kernel.K_diag(X) - tf.square(cov_X_s[:, 0]) / self.var_s()
        return k
