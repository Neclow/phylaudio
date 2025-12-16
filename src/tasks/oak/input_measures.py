# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import tensorflow as tf

# -


class Measure:
    pass


class UniformMeasure(Measure):
    """
    :param a: lower bound of the uniform distribution
    :param b: upper bound of the uniform distribution
    :return: Uniform measure for inputs
    """

    def __init__(self, a: float, b: float):
        self.a, self.b = a, b


class GaussianMeasure(Measure):
    """
    :param mu: Mean of Gaussian measure
    :param var: variance of Gaussian measure
    :return: Gaussian measure for inputs
    """

    def __init__(self, mu: float, var: float):
        self.mu, self.var = mu, var


class EmpiricalMeasure(Measure):
    """
    Empirical measure as a weighted sum of Dirac deltas.

    :param location: location of the input data, shape [N, D] or [N] for 1D
    :param weights: weights on the location of the data, shape [N, 1] or [N]
    :return: Empirical dirac measure for inputs with weights on the locations
    """

    def __init__(self, location: np.ndarray, weights: Optional[np.ndarray] = None):
        # Convert to numpy array and ensure at least 2D for location
        location = np.asarray(location)
        if location.ndim == 1:
            location = location.reshape(-1, 1)
        self.location = location

        # Get number of points
        n_points = location.shape[0]

        # Handle weights
        if weights is None:
            weights = np.ones((n_points, 1)) / n_points
        else:
            weights = np.asarray(weights)
            # Ensure weights is 2D column vector
            if weights.ndim == 1:
                weights = weights.reshape(-1, 1)
            elif weights.ndim == 2 and weights.shape[1] != 1:
                # If weights is 2D but not a column vector, ensure it is
                assert (
                    weights.shape[0] == 1 or weights.shape[1] == 1
                ), f"Weights must be a vector, got shape {weights.shape}"
                if weights.shape[0] == 1:
                    weights = weights.T

        # Validate dimensions match
        assert (
            weights.shape[0] == n_points
        ), f"Number of weights ({weights.shape[0]}) must match number of locations ({n_points})"

        # Normalize weights if needed (with small tolerance)
        weight_sum = weights.sum()
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            print(f"Warning: weights sum to {weight_sum}, normalizing to 1.0")
            weights = weights / weight_sum

        # Final validation
        assert np.isclose(
            weights.sum(), 1.0, atol=1e-6
        ), f"Weights do not sum to 1.0: {weights.sum()}"

        self.weights = weights

        # Store dimensionality info
        self.n_points = n_points
        self.dim = location.shape[1]

    @property
    def shape(self):
        """Return shape of the measure support"""
        return self.location.shape

    def __repr__(self):
        return f"EmpiricalMeasure(n_points={self.n_points}, dim={self.dim})"

    def __len__(self):
        """Number of support points"""
        return self.n_points


# class EmpiricalMeasure(Measure):
#     """
#     :param location: location of the input data
#     :param weights: weights on the location of the data
#     :return: Empirical dirac measure for inputs with weights on the locations
#     """

#     def __init__(self, location: np.ndarray, weights: Optional[np.ndarray] = None):
#         self.location = location
#         if weights is None:
#             weights = 1 / len(location) * np.ones((location.shape[0], 1))
#         assert np.isclose(
#             weights.sum(), 1.0, atol=1e-6
#         ), f"not close to 1 {weights.sum()}"
#         self.weights = weights


class MOGMeasure(Measure):
    """
    :param means: mean of the Gaussian measures
    :param variances: variances of the Gaussian measures
    :param weights: weights on the Gaussian measures
    :return: mixture of Gaussian measure
    """

    def __init__(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray):
        tf.debugging.assert_shapes(
            [(means, ("K",)), (variances, ("K",)), (weights, ("K",))]
        )
        assert np.isclose(
            weights.sum(), 1.0, atol=1e-6
        ), f"Weights not close to 1 {weights.sum()}"
        self.means, self.variances, self.weights = (
            means.astype(float),
            variances.astype(float),
            weights,
        )
