import os
from typing import List, Union

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Kernel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class PhyloKernel(Kernel):
    """
    A phylogenetic covariance kernel with *learnable* Pagel's λ and a scale σ².
    Inputs X should have a column of integer tip-indices (0…n-1)
    specified via `active_dims` (default = [0]).
    """

    def __init__(
        self,
        V: Union[tf.Tensor, tf.Variable, np.ndarray],
        active_dims: List[int] = [0],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.V = tf.constant(V, dtype=tf.float64)
        self.variance = gpflow.Parameter(
            1.0, transform=tfp.bijectors.Softplus(), name="variance"
        )
        self.lambda_raw = gpflow.Parameter(
            0.5, transform=tfp.bijectors.Sigmoid(), name="lambda"
        )
        self.active_dims = active_dims

    @property
    def lambda_(self) -> tf.Tensor:
        return self.lambda_raw

    def V_lambda(self) -> tf.Tensor:
        lam = self.lambda_
        D = tf.linalg.diag(tf.linalg.diag_part(self.V))
        return lam * (self.V - D) + D

    def K(self, X: tf.Tensor, X2: tf.Tensor = None) -> tf.Tensor:
        idx1 = tf.cast(
            tf.reshape(tf.gather(X, self.active_dims, axis=1), [-1]), tf.int32
        )
        idx2 = (
            idx1
            if X2 is None
            else tf.cast(
                tf.reshape(tf.gather(X2, self.active_dims, axis=1), [-1]), tf.int32
            )
        )
        V_l = self.V_lambda()
        rows = tf.gather(V_l, idx1, axis=0)
        W = tf.gather(rows, idx2, axis=1)
        return self.variance * W

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        idx = tf.cast(
            tf.reshape(tf.gather(X, self.active_dims, axis=1), [-1]), tf.int32
        )
        V_l = self.V_lambda()
        d = tf.gather(tf.linalg.diag_part(V_l), idx)
        return self.variance * d
