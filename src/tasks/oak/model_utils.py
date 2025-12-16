# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Type

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow import set_trainable
from gpflow.inducing_variables import InducingPoints
from gpflow.models import GPR, SGPR, GPModel
from gpflow.models.training_mixins import RegressionData
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tensorflow_probability import distributions as tfd

from .copula_flow import NormalizerGeneralizedCopula
from .input_measures import MOGMeasure
from .normalising_flow import Normalizer, Normalizer2D, NormalizerGeneralized
from .oak_kernel import OAKKernel, get_list_representation

# from .plotting_utils import FigureDescription, save_fig_list
from .utils import compute_sobol_oak, initialize_kmeans_with_categorical

# -

f64 = gpflow.utilities.to_default_float


def get_kmeans_centers(X: np.ndarray, K: int = 500) -> np.ndarray:
    """
    :param X: N * D input array
    :param K: number of clusters
    :return: K-means clustering of input X
    """
    np.random.seed(44)
    tf.random.set_seed(44)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    Z = kmeans.cluster_centers_
    return Z


def save_model(
    model: GPModel,
    filename: Path,
) -> None:
    """
    :param model: GPflow model parameters to save
    :param filename: location to save the model to
    :return save model parameters to a local directory
    """
    if isinstance(model, gpflow.models.SVGP):
        hyperparams = [
            model.parameters[i].numpy() for i in range(len(model.parameters))
        ]
    else:
        hyperparams = [
            model.trainable_parameters[i].numpy()
            for i in range(len(model.trainable_parameters))
        ]

    os.makedirs(filename.parents[0], exist_ok=True)
    np.savez(filename, hyperparams=hyperparams)


def concurvity_penalty_from_model(model, lam=1e-2, eps=1e-8, basis="auto"):
    """
    lam: strength of the concurvity regularizer
    basis: "auto" -> training X for GPR, inducing Z for SGPR; or pass a tensor explicitly
    """
    # Pick the basis points on which to measure overlap
    if basis == "auto":
        if hasattr(model, "inducing_variable") and model.inducing_variable is not None:
            B = model.inducing_variable.Z  # SGPR / sparse case
        else:
            B = model.data[0]  # GPR case
    else:
        B = basis  # a tensor you provide

    # Build per-block Gram matrices on the basis
    # Each base block kernel in your OAK kernel should apply its own active_dims internally
    K_blocks = [
        k.K(k.slice(B)[0]) for k in model.kernel.kernels
    ]  # list of [n,n] or [m,m]

    # Frobenius-cosine squared between all pairs (j<l)
    norms = [tf.sqrt(tf.reduce_sum(tf.square(Kj))) + eps for Kj in K_blocks]
    reg = 0.0
    for j in range(len(K_blocks)):
        for l in range(j):
            cos = tf.reduce_sum(K_blocks[j] * K_blocks[l]) / (norms[j] * norms[l])
            reg += tf.square(cos)
    return lam * reg


def make_regularized_closure(model, lam_concurvity=1e-2):
    base_closure = (
        model.training_loss_closure()
    )  # GPflow's built-in loss (MLL + priors)

    @tf.function  # optional; keeps it fast and differentiable
    def _closure():
        loss = base_closure()
        if lam_concurvity and lam_concurvity > 0.0:
            loss += concurvity_penalty_from_model(
                model, lam=lam_concurvity, basis="auto"
            )
        return loss

    return _closure


def load_model(
    model: GPModel,
    filename: Path,
    load_all_parameters=False,
) -> None:
    """
    :param model: GPflow model parameters to load
    :param filename: location to load the model from
    :param load_all_parameters: whether to load all parameters or only trainable parameters
    :return load model parameters from a local directory
    """
    # We need allow_pickle=True because model parameters include objects (e.g. InducingPoints)
    model_params = np.load(str(filename), allow_pickle=True)["hyperparams"]

    if load_all_parameters:
        for i in range(len(model.parameters)):
            model.parameters[i].assign(model_params[i])
    else:
        for i in range(len(model.trainable_parameters)):
            print(model_params[i], model.trainable_parameters[i])
            model.trainable_parameters[i].assign(model_params[i])


def create_model_oak(
    data: RegressionData,
    max_interaction_depth: int = 2,
    constrain_orthogonal: bool = True,
    inducing_pts: np.ndarray = None,
    optimise: bool = False,
    zfixed: bool = True,
    p0=None,
    p=None,
    lengthscale_bounds=None,
    empirical_locations=None,
    empirical_weights=None,
    use_sparsity_prior: bool = True,
    gmm_measures=None,
    share_var_across_orders: bool = True,
    base_kernels: Optional[List[Type[gpflow.kernels.Kernel]]] = None,
    active_dims: Optional[List[List[int]]] = None,
    noise_kernel: Optional[gpflow.kernels.Kernel] = None,
    spatial_block_idx=None,
    uniform_measure: Optional[bool] = None,
    lam_concurvity: float = 0.0,
) -> GPModel:
    """
    Build an OAK GP model.  `base_kernels` and `active_dims` are now
    understood to be specified **per block** (not per raw dimension).
    """
    X, Y = data
    num_dims = X.shape[1]

    if noise_kernel is not None:
        assert active_dims, "If noise_kernel is set, active_dims must be specified."

    # ------------------------------------------------------------------
    # 0) active-dims: default = one block per raw dim
    # ------------------------------------------------------------------
    if active_dims is None:
        active_dims = [[i] for i in range(num_dims)]
    num_blocks = len(active_dims)

    # ------------------------------------------------------------------
    # 1) p0 / p lists are still per raw dimension
    # ------------------------------------------------------------------
    if p0 is None:
        p0 = [None] * num_blocks
    if p is None:
        p = [None] * num_blocks

    # ------------------------------------------------------------------
    # 2) base_kernels: build default list **per block**
    # ------------------------------------------------------------------
    if base_kernels is None:
        base_kernels = []
        for block in active_dims:
            # a block is “continuous” only if *all* its dims are continuous
            if all((p0[d] is None and p[d] is None) for d in block):
                base_kernels.append(gpflow.kernels.RBF)  # default RBF
            else:
                base_kernels.append(None)  # handled by p0/p
    else:
        if len(base_kernels) != num_blocks:
            raise ValueError(
                f"base_kernels must have one entry per block "
                f"(len(active_dims) = {num_blocks}), got {len(base_kernels)}"
            )
    # ------------------------------------------------------------------
    # 3) Instantiate OAKKernel
    # ------------------------------------------------------------------
    k = OAKKernel(
        base_kernels=base_kernels,
        num_dims=num_dims,
        max_interaction_depth=max_interaction_depth,
        noise_kernel=noise_kernel,
        active_dims=active_dims,
        constrain_orthogonal=constrain_orthogonal,
        p0=p0,
        p=p,
        lengthscale_bounds=lengthscale_bounds,
        empirical_locations=empirical_locations,
        empirical_weights=empirical_weights,
        gmm_measures=gmm_measures,
        share_var_across_orders=share_var_across_orders,
        uniform_measure=uniform_measure,
        spatial_block_idx=spatial_block_idx,
    )

    # ------------------------------------------------------------------
    # 4) Choose GPR vs SGPR and finish exactly as before
    # ------------------------------------------------------------------
    if inducing_pts is not None:
        model = SGPR(
            data,
            mean_function=None,
            kernel=k,
            inducing_variable=InducingPoints(inducing_pts),
        )
        if zfixed:
            set_trainable(model.inducing_variable, False)
    else:
        model = GPR(data, mean_function=None, kernel=k)

    if use_sparsity_prior and share_var_across_orders:
        for v in model.kernel.variances:
            v.prior = tfd.Gamma(f64(1.0), f64(0.2))

    model.likelihood.variance.assign(0.01)

    for i, v in enumerate(model.trainable_variables):
        print(i, type(v), getattr(v, "name", None))
        if v is None or not isinstance(v, tf.Variable):
            print("-> BAD ENTRY at index", i, v)
    if optimise:
        t_start = time.time()
        # gpflow.optimizers.Scipy().minimize(
        #     model.training_loss_closure(),
        #     model.trainable_variables,
        #     method="BFGS",
        # )
        loss_closure = make_regularized_closure(model, lam_concurvity=lam_concurvity)
        gpflow.optimizers.Scipy().minimize(
            loss_closure,
            model.trainable_variables,
            method="BFGS",
        )
        print(f"Optimisation took {time.time() - t_start:.1f}s")

    return model


def apply_normalise_flow(
    X: tf.Tensor,
    active_dims: List[List[int]],
    input_flows: List[Optional[gpflow.base.Module]],
) -> tf.Tensor:
    """
    Return a copy of X where every column (or joint block of columns)
    has been passed through its bijector, if present.

    The same flow object may appear in several slots of `input_flows`
    (one per raw dimension).  We make sure to call it only once.
    """
    X = X.copy()  # work on NumPy copy

    for d, flow in enumerate(input_flows):
        this_active_dims = active_dims[d]

        if input_flows[d] is not None:
            if len(this_active_dims) == 1:  # old 1-D behaviour
                X[:, this_active_dims] = flow.bijector(X[:, this_active_dims])
            elif len(this_active_dims) > 1:  # joint block   (e.g. dims [1,2])
                X[:, this_active_dims] = flow.bijector(X[:, this_active_dims])
            else:
                raise NotImplementedError(
                    "Normalising flow not implemented for blocks of size > 2"
                )
    return X


class oak_model:
    def __init__(
        self,
        max_interaction_depth=2,
        num_inducing=200,
        lengthscale_bounds=[1e-3, 1e3],
        binary_feature: Optional[List[int]] = None,
        categorical_feature: Optional[List[int]] = None,
        empirical_measure: Optional[List[int]] = None,
        use_sparsity_prior: bool = True,
        gmm_measure: Optional[List[int]] = None,
        sparse: bool = False,
        use_normalising_flow: bool = True,
        share_var_across_orders: bool = True,
        base_kernels: Optional[List[Type[gpflow.kernels.Kernel]]] = None,
        active_dims: Optional[List[List[int]]] = None,
        noise_kernel: Optional[gpflow.kernels.Kernel] = None,
        spatial_block_idx=None,
        uniform_measure: Optional[bool] = None,
        lam_concurvity: float = 0.0,
        use_copula_blks: Optional[bool] = None,
    ):
        """
        :param max_interaction_depth: maximum number of interaction terms to consider
        :param num_inducing: number of inducing points
        :param lengthscale_bounds: bounds for lengthscale parameters
        :param binary_feature: list of indices for binary features
        :param categorical_feature: list of indices for categorical features
        :param empirical_measure: list of indices using empirical measures, if using Gaussian measure, this is set to None
        :param use_sparsity_prior: use sparsity prior on kernel variances
        :param gmm_measure: use gaussian mixture model. If index is 0 it will use a Gaussian measure, otherwise
        :param sparse: Boolean to indicate whether to use sparse GP with inducing points. Defaults to False.
        :param use_normalising_flow: whether to use normalising flow, if not, continuous features are standardised
        :param share_var_across_orders: whether to share the same variance across orders,
           if False, it uses kernel of the form \prod_i(1+k_i) in Duvenaud (2011).
        :return: OAK model class with model fitting, prediction, attribution and plotting utils.
        """
        self.max_interaction_depth = max_interaction_depth
        self.num_inducing = num_inducing
        self.lengthscale_bounds = lengthscale_bounds
        self.binary_feature = binary_feature
        self.categorical_feature = categorical_feature
        self.use_sparsity_prior = use_sparsity_prior

        # state filled in during fit call
        self.input_flows = None
        self.scaler_y = None
        self.Y_scaled = None
        self.X_scaled = None
        self.alpha = None
        self.continuous_index = None
        self.binary_index = None
        self.categorical_index = None
        self.empirical_measure = empirical_measure
        self.empirical_locations = None
        self.empirical_weights = None
        self.gmm_measure = gmm_measure
        self.estimated_gmm_measures = None  # sklearn GMM estimates
        self.sparse = sparse
        self.use_normalising_flow = use_normalising_flow
        self.share_var_across_orders = share_var_across_orders
        self._user_base_kernels = base_kernels
        self._user_active_dims = active_dims
        self.noise_kernel = noise_kernel
        self.lam_concurvity = lam_concurvity
        self.use_copula_blks = use_copula_blks
        if self.use_copula_blks is None:
            self.use_copula_blks = [None for _ in range(len(active_dims))]
        self.uniform_measure = uniform_measure
        self.spatial_block_idx = spatial_block_idx

    def fit(
        self,
        X: tf.Tensor,
        Y: tf.Tensor,
        optimise: bool = True,
        initialise_inducing_points: bool = True,
    ):
        """
        :param X, Y data to fit the model on
        :param optimise: whether to optimise the model
        :param initialise_inducing_points: whether to initialise inducing points with K-means
        """
        self._user_active_dims = (
            self._user_active_dims
            if self._user_active_dims is not None
            else [[i] for i in range(0, X.shape[1])]
        )
        self.xmin, self.xmax = X.min(0), X.max(0)
        self.num_dims = X.shape[1]
        num_blks = len(self._user_active_dims)

        (
            self.continuous_index,
            self.binary_index,
            self.categorical_index,
            p0,
            p,
        ) = _calculate_features(
            X,
            categorical_feature=self.categorical_feature,
            binary_feature=self.binary_feature,
        )

        # Validate empirical/GMM measures
        if self.empirical_measure is not None:
            self.empirical_measure_list = [
                item for sublist in self.empirical_measure for item in sublist
            ]
            if not set(self.empirical_measure_list).issubset(self.continuous_index):
                raise ValueError(
                    f"Empirical measure={self.empirical_measure} should only be used on non-binary/categorical inputs {self.continuous_index}"
                )
        if self.gmm_measure is not None:
            if len(self.gmm_measure) != len(self._user_active_dims):
                raise ValueError(
                    f"Must specify number of components for each inputs dimension 1..{len(self._user_active_dims)}"
                )
            idx_gmm = np.flatnonzero(self.gmm_measure)
            if not set(idx_gmm).issubset(self.continuous_index):
                raise ValueError(
                    f"GMM measure on inputs {idx_gmm} should only be used on continuous inputs {self.continuous_index}"
                )

        # Estimate any GMM measures
        self.estimated_gmm_measures = [None] * len(self._user_active_dims)
        if self.gmm_measure is not None:
            for i_dim in np.flatnonzero(self.gmm_measure):
                self.estimated_gmm_measures[i_dim] = estimate_one_dim_gmm(
                    K=self.gmm_measure[i_dim], X=X[:, i_dim]
                )

        # Prepare scaling / normalising flows across blocks
        if self.noise_kernel is not None:
            assert (
                self._user_active_dims
            ), "If noise_kernel is set, active_dims must be specified."

        blocks = self._user_active_dims or [[i] for i in range(self.num_dims)]
        self.input_flows = [None] * len(blocks)
        for i, blk in enumerate(blocks):
            # Only apply a joint flow if all dims are continuous and not empirical/GMM
            if not all(d in self.continuous_index for d in blk):
                continue
            if self.empirical_measure and any(
                d in self.empirical_measure_list for d in blk
            ):
                continue
            if self.gmm_measure and any(self.gmm_measure[d] for d in blk):
                continue

            if self.uniform_measure is not None:
                if self.uniform_measure[i]:
                    # use uniform measure, no transform
                    continue

            if self.use_normalising_flow:
                print(
                    f"Normalising flow for block {blk} with size {len(blk)} and {blk}"
                )
                if len(blk) == 1:
                    flow = Normalizer(X[:, blk[0]])
                elif len(blk) > 1:
                    if self.use_copula_blks[i]:
                        flow = NormalizerGeneralizedCopula(X[:, blk])
                    else:
                        flow = NormalizerGeneralized(X[:, blk])
                else:
                    print(len(blk) > 1)
                    raise NotImplementedError(
                        f"Normalising flow not implemented for blocks of size {len(blk)}"
                    )
                opt = gpflow.optimizers.Scipy()
                opt.minimize(flow.KL_objective, flow.trainable_variables)
                self.input_flows[i] = flow

        # Fit y-scaler
        self.scaler_y = preprocessing.StandardScaler().fit(Y)
        self.Y_scaled = self.scaler_y.transform(Y)

        # Empirical X-scaler
        if self.empirical_measure is not None:
            self.scaler_X_empirical = preprocessing.StandardScaler().fit(
                X[:, self.empirical_measure_list]
            )
        # Standard scaler if no flows
        if not self.use_normalising_flow:
            self.scaler_X_continuous = preprocessing.StandardScaler().fit(
                X[:, self.continuous_index]
            )

        # Transform inputs
        self.X_scaled = self._transform_x(X)
        self.empirical_locations = [None] * num_blks
        self.empirical_weights = [None] * num_blks

        # Compute empirical locations/weights
        if self.empirical_measure is not None:
            for blk, ii in enumerate(self.empirical_measure):
                # extract ii from block
                # account for ii being a list
                locs, counts = np.unique(
                    self.X_scaled[:, ii], axis=0, return_counts=True
                )
                self.empirical_locations[blk] = locs  # shape (M, d)
                self.empirical_weights[blk] = (counts / counts.sum()).reshape(
                    -1, 1
                )  # (M,1)

        # Sanity checks
        assert np.allclose(self.X_scaled[:, self.binary_index], X[:, self.binary_index])
        assert np.allclose(
            self.X_scaled[:, self.categorical_index], X[:, self.categorical_index]
        )
        if self.gmm_measure is not None:
            assert np.allclose(
                self.X_scaled[:, np.flatnonzero(self.gmm_measure)],
                X[:, np.flatnonzero(self.gmm_measure)],
            )
        if self.empirical_measure is not None:
            inv = [
                self._get_x_inverse_transformer(i)(self.X_scaled[:, i])
                for i in self.empirical_measure_list
            ]
            assert np.allclose(np.stack(inv, axis=1), X[:, self.empirical_measure_list])

        # Inducing points
        Z = None
        if X.shape[0] > 5000 or self.sparse:
            X_ind = self.X_scaled
            if initialise_inducing_points:
                if (p0 is None) and (p is None):
                    Z = (
                        KMeans(n_clusters=self.num_inducing, random_state=0)
                        .fit(X_ind)
                        .cluster_centers_
                    )
                else:
                    Z = initialize_kmeans_with_categorical(
                        X_ind,
                        binary_index=self.binary_index,
                        categorical_index=self.categorical_index,
                        continuous_index=self.continuous_index,
                        n_clusters=self.num_inducing,
                    )
            else:
                Z = X_ind[: self.num_inducing]

        # Build final GP model
        self.m = create_model_oak(
            (self.X_scaled, self.Y_scaled),
            max_interaction_depth=self.max_interaction_depth,
            inducing_pts=Z,
            optimise=optimise,
            p0=p0,
            p=p,
            lengthscale_bounds=self.lengthscale_bounds,
            use_sparsity_prior=self.use_sparsity_prior,
            empirical_locations=self.empirical_locations,
            empirical_weights=self.empirical_weights,
            gmm_measures=self.estimated_gmm_measures,
            share_var_across_orders=self.share_var_across_orders,
            base_kernels=self._user_base_kernels,
            active_dims=self._user_active_dims,
            noise_kernel=self.noise_kernel,
            lam_concurvity=self.lam_concurvity,
            uniform_measure=self.uniform_measure,
            spatial_block_idx=self.spatial_block_idx,
        )

    def optimise(
        self,
        compile: bool = True,
    ):

        print("Model prior to optimisation")
        gpflow.utilities.print_summary(self.m, fmt="notebook")
        self.alpha = None
        t_start = time.time()
        opt = gpflow.optimizers.Scipy()
        loss_closure = make_regularized_closure(
            self.m, lam_concurvity=self.lam_concurvity
        )
        opt.minimize(
            loss_closure,
            self.m.trainable_variables,
            method="BFGS",
            compile=compile,
        )
        gpflow.utilities.print_summary(self.m, fmt="notebook")
        print(f"Training took {time.time() - t_start:.1f} seconds.")

    def predict(self, X: tf.Tensor, clip=False) -> tf.Tensor:
        """
        :param X: inputs to predict the response on
        :param clip: whether to slip X between x_min and x_max along each dimension
        :return: predicted response on input X
        """
        if clip:
            X_scaled = self._transform_x(np.clip(X, self.xmin, self.xmax))
        else:
            X_scaled = self._transform_x(X)
        try:
            y_pred = self.m.predict_f(X_scaled)[0].numpy()
            if np.any(np.isnan(y_pred)):
                print(X_scaled)
                print("Warning: NaN values found in y_pred.")
            return self.scaler_y.inverse_transform(y_pred)[:, 0]
        except ValueError:
            print("test X is outside the range of training input, try clipping X.")

    def get_loglik(self, X: tf.Tensor, y: tf.Tensor, clip=False) -> tf.Tensor:
        """
        :param X,y: inputs and output
        :param clip: whether to slip X between x_min and x_max along each dimension
        :return log likelihood on (X,y)
        """
        if clip:
            X_scaled = self._transform_x(np.clip(X, self.xmin, self.xmax))
        else:
            X_scaled = self._transform_x(X)

        return (
            self.m.predict_log_density((X_scaled, self.scaler_y.transform(y)))
            .numpy()
            .mean()
        )

    def _transform_x(self, X: tf.Tensor) -> tf.Tensor:
        """
        :param X: input to do transformation on
        :return: transformation for continuous features: normalising flow with Gaussian measure or standardization with empirical measure
        """
        X = apply_normalise_flow(X, self._user_active_dims, self.input_flows)
        if self.empirical_measure is not None:
            X[:, self.empirical_measure_list] = self.scaler_X_empirical.transform(
                X[:, self.empirical_measure_list]
            )
        if not self.use_normalising_flow:
            X[:, self.continuous_index] = self.scaler_X_continuous.transform(
                X[:, self.continuous_index]
            )
        return X

    def _get_x_inverse_transformer(
        self, i: int
    ) -> Optional[Callable[[tf.Tensor], tf.Tensor]]:
        """
        Return a callable that maps the *transformed* column i back to the
        original data space, or None if we cannot provide a 1-D inverse
        (e.g. i is part of a 2-D joint flow).
        """
        assert i in self.continuous_index
        flow = self.input_flows[i]

        # empirical / GMM cases unchanged …
        if self.empirical_measure and i in self.empirical_measure_list:
            idx = self.empirical_measure_list.index(i)
            mean_i, std_i = self.scaler_X_empirical.mean_[idx], np.sqrt(
                self.scaler_X_empirical.var_[idx]
            )
            return lambda x: x * std_i + mean_i

        if self.gmm_measure and self.gmm_measure[i]:
            return None

        if isinstance(flow, Normalizer):
            return flow.bijector.inverse

        if isinstance(flow, NormalizerGeneralized):
            if flow.decorrelate:
                # joint whitening step → needs the full y vector
                def inv_full(y_full: tf.Tensor) -> tf.Tensor:
                    # undo entire chain, then slice out dimension i
                    x_recon = flow.bijector.inverse(y_full)  # [..., D]
                    return x_recon[..., i]  # [...,]

                return inv_full
            else:
                # independent per‐dim flows → can invert y_i alone
                def inv_1d(y_i: tf.Tensor) -> tf.Tensor:
                    return flow._block.bijectors[i].inverse(y_i)

                return inv_1d

        if isinstance(flow, Normalizer2D):
            # joint 2-D flow → we cannot invert one dim in isolation
            return None

        return None

    def get_sobol(
        self,
        likelihood_variance=False,
        time_point=None,
        time_dim=None,
        return_variance=False,
    ):
        """
        :param likelihood_variance: whether to include likelihood noise in Sobol calculation
        :return: normalised Sobol indices for each additive term in the model
        """
        if self.noise_kernel is not None:
            assert (
                self._user_active_dims
            ), "If noise_kernel is set, active_dims must be specified."
        num_dims = (
            sum(len(d) for d in self._user_active_dims)
            if self._user_active_dims
            else self.num_dims
        )

        delta = 1
        mu = 0

        selected_dims, _ = get_list_representation(
            self.m.kernel, num_dims=num_dims, _user_active_dims=self._user_active_dims
        )
        if self.spatial_block_idx:
            # remove spatial indices and apply get_list_rep on smaller set
            # from selected_dims, remove lists containing index 1
            selected_dims = [
                d for d in selected_dims if self.spatial_block_idx not in d
            ]

        tuple_of_indices = selected_dims[1:]
        if time_point is not None:
            time_point = self.input_flows[time_dim].bijector([time_point])

        model_indices, sobols = compute_sobol_oak(
            self.m,
            delta,
            mu,
            time_point,
            time_dim,
            self._user_active_dims,
            share_var_across_orders=self.share_var_across_orders,
            spatial_block_idx=self.spatial_block_idx,
        )
        total_var = np.sum(sobols)
        variances = {}
        if self.spatial_block_idx:
            # spatial sobol compute separately
            from .utils import compute_L, get_model_sufficient_statistics

            X_full = self.m.data[0].numpy()
            alpha_full = get_model_sufficient_statistics(self.m, get_L=False)  # (N,1)
            X, alpha = X_full, alpha_full
            l = self.m.kernel.spatial_kernel.base_kernel.lengthscales.numpy()
            v = self.m.kernel.variances[0]
            dims = self.m.kernel.spatial_kernel.active_dims
            L_np = compute_L(X, l, v, dims, delta, mu)
            L_tf = tf.convert_to_tensor(L_np, dtype=alpha.dtype)
            alpha_tf = tf.reshape(alpha, [-1, 1])
            num = tf.squeeze(
                tf.matmul(alpha_tf, tf.matmul(L_tf, alpha_tf), transpose_a=True)
            ).numpy()
            spatial_sobol = 0.0 if np.isnan(num) else max(0.0, num)
            print(f"individual_spatial_sobol: {spatial_sobol:.3f}")
            variances["individual_spatial_sobol"] = spatial_sobol
            total_var += spatial_sobol

        if likelihood_variance:
            total_var += self.m.likelihood.variance.numpy()
            variances["normalized_likelihood_variance"] = float(
                self.m.likelihood.variance.numpy()
            )
            print(f"Likelihood variance: {self.m.likelihood.variance.numpy():.3f}")
            if self.noise_kernel:
                # get V_lambda as a NumPy array
                V_lambda = self.noise_kernel.V_lambda().numpy()  # shape [n,n]
                sigma2 = float(self.noise_kernel.variance.numpy())
                n = V_lambda.shape[0]  # number of inducing points
                # variance contributed by the phylo component
                phylo_var = sigma2 * np.trace(V_lambda) / n
                total_var += phylo_var
                print(f"Phylovariance contribution: {phylo_var:.3f}")
                variances["normalized_phylo_variance"] = phylo_var

        if variances.keys() is not None:
            for key in variances.keys():
                variances[key] /= total_var

        print(f"Total variance excluding likelihood variance: {total_var:.3f}")
        normalised_sobols = sobols / total_var
        self.normalised_sobols = normalised_sobols
        self.tuple_of_indices = tuple_of_indices
        if return_variance:
            return normalised_sobols, variances
        else:
            return normalised_sobols

    def sobol_summary(
        self,
        covariate_names: List[str],
        time_point: Optional[float] = None,
        time_dim: Optional[int] = None,
        likelihood_variance: bool = False,
        return_variance: bool = False,
    ) -> pd.DataFrame:
        """
        Compute normalized Sobol indices and return as a DataFrame
        with one row per interaction (including single‐feature effects),
        using real covariate names.

        :param covariate_names: list of feature names in the same order as X’s columns
        :param likelihood_variance: whether to include the likelihood noise in normalization
        """
        print("Computing Sobol indices summary table")
        # run or re‐run Sobol
        if return_variance:
            sobols, variances = self.get_sobol(
                likelihood_variance=likelihood_variance,
                time_point=time_point,
                time_dim=time_dim,
                return_variance=return_variance,
            )
        else:
            sobols = self.get_sobol(
                likelihood_variance=likelihood_variance,
                time_point=time_point,
                time_dim=time_dim,
            )
        tuples = self.tuple_of_indices  # e.g. [(0,), (1,), (0,1), ...]
        print(tuples, covariate_names)
        print(sobols)

        def name_for(tup):
            # join the names of each index in the tuple
            return " & ".join(covariate_names[i] for i in tup)

        names = [name_for(t) for t in tuples]

        df = pd.DataFrame(
            {
                "interaction": names,
                "sobol_index": sobols,
            }
        )
        if return_variance:
            # add variances to the DataFrame
            variance_df = pd.DataFrame(
                [
                    {"interaction": key, "sobol_index": value}
                    for key, value in variances.items()
                ]
            )
            df = pd.concat([df, variance_df], ignore_index=True)
        return df.sort_values("sobol_index", ascending=False).reset_index(drop=True)

    def get_shapley(
        self, likelihood_variance: bool = False, return_variance: bool = False
    ):
        """
        Analytic Shapley values for this OAK model (any order).

        Parameters
        ----------
        likelihood_variance : bool, default False
            If True, the model's observation noise is included in the
            normalisation—exactly mirroring the flag in `get_sobol()`.

        Returns
        -------
        phi : (D,) ndarray
            Shapley value for each input dimension. If additional variance components
            (e.g. phylogenetic variance) are present, the entries sum to less than 1,
            with the remainder recorded in the returned variance components.
        """
        # 1) Get Sobol indices, along with any extra variance components (e.g. phylo, likelihood)
        sobol_result = self.get_sobol(
            likelihood_variance=likelihood_variance,
            return_variance=True,
        )
        if isinstance(sobol_result, tuple):
            sobol, variance_components = sobol_result
        else:
            sobol = sobol_result
            variance_components = {}

        # expose last variance breakdown for downstream inspection
        self.last_variance_components = variance_components

        tuples = self.tuple_of_indices  # created inside get_sobol()
        active_blocks = self._user_active_dims or [[i] for i in range(self.num_dims)]
        D = len(active_blocks)

        # 2) Allocate accumulator
        phi = np.zeros(D, dtype=float)

        # 3) For every additive component u  (e.g. (1,), (0,3), … )
        #    split its Sobol mass equally among its |u| members
        for S_u, u in zip(sobol, tuples):
            order = len(u)  # |u|
            share = S_u / order  # fair share for each member
            for j in u:
                phi[j] += share

        # 4) If additional variance components exist (e.g. phylogenetic), Shapley entries
        #    represent their share of the total directly—no re-normalisation.
        #    Without extras, retain legacy normalisation for numerical stability.
        if not variance_components:
            total = phi.sum()
            if total > 0:
                phi /= total

        if return_variance:
            return phi, variance_components

        return phi

    # def plot(
    #     self,
    #     transformer_y=None,
    #     X_columns=None,
    #     X_lists=None,
    #     top_n=None,
    #     likelihood_variance=False,
    #     semilogy=True,
    #     save_fig: Optional[str] = None,
    #     tikz_path: Optional[str] = None,
    #     ylim: Optional[List[float]] = None,
    #     quantile_range: Optional[List[float]] = None,
    #     log_axis: Optional[List[bool]] = [False, False],
    #     grid_range: Optional[List[np.ndarray]] = None,
    #     log_bin: Optional[List[bool]] = None,
    #     num_bin: Optional[int] = 100,
    # ):
    #     # -------------------------------------------------------------------------
    #     # 0.  House-keeping / defaults
    #     # -------------------------------------------------------------------------
    #     if X_columns is None:
    #         X_columns = [f"feature {i}" for i in range(self.num_dims)]
    #     if X_lists is None:
    #         X_lists = [None] * len(X_columns)
    #     if grid_range is None:
    #         grid_range = [None] * len(X_columns)
    #     if ylim is None:
    #         ylim = [None] * len(X_columns)
    #     if quantile_range is None:
    #         quantile_range = [None] * len(X_columns)
    #     if log_bin is None:
    #         log_bin = [False] * len(X_columns)

    #     # -------------------------------------------------------------------------
    #     # 1.  Map each dimension → the active-dims block it belongs to
    #     # -------------------------------------------------------------------------
    #     if self.noise_kernel is not None:
    #         assert (
    #             self._user_active_dims
    #         ), "If noise_kernel is set, active_dims must be specified."
    #     blocks = self._user_active_dims or [[i] for i in range(self.num_dims)]
    #     dim_to_block = {}
    #     for blk in blocks:
    #         for d in blk:
    #             dim_to_block[d] = blk

    #     # -------------------------------------------------------------------------
    #     # 2.  Sobol ordering
    #     # -------------------------------------------------------------------------
    #     sel, _ = get_list_representation(
    #         self.m.kernel,
    #         num_dims=self.num_dims,
    #         _user_active_dims=self._user_active_dims,
    #     )
    #     if self.spatial_block_idx:
    #         # remove spatial indices and apply get_list_rep on smaller set
    #         # from selected_dims, remove lists containing index 1
    #         sel = [d for d in sel if self.spatial_block_idx not in d]

    #     self.get_sobol(likelihood_variance)
    #     tuple_of_indices = sel[1:]  # drop constant term
    #     order = np.argsort(self.normalised_sobols)[::-1]
    #     print(tuple_of_indices)

    #     # -------------------------------------------------------------------------
    #     # 3.  Build figure list
    #     # -------------------------------------------------------------------------
    #     fig_list: List[FigureDescription] = []
    #     if top_n is None:
    #         top_n = len(order)

    #     for n in order[:top_n]:
    #         dims = tuple_of_indices[n]
    #         # ---------- 1-D effect ----------------------------------------------
    #         if len(dims) == 1:
    #             d = dims[0]
    #             # skip if this dim sits inside a multi-dim block
    #             if len(dim_to_block[d]) > 1:
    #                 continue
    #             # which block
    #             block_idx = self._user_active_dims.index(dim_to_block[d])

    #             # otherwise unchanged (continuous / binary / categorical)
    #             if d in self.continuous_index:
    #                 fig_list.append(
    #                     plotting_utils.plot_single_effect(
    #                         m=self.m,
    #                         i=block_idx,
    #                         covariate_name=X_columns[block_idx],
    #                         title=f"{X_columns[block_idx]} (R={self.normalised_sobols[n]:.3f})",
    #                         x_transform=self._get_x_inverse_transformer(block_idx),
    #                         y_transform=transformer_y,
    #                         semilogy=semilogy,
    #                         plot_corrected_data=False,
    #                         plot_raw_data=False,
    #                         X_list=X_lists[block_idx],
    #                         tikz_path=tikz_path,
    #                         ylim=ylim[block_idx],
    #                         quantile_range=quantile_range[block_idx],
    #                         log_bin=log_bin[block_idx],
    #                         num_bin=num_bin,
    #                     )
    #                 )
    #             elif d in self.binary_index:
    #                 fig_list.append(
    #                     plotting_utils.plot_single_effect_binary(
    #                         self.m,
    #                         d,
    #                         ["0", "1"],
    #                         title=f"{X_columns[d]} (R={self.normalised_sobols[n]:.3f})",
    #                         y_transform=transformer_y,
    #                         semilogy=semilogy,
    #                         tikz_path=tikz_path,
    #                     )
    #                 )
    #             else:  # categorical
    #                 fig_list.append(
    #                     plotting_utils.plot_single_effect_categorical(
    #                         self.m,
    #                         d,
    #                         [str(k) for k in range(self.m.kernel.kernels[d].num_cat)],
    #                         title=f"{X_columns[d]} (R={self.normalised_sobols[n]:.3f})",
    #                         y_transform=transformer_y,
    #                         semilogy=semilogy,
    #                         tikz_path=tikz_path,
    #                     )
    #                 )

    #         # ---------- 2-D interaction -----------------------------------------
    #         elif len(dims) == 2:
    #             # skip if either kernel-index corresponds to a true 2D-active_dims kernel
    #             if any(len(self._user_active_dims[kidx]) > 1 for kidx in dims):
    #                 continue

    #             i, j = dims

    #             # continuous–continuous
    #             print(i, j)
    #             print(X_columns[i], X_columns[j])
    #             if i in self.continuous_index and j in self.continuous_index:
    #                 fig_list.append(
    #                     plotting_utils.plot_second_order(
    #                         self.m,
    #                         i,
    #                         j,
    #                         [X_columns[i], X_columns[j]],
    #                         [
    #                             self._get_x_inverse_transformer(i),
    #                             self._get_x_inverse_transformer(j),
    #                         ],
    #                         transformer_y,
    #                         title=f"{X_columns[i]} & {X_columns[j]} "
    #                         f"(R={self.normalised_sobols[n]:.3f})",
    #                         tikz_path=tikz_path,
    #                         quantile_range=[quantile_range[i], quantile_range[j]],
    #                         log_axis=log_axis,
    #                         xx=grid_range[i],
    #                         yy=grid_range[j],
    #                         num_bin=num_bin,
    #                     )
    #                 )

    #             # continuous–binary
    #             elif i in self.continuous_index and j in self.binary_index:
    #                 fig_list.append(
    #                     plotting_utils.plot_second_order_binary(
    #                         self.m,
    #                         i,
    #                         j,
    #                         ["0", "1"],
    #                         [X_columns[i], X_columns[j]],
    #                         x_transforms=[self._get_x_inverse_transformer(i)],
    #                         y_transform=transformer_y,
    #                         title=f"{X_columns[i]} & {X_columns[j]} "
    #                         f"(R={self.normalised_sobols[n]:.3f})",
    #                         tikz_path=tikz_path,
    #                     )
    #                 )

    #             # binary–continuous
    #             elif i in self.binary_index and j in self.continuous_index:
    #                 fig_list.append(
    #                     plotting_utils.plot_second_order_binary(
    #                         self.m,
    #                         j,
    #                         i,
    #                         ["0", "1"],
    #                         [X_columns[j], X_columns[i]],
    #                         x_transforms=[self._get_x_inverse_transformer(j)],
    #                         y_transform=transformer_y,
    #                         title=f"{X_columns[i]} & {X_columns[j]} "
    #                         f"(R={self.normalised_sobols[n]:.3f})",
    #                         tikz_path=tikz_path,
    #                     )
    #                 )

    #         else:
    #             raise NotImplementedError("Higher-order plots are not yet supported.")

    #     # -------------------------------------------------------------------------
    #     # 4.  Save if requested
    #     # -------------------------------------------------------------------------
    #     if save_fig is not None:
    #         save_fig_list(fig_list, dirname=Path(save_fig))

    #     return fig_list


def _calculate_features(
    X: tf.Tensor, categorical_feature: List[int], binary_feature: List[int]
):
    """
    Calculate features index set
    :param X: input data
    :param categorical_feature: index of categorical features
    :param binary_feature: index of binary features
    :return:
        continuous_index, binary_index, categorical_index: list of indices for type of feature
        p0: list of probability measure for binary kernels, for continuous/categorical kernel, it is set to None
        p: list of probability measure for categorical kernels, for continuous/binary kernel, it is set to None
    """
    if binary_feature is None and categorical_feature is None:
        # all features are continuous
        p0 = None
        p = None
        continuous_index = list(range(X.shape[1]))
        binary_index = []
        categorical_index = []
    else:
        if binary_feature is not None and categorical_feature is not None:
            overlapping_set = set(binary_feature).intersection(categorical_feature)
            if len(overlapping_set) > 0:
                raise ValueError(f"Overlapping feature set {overlapping_set}")
        binary_index, categorical_index, continuous_index, p0, p = [], [], [], [], []
        for j in range(X.shape[1]):
            if binary_feature is not None and j in binary_feature:
                p0.append(1 - X[:, j].mean())
                p.append(None)
                binary_index.append(j)
            elif categorical_feature is not None and j in categorical_feature:
                p0.append(None)
                prob = []
                for jj in np.unique(X[:, j]):
                    prob.append(len(np.where(X[:, j] == jj)[0]) / len(X[:, j]))
                p.append(np.reshape(prob, (-1, 1)))
                assert np.abs(p[-1].sum() - 1) < 1e-6
                categorical_index.append(j)
            else:
                p.append(None)
                p0.append(None)
                continuous_index.append(j)
    print("indices of binary feature ", binary_index)
    print("indices of continuous feature ", continuous_index)
    print("indices of categorical feature ", categorical_index)

    return continuous_index, binary_index, categorical_index, p0, p


def estimate_one_dim_gmm(K: int, X: np.ndarray) -> MOGMeasure:
    """
    :param K: number of mixtures
    :param X: input data
    :return: estimated Gaussian mixture model on the data X
    """
    tf.debugging.assert_shapes([(X, ("N",))])
    assert K > 0
    gm = GaussianMixture(
        n_components=K, random_state=0, covariance_type="spherical"
    ).fit(X.reshape(-1, 1))
    assert np.allclose(gm.weights_.sum(), 1.0)
    assert gm.means_.shape == (K, 1)
    assert gm.covariances_.shape == (K,)
    assert gm.weights_.shape == (K,)
    return MOGMeasure(
        weights=gm.weights_, means=gm.means_.reshape(-1), variances=gm.covariances_
    )
