# normalizer_copula.py
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats

tfd, tfb = tfp.distributions, tfp.bijectors
DTYPE = gpflow.default_float()
SMALL = 1e-6


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _standardiser(x, eps):
    """Return Shift(-mean) then Scale(1/std)."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    std = np.std(x) + eps
    return [
        tfb.Shift(tf.constant(-mean, dtype=DTYPE), name="shift"),
        tfb.Scale(tf.constant(1.0 / std, dtype=DTYPE), name="scale"),
    ]


# ----------------------------------------------------------------------
# Normalizer with copula flow
# ----------------------------------------------------------------------
class NormalizerGeneralizedCopula(gpflow.base.Module):
    """
    D-dimensional normalizer with:
      (1) per-dim marginal flows (gaussianize each coord),
      (2) optional learned copula flow (MAF stack) to remove nonlinear dependence.

    The pipeline is fully invertible; you can transform and inverse-transform.
    You can also invert a *single variable* only through the marginal flow
    (before the copula), via inverse_marginal(...).

    Design for stable embedded-space attribution:
      • Deterministic MADE input orders (alternate L→R / R→L).
      • No random permutations.
      • (Optional) fixed internal permutations + final order-restoring perm,
        while keeping FINAL z-index order identical to input order.
    """

    def __init__(
        self,
        x,
        log=None,
        use_copula_flow: bool = True,
        n_maf: int = 3,
        hidden_units=(128, 128),
        activation=tf.nn.tanh,
        eps: float = SMALL,
        name="normalizer_copula",
        # --- New knobs for determinism / ordering ---
        deterministic: bool = True,
        use_fixed_permutations: bool = False,
        restore_final_order: bool = True,
        fixed_permutations=None,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # --- accept both NumPy and TF arrays ---
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        x = np.asarray(x, dtype=np.float64)
        self.x = x
        self.eps = eps

        D = x.shape[1]
        if log is None:
            log = [False] * D
        assert x.ndim == 2, "Expected input shape (N, D)."
        assert len(log) == D, "Length of `log` must match number of dimensions"

        self.use_copula_flow = use_copula_flow
        self.n_maf = int(n_maf)
        self.hidden_units = tuple(hidden_units)
        self.activation = activation

        # Determinism
        self.deterministic = deterministic
        if self.deterministic:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # ------------------------------------------------------------------
        # 1) Per-dimension 1D marginal flows
        # ------------------------------------------------------------------
        per_dim = []
        self._per_dim_offsets = [0.0] * D
        for d in range(D):
            xd = x[:, d]
            if log[d]:
                offset = float(np.min(xd) - 1.0 - eps)
                self._per_dim_offsets[d] = offset
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
            per_dim.append(chain)

        block = tfb.Blockwise(per_dim, block_sizes=[1] * D, name="per_dim_block")
        self._block = block  # z = block(x)

        # ------------------------------------------------------------------
        # 2) Optional learned copula flow (MAF stack) with stable ordering
        # ------------------------------------------------------------------
        pieces = [block]
        self._maf_stack = None
        self._internal_perms = []  # store any fixed internal permutations
        self._final_restore_perm = None

        if use_copula_flow and self.n_maf > 0:
            mafs = []

            # Helper to make a deterministic input order for MADE
            def _made_order(k):
                # Alternate L->R, R->L for depth
                return "left-to-right" if (k % 2 == 0) else "right-to-left"

            # Build MAF + (optional fixed perm) stack
            for k in range(self.n_maf):
                made = tfb.AutoregressiveNetwork(
                    params=2,
                    hidden_units=list(self.hidden_units),
                    activation=self.activation,
                    kernel_initializer="glorot_uniform",
                    input_order=_made_order(k),  # key: deterministic, no randomization
                    name=f"made_{k}",
                )
                maf = tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=made, name=f"maf_{k}"
                )
                mafs.append(maf)

                if use_fixed_permutations:
                    if fixed_permutations is None:
                        # Provide a simple deterministic pattern: identity, reverse, identity, reverse, ...
                        perm = (
                            list(range(D))
                            if (k % 2 == 0)
                            else list(range(D - 1, -1, -1))
                        )
                    else:
                        # Use user-supplied sequence (cycled if length < n_maf)
                        perm = list(fixed_permutations[k % len(fixed_permutations)])
                        assert (
                            len(perm) == D
                        ), "Each fixed permutation must have length D"
                    self._internal_perms.append(perm)
                    mafs.append(tfb.Permute(permutation=perm, name=f"perm_{k}"))

            # If we used any internal perms but want final z aligned to input order,
            # append a final restoring permutation that inverts the composed internal perms.
            if (
                use_fixed_permutations
                and restore_final_order
                and len(self._internal_perms) > 0
            ):
                comp = list(range(D))
                for p in self._internal_perms:
                    comp = [p[i] for i in comp]
                # compute inverse permutation
                inv = [0] * D
                for i, j in enumerate(comp):
                    inv[j] = i
                self._final_restore_perm = inv
                mafs.append(tfb.Permute(permutation=inv, name="restore_order"))

            self._maf_stack = tfb.Chain(mafs, name="copula_flow")
            pieces = [self._maf_stack] + pieces

        # Final bijector & base
        self.bijector = tfb.Chain(pieces, name="full_flow")
        self._base = tfd.MultivariateNormalDiag(
            loc=tf.zeros(D, DTYPE), scale_diag=tf.ones(D, DTYPE)
        )

    # ----------------------------------------------------------------------
    # Forward / inverse
    # ----------------------------------------------------------------------
    def forward(self, x):
        return self.bijector.forward(tf.cast(x, DTYPE))

    def inverse(self, y):
        return self.bijector.inverse(tf.cast(y, DTYPE))

    def log_det_jacobian(self, x):
        return self.bijector.forward_log_det_jacobian(tf.cast(x, DTYPE), event_ndims=1)

    def forward_marginals(self, x):
        return self._block.forward(tf.cast(x, DTYPE))

    def inverse_marginal(self, z_i, dim: int):
        z_i = tf.convert_to_tensor(z_i, dtype=DTYPE)
        return self._block.bijectors[dim].inverse(z_i)

    # ----------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------
    def KL_objective(self, x=None):
        if x is None:
            x = self.x
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        y = self.forward(x)
        return 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(y), axis=-1)
        ) - tf.reduce_mean(self.log_det_jacobian(x))

    def kstest(self, x=None):
        if x is None:
            x = self.x
        y = self.forward(x).numpy()
        res = []
        for d in range(y.shape[1]):
            s, p = stats.kstest(y[:, d], "norm")
            res.append((s, p))
            print(f"Dim {d}: KS stat = {s:.3f}, p = {p:.3g}")
        return res

    def pairwise_corr(self, x=None):
        if x is None:
            x = self.x
        y = self.forward(x).numpy()
        return np.corrcoef(y.T)

    # ----------------------------------------------------------------------
    # Training only the copula flow
    # ----------------------------------------------------------------------
    @tf.function
    def _nll_step(self, x_batch, optimizer):
        with tf.GradientTape() as tape:
            y = self.forward(x_batch)
            log_det = self.log_det_jacobian(x_batch)
            log_p = self._base.log_prob(y) + log_det
            nll = -tf.reduce_mean(log_p)
        vars_trainable = (
            self._maf_stack.trainable_variables if self._maf_stack is not None else []
        )
        grads = tape.gradient(nll, vars_trainable)
        optimizer.apply_gradients(zip(grads, vars_trainable))
        return nll

    def fit_flow(
        self, x_train=None, epochs=200, batch_size=1024, lr=1e-3, verbose=True, seed=0
    ):
        if not (self.use_copula_flow and self._maf_stack is not None):
            if verbose:
                print("Nothing to train: use_copula_flow is False or n_maf=0.")
            return
        if x_train is None:
            x_train = self.x
        x_train = tf.convert_to_tensor(x_train, dtype=DTYPE)
        n = tf.shape(x_train)[0]
        rng = tf.random.Generator.from_seed(seed)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        @tf.function
        def get_batch():
            idx = rng.uniform(shape=(batch_size,), maxval=n, dtype=tf.int32)
            return tf.gather(x_train, idx)

        for e in range(1, epochs + 1):
            xb = get_batch()
            nll = self._nll_step(xb, opt)
            if verbose and (e == 1 or e % max(1, epochs // 10) == 0 or e == epochs):
                print(
                    f"[{e:04d}/{epochs}] NLL: {float(nll):.4f} | KL proxy: {float(self.KL_objective(x_train)):.4f}"
                )


# ----------------------------------------------------------------------
# Independence diagnostics
# ----------------------------------------------------------------------
def _pdist_centered(x):
    x = np.asarray(x, float)
    n = x.shape[0]
    D = np.abs(x - x.T)
    A = D - D.mean(axis=0, keepdims=True) - D.mean(axis=1, keepdims=True) + D.mean()
    return A


def distance_correlation(x, y):
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float).reshape(-1, 1)
    A = _pdist_centered(x)
    B = _pdist_centered(y)
    dcov2_xy = (A * B).mean()
    dcov2_xx = (A * A).mean()
    dcov2_yy = (B * B).mean()
    denom = np.sqrt(dcov2_xx * dcov2_yy) + 1e-12
    return np.sqrt(max(dcov2_xy, 0.0)) / denom


def dcor_matrix(X):
    n, d = X.shape
    M = np.zeros((d, d), float)
    for i in range(d):
        M[i, i] = 1.0
        for j in range(i + 1, d):
            val = distance_correlation(X[:, i], X[:, j])
            M[i, j] = M[j, i] = val
    return M


def _rbf_kernel(z, sigma=None):
    z = np.asarray(z, float).reshape(-1, 1)
    D2 = (z - z.T) ** 2
    if sigma is None:
        dists = np.sqrt(np.maximum(D2[np.triu_indices_from(D2, 1)], 0.0))
        med = np.median(dists)
        sigma = med if med > 0 else np.std(z) + 1e-12
    return np.exp(-D2 / (2 * sigma**2))


def hsic_unbiased(x, y):
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float).reshape(-1, 1)
    n = x.shape[0]
    K = _rbf_kernel(x)
    L = _rbf_kernel(y)
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    term1 = (K * L).sum() / (n * (n - 3))
    term2 = (K.sum(axis=0) * L.sum(axis=0)).sum() / (n * (n - 3) * (n - 1))
    term3 = (K.sum() * L.sum()) / (n * (n - 1) * (n - 2) * (n - 3))
    hsic = term1 + term3 - 2 * term2
    return max(hsic, 0.0)


def hsic_perm_test(x, y, n_perm=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    obs = hsic_unbiased(x, y)
    n = len(x)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        val = hsic_unbiased(x, y[perm])
        count += val >= obs
    p = (count + 1) / (n_perm + 1)
    return obs, p


def independence_report(Z, check_hsic_topk=5, n_perm=200):
    M = dcor_matrix(Z)
    d = Z.shape[1]
    pairs = [(M[i, j], i, j) for i in range(d) for j in range(i + 1, d)]
    pairs.sort(reverse=True)
    top = pairs[:check_hsic_topk]
    hsic_results = []
    for _, i, j in top:
        hsic, p = hsic_perm_test(Z[:, i], Z[:, j], n_perm=n_perm)
        hsic_results.append({"i": i, "j": j, "hsic": hsic, "p": p, "dcor": M[i, j]})
    return M, hsic_results
