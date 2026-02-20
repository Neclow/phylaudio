# pylint: disable=invalid-name

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize



# -----------------------------
# Utilities
# -----------------------------
def nmf_fit(X, k, seed, max_iter=2000, loss="kullback-leibler"):
    """
    Fits NMF on nonnegative X.
    For binary data, KL (Poisson-like) often behaves nicely; Frobenius also works.
    """
    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="mu",
        beta_loss=loss,  # "kullback-leibler" or "frobenius"
        max_iter=max_iter,
        random_state=seed,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    W = model.fit_transform(X)
    H = model.components_
    # sklearn's reconstruction_err_ matches the chosen loss in a consistent way for comparisons
    err = model.reconstruction_err_
    return W, H, err


def match_components(Wa, Wb):
    """
    Match columns (components) of Wb to Wa by maximizing cosine similarity.
    Returns Wb with columns permuted to best match Wa.
    """
    # normalize columns to unit length for cosine similarity
    A = normalize(Wa, axis=0)
    B = normalize(Wb, axis=0)
    # similarity matrix (k x k)
    S = A.T @ B
    # Hungarian algorithm to maximize total similarity -> minimize negative similarity
    row_ind, col_ind = linear_sum_assignment(-S)
    return Wb[:, col_ind], S[row_ind, col_ind].mean()


def stability_score(W_list):
    """
    Average matched cosine similarity across all pairs of runs.
    Higher = more stable solutions for that K.
    """
    sims = []
    for i in range(len(W_list)):
        for j in range(i + 1, len(W_list)):
            _, sim = match_components(W_list[i], W_list[j])
            sims.append(sim)
    return float(np.mean(sims)) if sims else np.nan


def normalize_rows_to_proportions(W, eps=1e-12):
    row_sums = W.sum(axis=1, keepdims=True) + eps
    return W / row_sums


# -----------------------------
# Main sweep over K
# -----------------------------
def nmf_k_sweep(X, k_min=2, k_max=10, n_restarts=20, seed0=0, loss="kullback-leibler"):
    results = {}
    for k in range(k_min, k_max + 1):
        W_runs, H_runs, errs = [], [], []
        for r in range(n_restarts):
            W, H, err = nmf_fit(X, k, seed=seed0 + 1000 * k + r, loss=loss)
            W_runs.append(W)
            H_runs.append(H)
            errs.append(err)

        best_idx = int(np.argmin(errs))
        best = {
            "W": W_runs[best_idx],
            "H": H_runs[best_idx],
            "err": float(errs[best_idx]),
        }

        stab = stability_score(W_runs)

        results[k] = {
            "best": best,
            "errs": np.array(errs, dtype=float),
            "stability": float(stab),
        }
        print(
            f"K={k:2d}  best_err={best['err']:.4f}  mean_err={np.mean(errs):.4f}  stability={stab:.4f}"
        )
    return results


def choose_k(results, error_slack=0.02):
    """
    Pick K with highest stability among those whose best error is within
    (1 + error_slack) of the overall best error.
    """
    ks = sorted(results.keys())
    best_errors = np.array([results[k]["best"]["err"] for k in ks])
    stabilities = np.array([results[k]["stability"] for k in ks])

    global_best = best_errors.min()
    admissible = best_errors <= global_best * (1 + error_slack)

    if np.any(admissible):
        cand_ks = np.array(ks)[admissible]
        cand_stab = stabilities[admissible]
        k_star = int(cand_ks[np.argmax(cand_stab)])
    else:
        # fallback: max stability
        k_star = int(np.array(ks)[np.argmax(stabilities)])
    return k_star


def plot_k_diagnostics(results, k_star=None):
    ks = sorted(results.keys())
    best_err = [results[k]["best"]["err"] for k in ks]
    mean_err = [results[k]["errs"].mean() for k in ks]
    stab = [results[k]["stability"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(ks, best_err, "o-", label="Best reconstruction error")
    ax.plot(ks, mean_err, "s-", label="Mean reconstruction error")
    if k_star is not None:
        ax.axvline(k_star, ls="--", color="grey", label=f"K* = {k_star}")
    ax.set_xlabel("K")
    ax.set_ylabel("Error")
    ax.set_xticks(ks)
    ax.legend()

    ax = axes[1]
    ax.plot(ks, stab, "o-", color="tab:green")
    if k_star is not None:
        ax.axvline(k_star, ls="--", color="grey")
    ax.set_xlabel("K")
    ax.set_ylabel("Stability (avg matched cosine sim)")
    ax.set_xticks(ks)

    fig.tight_layout()
    return fig


def plot_structure(
    W,
    labels=None,
    sort_by_component=True,
    title=None,
    err=None,
    stability=None,
    k_val=None,
):
    """
    STRUCTURE-style stacked bars using normalized W.
    Returns the figure.
    """
    P = normalize_rows_to_proportions(W)
    n, k = P.shape

    order = np.arange(n)
    if sort_by_component:
        max_comp = np.argmax(P, axis=1)
        max_val = P[np.arange(n), max_comp]
        order = np.lexsort((-max_val, max_comp))

    P = P[order]
    if labels is not None:
        labels = np.array(labels)[order]

    fig, ax = plt.subplots(figsize=(max(14, n * 0.35), 5))
    bottom = np.zeros(n)
    x = np.arange(n)
    cmap_name = "tab20" if k > 8 else "Set2"
    cmap = mpl.colormaps[cmap_name]
    colors = cmap(np.linspace(0, 1, max(k, 2)))

    for j in range(k):
        ax.bar(x, P[:, j], bottom=bottom, width=1.0, color=colors[j], label=f"{j + 1}")
        bottom += P[:, j]

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Component proportion")
    ax.set_xlabel(
        "Languages (sorted by dominant component)" if sort_by_component else "Languages"
    )
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])

    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold")
    elif k_val is not None:
        parts = [f"K = {k_val}"]
        if err is not None:
            parts.append(f"err={err:.1f}")
        if stability is not None:
            parts.append(f"stab={stability:.3f}")
        ax.set_title(
            "  ".join(parts[:1])
            + ("  (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else ""),
            fontsize=12,
            fontweight="bold",
        )

    if k <= 20:
        ax.legend(loc="upper right", fontsize=6, ncol=2, title="Comp.")

    fig.tight_layout()
    return fig
