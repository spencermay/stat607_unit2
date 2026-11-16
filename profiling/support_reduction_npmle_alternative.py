"""
Short support-reduction NPMLE implementation.

This is a fixed-grid, fully corrective active-set implementation:
  - Search every atom on the coarse grid.
  - Add the atom with the largest score.
  - Optimize all current support weights (restricted SLSQP solve).
  - Remove sufficiently small weights.
It does not interpolate or optimize atom locations continuously. Its
computational benefit comes from solving smaller weight problems; the coarse
grid is the naive atom search.

Usage sketch (an adapter that mirrors get_joint_mixing_distribution_npmle and
turns the full-grid weights into a GaussianMixture lives in
speed_comparison_convex_solvers.py)::

    fit = fit_mixture_support_reduction(A_joint)
    print(fit["status"])
    print("Active atoms:", len(fit["support"]))
    print("Average-loglik gap:", fit["gap"])
    weights = fit["weights"]
    keep = weights > 0
    GaussianMixture(
        MixJointMeanVariance(alpha_atoms[keep], theta_atoms[keep], weights[keep])
    )
"""

import numpy as np

from scipy_npmle_alternative import fit_mixture_weights


def fit_mixture_support_reduction(
    A,
    gap_tol=1e-6,
    drop_tol=1e-12,
    max_steps=200,
    inner_ftol=1e-11,
    inner_maxiter=1000,
):
    """
    Fixed-grid support reduction using SLSQP for restricted weight fits.

    Returns full-grid weights and diagnostics.
    gap bounds the remaining average-log-likelihood improvement
    on this fixed grid.
    """
    A = np.asarray(A, dtype=float)
    n, p = A.shape

    if (
        n == 0 or p == 0
        or not np.all(np.isfinite(A))
        or np.any(A < 0)
        or np.any(A.max(axis=1) <= 0)
    ):
        raise ValueError("Invalid likelihood matrix.")

    # Start with the best single atom having positive likelihood in every row.
    # This also avoids starting with an undefined log likelihood.
    eligible = np.flatnonzero(np.all(A > 0, axis=0))
    if eligible.size == 0:
        raise ValueError(
            "No atom has positive likelihood in every row. "
            "Use more stable likelihood evaluation or a multi-atom initializer."
        )

    scores = np.mean(np.log(A[:, eligible]), axis=0)
    support = np.array([eligible[np.argmax(scores)]])
    w = np.array([1.0])

    status = "maximum steps reached"
    steps = 0

    for step in range(max_steps):
        steps = step + 1

        # Fully correct the weights on the current support.
        result = fit_mixture_weights(
            A[:, support],
            x0=w,
            ftol=inner_ftol,
            maxiter=inner_maxiter,
        )

        if not result.success:
            status = f"restricted solve failed: {result.message}"
            break

        w = np.maximum(result.x, 0)
        w /= w.sum()

        # Numerical support reduction.
        keep = w > drop_tol
        keep[np.argmax(w)] = True
        support, w = support[keep], w[keep]
        w /= w.sum()

        # Naive search over the whole coarse grid.
        q = A[:, support] @ w
        g = (A.T @ (1.0 / q)) / n
        j = int(np.argmax(g))
        gap = max(0.0, float(g[j] - 1.0))

        if gap <= gap_tol:
            status = "converged"
            break

        if j in support:
            status = (
                "restricted solve insufficiently accurate; "
                "largest improving score is already in the support"
            )
            break

        # Add the atom at zero weight; the next solve chooses its mass.
        # Do not add on the final iteration, when it would remain unoptimized.
        if step + 1 < max_steps:
            support = np.append(support, j)
            w = np.append(w, 0.0)

    full_weights = np.zeros(p)
    full_weights[support] = w

    q = A @ full_weights
    gap = max(0.0, float(np.max(A.T @ (1.0 / q)) / n - 1.0))

    return {
        "weights": full_weights,
        "support": support,
        "gap": gap,
        "mean_loglik": float(np.mean(np.log(q))),
        "steps": steps,
        "success": gap <= gap_tol,
        "status": status,
    }
