"""
Support-reduction NPMLE solver (production copy).

This is the fixed-grid, fully corrective active-set NPMLE solver used by the
production joint-NPMLE fit in stats_methods.get_joint_mixing_distribution_npmle.
It replaces the earlier cvxpy convex solve. Two pieces live here:

  - fit_mixture_weights: an SLSQP solve for the mixture weights on a fixed
    likelihood matrix A (the restricted/inner solve).
  - fit_mixture_support_reduction: the fixed-grid active-set loop that grows
    the support one atom at a time, re-optimizing weights via
    fit_mixture_weights.

The algorithms are copied verbatim from the profiling/ benchmark versions
(profiling/scipy_npmle_alternative.py and
profiling/support_reduction_npmle_alternative.py). The profiling/ copies remain
for the benchmark harness, which imports FROM src; this module is the
production home so production never depends on profiling/. The only intended
difference from the benchmark copy is start_floor's default below.
"""

import numpy as np
from scipy.optimize import minimize


def fit_mixture_weights(A, x0=None, ftol=1e-10, maxiter=1000, start_floor=1e-5):
    """Optimize mixture weights on a fixed likelihood matrix.

    start_floor : float
        Minimum value given to each weight in the starting point before the
        SLSQP solve (then renormalized to the simplex). This keeps x0 strictly
        interior. A start exactly on a simplex vertex -- which the support-
        reduction caller produces by appending a new atom at weight 0 -- makes
        q = A @ x0 have near-zero entries, so the objective gradient
        -A.T @ (1/q) / n explodes and SLSQP's first QP step reports
        "Inequality constraints incompatible" and fails. Flooring the start off
        the vertex avoids that. Set to 0 to disable.
    """
    A = np.asarray(A, dtype=float)
    n, p = A.shape

    if (
        n == 0 or p == 0
        or not np.all(np.isfinite(A))
        or np.any(A < 0)
        or np.any(A.max(axis=1) <= 0)
    ):
        raise ValueError("A must be finite, nonnegative, and positive in each row.")

    if x0 is None:
        x0 = np.full(p, 1.0 / p)
    else:
        x0 = np.asarray(x0, dtype=float).copy()
        if x0.shape != (p,) or np.any(x0 < 0) or x0.sum() <= 0:
            raise ValueError("Invalid initial weights.")
        x0 /= x0.sum()

    # Nudge the start off any simplex vertex/face so SLSQP does not begin where
    # the log-likelihood gradient is numerically singular (see start_floor).
    if start_floor > 0:
        x0 = np.maximum(x0, start_floor)
        x0 /= x0.sum()

    def objective(w):
        q = np.maximum(A @ w, 1e-200)
        if np.any(q <= 0):
            return np.inf, np.zeros(p)
        value = -np.mean(np.log(q))
        gradient = -(A.T @ (1.0 / q)) / n
        return value, gradient

    return minimize(
        objective,
        x0,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * p,
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1.0,
            "jac": lambda w: np.ones(p),
        },
        options={"ftol": ftol, "maxiter": maxiter},
    )


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
        q = np.maximum(A[:, support] @ w, 1e-200)
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
