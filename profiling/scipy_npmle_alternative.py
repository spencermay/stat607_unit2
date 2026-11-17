import numpy as np
from scipy.optimize import minimize


def fit_mixture_weights(A, x0=None, ftol=1e-10, maxiter=1000, start_floor=1e-3):
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
        q = np.max(A @ w, 1e-200)
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