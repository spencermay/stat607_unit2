import numpy as np
from scipy.optimize import minimize


def fit_mixture_weights(A, x0=None, ftol=1e-10, maxiter=1000):
    """Optimize mixture weights on a fixed likelihood matrix."""
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

    def objective(w):
        q = A @ w
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
    
"""
Use it directly:
result = fit_mixture_weights(A_joint)

if not result.success:
    raise RuntimeError(result.message)

weights = np.maximum(result.x, 0)
weights /= weights.sum()

q = A_joint @ weights
gap = np.max(A_joint.T @ (1.0 / q)) / len(q) - 1.0

For feasible weights, gap is an upper bound on the remaining improvement in average log likelihood, up to numerical error.
"""