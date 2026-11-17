"""
Throwaway prototype (not wired into src/ or profiling/): measure two candidate
fixes for the support-reduction NPMLE inner weight solve at a 40x40 grid.

Baselines:
  - original: full-grid cvxpy conic solve (get_joint_mixing_distribution_npmle).
  - slsqp-current: the shipped support reduction (SLSQP, zero-weight warm start)
    -- the one that fails at 40x40.

Candidate fixes (inner weight solve on the active set only):
  - option3-slsqp-floored: same SLSQP, but any zero-weight warm-start entry is
    floored to a small positive value and renormalized before the solve, so it
    never starts exactly at a simplex vertex.
  - option5-cvxpy-param: inner solve done with a PARAMETRIZED cvxpy problem
    (one Problem per support size, reused across steps) solved with warm_start,
    to avoid re-canonicalization overhead on every step.

All active-set variants share the exact grid/A_joint the default builds, so the
accuracy distance to the original is apples-to-apples.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "profiling")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import cvxpy as cp
from scipy.stats import norm, gamma
from scipy.optimize import minimize

import speed_comparison_convex_solvers as S
from stats_methods import get_joint_mixing_distribution_npmle

GRID = 40
SEED = 0


def build_A(ybar, Sv, m, g):
    r = (m - 1) / 2
    ma = np.linspace(min(ybar), max(ybar), g)
    va = np.linspace(min(Sv / r), max(Sv / r), g)
    al = np.repeat(ma, g)
    th = np.tile(va, g)
    mp = norm.pdf(ybar[:, None], loc=al, scale=np.sqrt(th / m))
    vp = gamma.pdf(Sv[:, None], a=r, scale=2 * th)
    return mp * vp, al, th


# --- inner weight solvers (signature: (Asub, x0) -> weights) ---------------

def inner_slsqp(Asub, x0, floor=0.0, ftol=1e-11, maxiter=1000):
    n, p = Asub.shape
    x0 = np.asarray(x0, float).copy()
    if floor > 0:
        x0 = np.maximum(x0, floor)
    x0 /= x0.sum()

    def obj(w):
        q = Asub @ w
        if np.any(q <= 0):
            return np.inf, np.zeros(p)
        return -np.mean(np.log(q)), -(Asub.T @ (1.0 / q)) / n

    res = minimize(
        obj, x0, jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * p,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                     "jac": lambda w: np.ones(p)},
        options={"ftol": ftol, "maxiter": maxiter},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")
    w = np.maximum(res.x, 0)
    return w / w.sum()


class CvxpyInnerCache:
    """Parametrized cvxpy inner solve, one reusable Problem per support size."""
    def __init__(self, solver="CLARABEL"):
        self._by_size = {}
        self.solver = solver

    def _problem(self, n, p):
        key = (n, p)
        prob = self._by_size.get(key)
        if prob is None:
            Apar = cp.Parameter((n, p), nonneg=True)
            w = cp.Variable(p, nonneg=True)
            objective = cp.Maximize(cp.sum(cp.log(Apar @ w)))
            prob = cp.Problem(objective, [cp.sum(w) == 1])
            self._by_size[key] = (prob, Apar, w)
        return self._by_size[key]

    def solve(self, Asub, x0=None):
        n, p = Asub.shape
        prob, Apar, w = self._problem(n, p)
        Apar.value = Asub
        if x0 is not None:
            w.value = np.asarray(x0, float)
        prob.solve(solver=self.solver, warm_start=True)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"cvxpy inner status {prob.status!r}")
        val = np.maximum(np.asarray(w.value).ravel(), 0)
        return val / val.sum()


# --- support reduction loop parametrized by an inner solver ----------------

def support_reduction(A, inner, gap_tol=1e-6, drop_tol=1e-12, max_steps=200):
    n, p = A.shape
    eligible = np.flatnonzero(np.all(A > 0, axis=0))
    scores = np.mean(np.log(A[:, eligible]), axis=0)
    support = np.array([eligible[np.argmax(scores)]])
    w = np.array([1.0])
    status, steps = "maximum steps reached", 0

    for step in range(max_steps):
        steps = step + 1
        try:
            w = inner(A[:, support], w)
        except RuntimeError as exc:
            status = f"inner solve failed: {exc}"
            break
        keep = w > drop_tol
        keep[np.argmax(w)] = True
        support, w = support[keep], w[keep]
        w /= w.sum()

        q = A[:, support] @ w
        g = (A.T @ (1.0 / q)) / n
        j = int(np.argmax(g))
        gap = max(0.0, float(g[j] - 1.0))
        if gap <= gap_tol:
            status = "converged"
            break
        if j in support:
            status = "largest improving score already in support"
            break
        if step + 1 < max_steps:
            support = np.append(support, j)
            w = np.append(w, 0.0)

    full = np.zeros(p)
    full[support] = w
    q = A @ full
    return {
        "support_idx": support, "weights": full,
        "gap": max(0.0, float(np.max(A.T @ (1.0 / q)) / n - 1.0)),
        "mean_loglik": float(np.mean(np.log(q))),
        "steps": steps, "status": status, "n_support": len(support),
    }


def atoms_weights(full, al, th):
    keep = full > 0
    sup = np.column_stack([al[keep], th[keep]])
    return sup, full[keep]


def main():
    ybar, Sv, m = S.make_data(seed=SEED)
    A, al, th = build_A(ybar, Sv, m, GRID)

    # Reference: original full-grid cvxpy solve, same grid.
    t0 = time.perf_counter()
    ref = get_joint_mixing_distribution_npmle(ybar, Sv, m, gridsize_mean=GRID, gridsize_S=GRID)
    ref_t = time.perf_counter() - t0
    ref_sup, ref_w = S._atoms_and_weights(ref)
    # original mean loglik on this A for a like-for-like likelihood comparison
    rsup_full = np.zeros(A.shape[1])
    # map ref atoms back is awkward; instead report ref mean loglik via its own pmf:
    ref_ll = None  # computed per-variant below against A isn't needed for ref

    variants = {
        "slsqp-current": lambda Asub, x0: inner_slsqp(Asub, x0, floor=0.0),
        "option3-slsqp-floored": lambda Asub, x0: inner_slsqp(Asub, x0, floor=1e-3),
        "option5-cvxpy-param": CvxpyInnerCache().solve,
    }

    print(f"\n40x40 grid ({A.shape[1]} atoms), seed {SEED}")
    print("-" * 92)
    print(f"{'variant':<26}{'time (s)':>12}{'n_supp':>8}{'steps':>7}{'mean_loglik':>14}{'W2^2 vs orig':>16}  status")
    print("-" * 92)
    print(f"{'original (full cvxpy)':<26}{ref_t:>12.4f}{len(ref_w):>8}{'-':>7}{'-':>14}{0.0:>16.3e}  reference")

    for name, inner in variants.items():
        t0 = time.perf_counter()
        try:
            out = support_reduction(A, inner)
            elapsed = time.perf_counter() - t0
            sup, w = atoms_weights(out["weights"], al, th)
            dist = S._distance_between(sup, w, ref_sup, ref_w)
            print(f"{name:<26}{elapsed:>12.4f}{out['n_support']:>8}{out['steps']:>7}"
                  f"{out['mean_loglik']:>14.6f}{dist:>16.3e}  {out['status']}")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            print(f"{name:<26}{elapsed:>12.4f}{'-':>8}{'-':>7}{'-':>14}{'-':>16}  CRASH: {exc}")
    print("-" * 92)


if __name__ == "__main__":
    main()
