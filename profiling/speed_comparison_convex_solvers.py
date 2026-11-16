"""
Upon consulting https://www.cvxpy.org/tutorial/solvers/index.html, it appears
I have more solvers available that may be better suited to the tasks I want to
solve.

For instance I have these solvers but I've only been using the first 2:
>>> cp.installed_solvers()
['CLARABEL', 'SCS', 'SCIPY', 'HIGHS', 'OSQP']
"""

import os
import sys
import time
import multiprocessing as mp

SOLVE_TIMEOUT_S = 15.0

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for filepath in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "tests")):
    if filepath not in sys.path:
        sys.path.insert(0, filepath)

import numpy as np
import cvxpy as cp
from scipy.stats import norm, gamma

from stats_methods import get_joint_mixing_distribution_npmle, _prune_pmf
from data_generating_processes import GaussianMixture, MixJointMeanVariance
from tests.test_stats_methods import make_correlated_dgp, M

# The two hand-written alternatives to the cvxpy convex solve. They take a
# fixed likelihood matrix A and return mixture weights; the adapters below wrap
# them with the reference's exact grid/A_joint construction so their output is
# comparable to get_joint_mixing_distribution_npmle.
from scipy_npmle_alternative import fit_mixture_weights
from support_reduction_npmle_alternative import fit_mixture_support_reduction

# Shared grid for the apples-to-apples alternatives comparison.
#
# The reference default is gridsize_mean = gridsize_S = 40 (1600 joint atoms).
# The scipy SLSQP weight solve optimizes one variable per atom under a dense
# equality + box constraint set, so 1600 variables is far too slow (it blows
# past the 15s budget). We therefore run the alternatives comparison on a
# smaller SHARED grid and fit the reference solver on that SAME grid, so the
# accuracy distance stays meaningful (apples-to-apples). 20x20 = 400 atoms
# keeps SLSQP tractable while staying a genuine joint NPMLE. Change both values
# together -- never score solvers fit on different grids against each other.
ALT_GRIDSIZE_MEAN = 20
ALT_GRIDSIZE_S = 20

try:
    import ot  # POT: python optimal transport
    _HAVE_OT = True
except ImportError:
    _HAVE_OT = False


def _build_joint_grid_and_A(ybar, S, m, gridsize_mean, gridsize_S):
    """Build the joint (alpha, theta) atom grid and likelihood matrix A_joint
    EXACTLY as get_joint_mixing_distribution_npmle does, so the alternatives
    solve the identical NPMLE problem and share its atom layout before pruning.

    Returns (alpha_atoms, theta_atoms, A_joint) with atom index p = k*gridsize_S
    + l mapping to (mean_axis[k], variance_axis[l]) (C-order)."""
    ybar = np.asarray(ybar, dtype=float)
    S = np.asarray(S, dtype=float)
    r = (m - 1) / 2

    mean_axis = np.linspace(min(ybar), max(ybar), gridsize_mean)
    variance_axis = np.linspace(min(S / r), max(S / r), gridsize_S)

    alpha_atoms = np.repeat(mean_axis, gridsize_S)       # (P,)
    theta_atoms = np.tile(variance_axis, gridsize_mean)  # (P,)

    ybar_var = theta_atoms / m
    mean_part = norm.pdf(ybar[:, None], loc=alpha_atoms, scale=np.sqrt(ybar_var))
    var_part = gamma.pdf(S[:, None], a=r, scale=2 * theta_atoms)
    A_joint = mean_part * var_part                        # (N, P)
    return alpha_atoms, theta_atoms, A_joint


def _mixture_from_weights(alpha_atoms, theta_atoms, weights, prune_tol):
    """Clip/normalize full-grid weights and prune negligible atoms exactly as
    get_joint_mixing_distribution_npmle does, returning a GaussianMixture."""
    weights = np.clip(np.asarray(weights, dtype=float).ravel(), 0, None)
    weights = weights / weights.sum()

    keep = weights > prune_tol * weights.max() if prune_tol > 0 else slice(None)
    alpha_keep = alpha_atoms[keep]
    theta_keep = theta_atoms[keep]
    weights_keep = weights[keep]
    weights_keep = weights_keep / weights_keep.sum()

    return GaussianMixture(
        MixJointMeanVariance(alpha_keep, theta_keep, weights_keep)
    )


def fit_scipy_slsqp(ybar, S, m, gridsize_mean=ALT_GRIDSIZE_MEAN,
                    gridsize_S=ALT_GRIDSIZE_S, prune_tol=1e-4):
    """Adapter: same interface as get_joint_mixing_distribution_npmle, but the
    joint-simplex weight solve is done by scipy SLSQP (fit_mixture_weights)."""
    alpha_atoms, theta_atoms, A_joint = _build_joint_grid_and_A(
        ybar, S, m, gridsize_mean, gridsize_S
    )
    result = fit_mixture_weights(A_joint)
    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")
    return _mixture_from_weights(alpha_atoms, theta_atoms, result.x, prune_tol)


def fit_support_reduction(ybar, S, m, gridsize_mean=ALT_GRIDSIZE_MEAN,
                          gridsize_S=ALT_GRIDSIZE_S, prune_tol=1e-4):
    """Adapter: same interface as get_joint_mixing_distribution_npmle, but the
    weights come from the fixed-grid support-reduction active-set method."""
    alpha_atoms, theta_atoms, A_joint = _build_joint_grid_and_A(
        ybar, S, m, gridsize_mean, gridsize_S
    )
    fit = fit_mixture_support_reduction(A_joint)
    return _mixture_from_weights(
        alpha_atoms, theta_atoms, fit["weights"], prune_tol
    )


def make_data(seed=0, n=400, m=M):
    """Draw (ybar, S) from the correlated joint DGP in make_correlated_dgp().

    make_correlated_dgp() returns the mixing distribution, not data, so we
    sample units from it the same way tests/test_stats_methods._draw_correlated
    does: n units, each with m Gaussian draws, reduced to the sufficient
    statistics (ybar_i, S_i)."""
    dgp, _means, _variances, _weights = make_correlated_dgp()
    rng = np.random.default_rng(seed)
    y, _alpha, _theta = dgp.generate(rng, n, m, return_latents=True)
    ybar = y.mean(axis=1)
    S = ((y - ybar[:, None]) ** 2).sum(axis=1)
    return ybar, S, m

def make_single_solver_wrapper(solver):
    def _wrapper(problem):
        problem.solve(solver=solver)
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise cp.error.SolverError(
                f"{solver} returned status {problem.status!r}"
            )
    return _wrapper


def _atoms_and_weights(gm):
    means, variances, weights = gm.mixing_distribution.mean_and_variance_joint_approx_pmf()
    support = np.column_stack([np.asarray(means, float), np.asarray(variances, float)])
    return support, np.asarray(weights, float)


def _distance_between(sup_a, w_a, sup_b, w_b):
    """Wasserstein-2 (squared) distance between two joint mixtures given as
    (support, weights) arrays.

    Uses POT when available. Without POT, falls back to comparing the first two
    moments (mean vector and covariance of (alpha, theta)) -- coarser, but it
    still flags a solver that converged to a clearly different distribution.
    """
    if _HAVE_OT:
        cost = ot.dist(sup_a, sup_b, metric="sqeuclidean")
        return float(ot.emd2(w_a, w_b, cost))

    # Fallback: distance between (mean, cov) summaries of the two mixtures.
    def summary(sup, w):
        mean = w @ sup
        centered = sup - mean
        cov = (w[:, None] * centered).T @ centered
        return mean, cov

    m_a, c_a = summary(sup_a, w_a)
    m_b, c_b = summary(sup_b, w_b)
    return float(np.sum((m_a - m_b) ** 2) + np.sum((c_a - c_b) ** 2))


def _solve_worker(queue, solver, ybar, S, m, ref_sup, ref_w, grid=None):
    """Run one fit with `solver`, score it against the reference, and put
    (elapsed_seconds, distance) on the queue. On failure, put ("error", repr).

    `grid` optionally overrides (gridsize_mean, gridsize_S); default (None)
    keeps get_joint_mixing_distribution_npmle's own 40x40 default."""
    try:
        wrapper = make_single_solver_wrapper(solver)
        grid_kwargs = {}
        if grid is not None:
            grid_kwargs = {"gridsize_mean": grid[0], "gridsize_S": grid[1]}
        t0 = time.perf_counter()
        fit = get_joint_mixing_distribution_npmle(
            ybar, S, m, cvxpy_solve_wrapper=wrapper, **grid_kwargs
        )
        elapsed = time.perf_counter() - t0
        sup, w = _atoms_and_weights(fit)
        distance = _distance_between(sup, w, ref_sup, ref_w)
        queue.put((elapsed, distance))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _solve_with_timeout(solver, ybar, S, m, ref_sup, ref_w, timeout, grid=None):
    """Return (elapsed, distance) for one timed+scored solve, or raise:
    TimeoutError if it exceeds `timeout`, RuntimeError if the solve failed.

    `grid` optionally overrides the solver's (gridsize_mean, gridsize_S)."""
    ctx = mp.get_context("spawn")  # spawn: clean process, no inherited solver state
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_solve_worker,
        args=(queue, solver, ybar, S, m, ref_sup, ref_w, grid),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"exceeded {timeout:g}s")

    if queue.empty():
        raise RuntimeError(f"worker exited without a result (exit code {proc.exitcode})")

    payload = queue.get()
    if payload[0] == "error":
        raise RuntimeError(payload[1])
    return payload  # (elapsed, distance)


# Fit callables for the alternatives comparison, keyed by a picklable name so a
# spawned worker can look them up (spawn re-imports this module).
_ALT_FITS = {
    "scipy-SLSQP": fit_scipy_slsqp,
    "support-reduction": fit_support_reduction,
}


def _fit_worker(queue, fit_key, grid, ybar, S, m, ref_sup, ref_w):
    """Run one alternative fit on the shared grid, score it against the
    reference, and put (elapsed, distance) on the queue; on failure put
    ("error", repr)."""
    try:
        gridsize_mean, gridsize_S = grid
        fit_fn = _ALT_FITS[fit_key]
        t0 = time.perf_counter()
        fit = fit_fn(ybar, S, m, gridsize_mean=gridsize_mean, gridsize_S=gridsize_S)
        elapsed = time.perf_counter() - t0
        sup, w = _atoms_and_weights(fit)
        distance = _distance_between(sup, w, ref_sup, ref_w)
        queue.put((elapsed, distance))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _fit_with_timeout(fit_key, grid, ybar, S, m, ref_sup, ref_w, timeout):
    """Timeout-guarded variant of _solve_with_timeout for the alternative fit
    callables. Same return/raise contract."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_fit_worker,
        args=(queue, fit_key, grid, ybar, S, m, ref_sup, ref_w),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"exceeded {timeout:g}s")

    if queue.empty():
        raise RuntimeError(f"worker exited without a result (exit code {proc.exitcode})")

    payload = queue.get()
    if payload[0] == "error":
        raise RuntimeError(payload[1])
    return payload  # (elapsed, distance)


def compare_solvers(seed=0, repeats=3, accuracy_atol=1e-4, timeout=SOLVE_TIMEOUT_S):
    """Time every installed solver on the same joint-NPMLE problem and check
    each result for accuracy against a reference fit.

    Parameters
    ----------
    repeats : int
        Number of timed runs per solver; the reported time is the best (min)
        wall-clock time, which is the standard way to reduce noise.
    accuracy_atol : float
        A solver is "accurate" if its distribution distance to the reference is
        <= this. Set loose enough (1e-4) to absorb the small distribution
        differences from solver-to-solver numerical noise and support pruning,
        while still flagging a solver that converged to a genuinely different
        distribution.
    timeout : float
        Per-solve wall-clock budget in seconds. A solve exceeding it is killed
        and recorded as TIMEOUT instead of blocking the comparison.
    """
    ybar, S, m = make_data(seed=seed)

    ref_sup, ref_w = _atoms_and_weights(get_joint_mixing_distribution_npmle(ybar, S, m))

    results = []  # (solver, best_time_or_None, distance_or_None, status)
    for solver in cp.installed_solvers():
        best_time = None
        distance = None
        status = "ok"
        for _ in range(repeats):
            try:
                elapsed, distance = _solve_with_timeout(
                    solver, ybar, S, m, ref_sup, ref_w, timeout
                )
            except TimeoutError as exc:
                status = f"TIMEOUT ({exc})"
                best_time = None
                distance = None
                break
            except RuntimeError as exc:
                status = f"FAILED: {exc}"
                best_time = None
                distance = None
                break
            best_time = elapsed if best_time is None else min(best_time, elapsed)

        if status == "ok" and distance is not None:
            if not np.isclose(distance, 0.0, rtol=0, atol=accuracy_atol):
                status = "INACCURATE"

        results.append((solver, best_time, distance, status))

    _print_report(results, accuracy_atol)
    return results


def _print_report(results, accuracy_atol):
    metric = "W2^2 (POT)" if _HAVE_OT else "moment-dist (no POT)"
    print()
    print(f"Joint-NPMLE solver comparison  (accuracy metric: {metric}, atol={accuracy_atol:g})")
    if not _HAVE_OT:
        print("  NOTE: POT ('ot') not installed -- using the coarse moment-summary "
              "fallback; install POT for the true transport-distance check.")
    print("-" * 72)
    print(f"{'solver':<12}{'best time (s)':>16}{'accuracy dist':>18}{'  status':<8}")
    print("-" * 72)

    # Sort: accurate solvers first, fastest at the top; failures/inaccurate last.
    def sort_key(row):
        _solver, t, _d, st = row
        ok = st == "ok"
        return (0 if ok else 1, t if t is not None else float("inf"))

    for solver, t, dist, st in sorted(results, key=sort_key):
        time_str = f"{t:.4f}" if t is not None else "--"
        dist_str = f"{dist:.3e}" if dist is not None else "--"
        print(f"{solver:<12}{time_str:>16}{dist_str:>18}  {st}")
    print("-" * 72)

    accurate = [r for r in results if r[3] == "ok" and r[1] is not None]
    if accurate:
        winner = min(accurate, key=lambda r: r[1])
        print(f"Fastest accurate solver: {winner[0]}  ({winner[1]:.4f} s)")
    else:
        print("No solver produced an accurate solution.")


def compare_alternatives(seed=0, repeats=1, accuracy_atol=1e-4,
                         timeout=SOLVE_TIMEOUT_S,
                         grid=(ALT_GRIDSIZE_MEAN, ALT_GRIDSIZE_S)):
    """Time the default cvxpy solver against the two hand-written alternatives
    (scipy SLSQP and support reduction) on one joint-NPMLE problem.

    All three are fit on the SAME grid so the accuracy distances are comparable
    (apples-to-apples). The reference baseline is the default solver
    (CLARABEL->SCS pipeline via get_joint_mixing_distribution_npmle) fit on that
    same grid; every fit runs in a spawned worker under the shared `timeout`.

    repeats defaults to 1 here: the scipy SLSQP solve is expensive, so a single
    timed run keeps the comparison quick. Raise it for a less noisy time.
    """
    ybar, S, m = make_data(seed=seed)
    gridsize_mean, gridsize_S = grid

    # Reference baseline fit on the SHARED grid (not the 40x40 default), using
    # the default CLARABEL->SCS pipeline. Scoring against a baseline fit on a
    # different grid would make the accuracy distances meaningless.
    ref_sup, ref_w = _atoms_and_weights(
        get_joint_mixing_distribution_npmle(
            ybar, S, m, gridsize_mean=gridsize_mean, gridsize_S=gridsize_S
        )
    )

    # Entries: (label, kind, key). kind "solver" uses the default cvxpy
    # pipeline; kind "fit" uses one of the alternative fit callables.
    entries = [
        ("default (CLARABEL->SCS)", "solver", None),
        ("scipy-SLSQP", "fit", "scipy-SLSQP"),
        ("support-reduction", "fit", "support-reduction"),
    ]

    results = []  # (label, best_time_or_None, distance_or_None, status)
    for label, kind, key in entries:
        best_time = None
        distance = None
        status = "ok"
        for _ in range(repeats):
            try:
                if kind == "solver":
                    # Default pipeline: try CLARABEL then SCS, same as the
                    # library default, by using the module's own wrapper.
                    elapsed, distance = _solve_default_with_timeout(
                        ybar, S, m, ref_sup, ref_w, timeout, grid
                    )
                else:
                    elapsed, distance = _fit_with_timeout(
                        key, grid, ybar, S, m, ref_sup, ref_w, timeout
                    )
            except TimeoutError as exc:
                status = f"TIMEOUT ({exc})"
                best_time = None
                distance = None
                break
            except RuntimeError as exc:
                status = f"FAILED: {exc}"
                best_time = None
                distance = None
                break
            best_time = elapsed if best_time is None else min(best_time, elapsed)

        if status == "ok" and distance is not None:
            if not np.isclose(distance, 0.0, rtol=0, atol=accuracy_atol):
                status = "INACCURATE"

        results.append((label, best_time, distance, status))

    _print_alt_report(results, accuracy_atol, grid)
    return results


def _default_pipeline_worker(queue, ybar, S, m, ref_sup, ref_w, grid):
    """Worker that fits with the library-default CLARABEL->SCS pipeline on the
    shared grid and scores it against the reference."""
    try:
        from stats_methods import _solve_npmle_problem
        gridsize_mean, gridsize_S = grid
        t0 = time.perf_counter()
        fit = get_joint_mixing_distribution_npmle(
            ybar, S, m,
            gridsize_mean=gridsize_mean, gridsize_S=gridsize_S,
            cvxpy_solve_wrapper=_solve_npmle_problem,
        )
        elapsed = time.perf_counter() - t0
        sup, w = _atoms_and_weights(fit)
        distance = _distance_between(sup, w, ref_sup, ref_w)
        queue.put((elapsed, distance))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _solve_default_with_timeout(ybar, S, m, ref_sup, ref_w, timeout, grid):
    """Timeout-guarded fit with the default CLARABEL->SCS pipeline."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_default_pipeline_worker,
        args=(queue, ybar, S, m, ref_sup, ref_w, grid),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"exceeded {timeout:g}s")

    if queue.empty():
        raise RuntimeError(f"worker exited without a result (exit code {proc.exitcode})")

    payload = queue.get()
    if payload[0] == "error":
        raise RuntimeError(payload[1])
    return payload


def _print_alt_report(results, accuracy_atol, grid):
    metric = "W2^2 (POT)" if _HAVE_OT else "moment-dist (no POT)"
    print()
    print(
        f"Alternatives comparison on shared {grid[0]}x{grid[1]} grid "
        f"({grid[0] * grid[1]} atoms)  "
        f"(accuracy metric: {metric}, atol={accuracy_atol:g})"
    )
    if not _HAVE_OT:
        print("  NOTE: POT ('ot') not installed -- using the coarse moment-summary "
              "fallback; install POT for the true transport-distance check.")
    print("-" * 78)
    print(f"{'method':<26}{'best time (s)':>16}{'accuracy dist':>18}{'  status':<8}")
    print("-" * 78)

    def sort_key(row):
        _label, t, _d, st = row
        ok = st == "ok"
        return (0 if ok else 1, t if t is not None else float("inf"))

    for label, t, dist, st in sorted(results, key=sort_key):
        time_str = f"{t:.4f}" if t is not None else "--"
        dist_str = f"{dist:.3e}" if dist is not None else "--"
        print(f"{label:<26}{time_str:>16}{dist_str:>18}  {st}")
    print("-" * 78)


if __name__ == "__main__":
    compare_solvers()
    compare_alternatives()
