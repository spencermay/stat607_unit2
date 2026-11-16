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

from stats_methods import get_joint_mixing_distribution_npmle
from tests.test_stats_methods import make_correlated_dgp, M

try:
    import ot  # POT: python optimal transport
    _HAVE_OT = True
except ImportError:
    _HAVE_OT = False


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


def _solve_worker(queue, solver, ybar, S, m, ref_sup, ref_w):
    """Run one fit with `solver`, score it against the reference, and put
    (elapsed_seconds, distance) on the queue. On failure, put ("error", repr)."""
    try:
        wrapper = make_single_solver_wrapper(solver)
        t0 = time.perf_counter()
        fit = get_joint_mixing_distribution_npmle(
            ybar, S, m, cvxpy_solve_wrapper=wrapper
        )
        elapsed = time.perf_counter() - t0
        sup, w = _atoms_and_weights(fit)
        distance = _distance_between(sup, w, ref_sup, ref_w)
        queue.put((elapsed, distance))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _solve_with_timeout(solver, ybar, S, m, ref_sup, ref_w, timeout):
    """Return (elapsed, distance) for one timed+scored solve, or raise:
    TimeoutError if it exceeds `timeout`, RuntimeError if the solve failed."""
    ctx = mp.get_context("spawn")  # spawn: clean process, no inherited solver state
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_solve_worker,
        args=(queue, solver, ybar, S, m, ref_sup, ref_w),
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


def compare_solvers(seed=0, repeats=3, accuracy_atol=1e-6, timeout=SOLVE_TIMEOUT_S):
    """Time every installed solver on the same joint-NPMLE problem and check
    each result for accuracy against a reference fit.

    Parameters
    ----------
    repeats : int
        Number of timed runs per solver; the reported time is the best (min)
        wall-clock time, which is the standard way to reduce noise.
    accuracy_atol : float
        A solver is "accurate" if its distribution distance to the reference is
        <= this. The POT path returns a true transport cost (tight tolerance is
        fine); the moment-summary fallback is looser, so loosen this if you are
        running without POT.
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


if __name__ == "__main__":
    compare_solvers()
