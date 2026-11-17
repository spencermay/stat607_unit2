"""
Diagnostic for the support-reduction vs default NPMLE
divergence at 40x40. Does NOT modify src/ or profiling/ behavior; it only
imports their helpers and runs fits.

This is basically a wrapper for my old figure methods thanks to Claude.

Run with the repo venv:
    .venv/bin/python profiling/diagnose.py
"""
import os
import sys

_ROOT = "/Users/svmay/Projects/classes/stat607/unit2/stat607_unit2"
for p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "profiling"),
          os.path.join(_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ot

from speed_comparison_convex_solvers import (
    make_data, _build_joint_grid_and_A, _atoms_and_weights,
    fit_support_reduction,
)
from stats_methods import get_joint_mixing_distribution_npmle, _solve_npmle_problem
from support_reduction_npmle_alternative import fit_mixture_support_reduction
from figures import plot_support_points_ybar_and_S

OUT = "/Users/svmay/Projects/classes/stat607/unit2/stat607_unit2/profiling/diagnostic_plots/"


def mean_loglik(A, weights):
    """Mean log-likelihood (1/n) sum log(A @ w) on the FULL grid A and weights."""
    w = np.clip(np.asarray(weights, float).ravel(), 0, None)
    w = w / w.sum()
    q = A @ w
    return float(np.mean(np.log(q)))


def gap_bound(A, weights):
    """Upper bound on remaining mean-loglik improvement: max_j (A^T (1/q))/n - 1."""
    w = np.clip(np.asarray(weights, float).ravel(), 0, None)
    w = w / w.sum()
    q = A @ w
    return max(0.0, float(np.max(A.T @ (1.0 / q)) / len(q) - 1.0))


def default_full_grid_weights(ybar, S, m, gm, gs):
    """Re-solve the default cvxpy NPMLE and return FULL-grid (pre-prune) weights
    aligned with _build_joint_grid_and_A's atom order, by replicating the exact
    construction in get_joint_mixing_distribution_npmle (so we can score mean
    log-likelihood on the identical A_joint)."""
    import cvxpy as cp
    alpha_atoms, theta_atoms, A = _build_joint_grid_and_A(ybar, S, m, gm, gs)
    n_atoms = alpha_atoms.size
    f = cp.Variable(n_atoms)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.log(A @ f))),
                      [f >= 0, cp.sum(f) == 1])
    _solve_npmle_problem(prob)
    w = np.clip(np.asarray(f.value).ravel(), 0, None)
    w = w / w.sum()
    return alpha_atoms, theta_atoms, A, w


def run_grid(ybar, S, m, gm, gs):
    print(f"\n===== grid {gm}x{gs} ({gm*gs} atoms) =====")
    alpha_atoms, theta_atoms, A, w_def_full = default_full_grid_weights(
        ybar, S, m, gm, gs
    )

    # support-reduction raw diagnostics on the identical A
    fit = fit_mixture_support_reduction(A)

    ll_def = mean_loglik(A, w_def_full)
    ll_sr = fit["mean_loglik"]
    gap_def = gap_bound(A, w_def_full)
    gap_sr = fit["gap"]

    print("support-reduction diagnostics:")
    print(f"  status      : {fit['status']!r}")
    print(f"  success     : {fit['success']}")
    print(f"  steps       : {fit['steps']}  (max_steps=200)")
    print(f"  gap         : {fit['gap']:.3e}  (gap_tol=1e-6)")
    print(f"  #active atoms (raw): {len(fit['support'])}")
    print(f"  mean_loglik : {ll_sr:.10f}")
    print("default cvxpy (full grid, pre-prune):")
    print(f"  mean_loglik : {ll_def:.10f}")
    print(f"  gap bound   : {gap_def:.3e}")
    print(f"  #atoms > 1e-4*max : {int(np.sum(w_def_full > 1e-4*w_def_full.max()))}")
    print(f"mean_loglik(default) - mean_loglik(SR) = {ll_def - ll_sr:.3e}")

    # Pruned GaussianMixtures (what the adapter returns) and W2^2 between them.
    gm_def = get_joint_mixing_distribution_npmle(ybar, S, m,
                                                 gridsize_mean=gm, gridsize_S=gs)
    gm_sr = fit_support_reduction(ybar, S, m, gridsize_mean=gm, gridsize_S=gs)
    sup_def, wt_def = _atoms_and_weights(gm_def)
    sup_sr, wt_sr = _atoms_and_weights(gm_sr)
    cost = ot.dist(sup_def, sup_sr, metric="sqeuclidean")
    w2sq = float(ot.emd2(wt_def, wt_sr, cost))
    print(f"W2^2(default pruned, SR pruned) = {w2sq:.6e}")
    print(f"pruned support sizes: default={len(sup_def)}  SR={len(sup_sr)}")

    return {
        "gm": gm, "gs": gs,
        "fit": fit,
        "ll_def": ll_def, "ll_sr": ll_sr,
        "gap_def": gap_def, "gap_sr": gap_sr,
        "sup_def": sup_def, "wt_def": wt_def,
        "sup_sr": sup_sr, "wt_sr": wt_sr,
        "w2sq": w2sq,
    }


def overlay_figure(res, m, path):
    """Overlay the two pruned support sets in (ybar, S) coords using the EXACT
    mapping from figures.plot_support_points_ybar_and_S."""
    r = res
    sup_def, wt_def = r["sup_def"], r["wt_def"]
    sup_sr, wt_sr = r["sup_sr"], r["wt_sr"]
    shared_max = max(wt_def.max(), wt_sr.max())

    fig, ax = plt.subplots(figsize=(7, 6))
    # support arrays are (means=alpha, variances=theta) columns
    plot_support_points_ybar_and_S(
        sup_def[:, 0], sup_def[:, 1], wt_def, color="tab:blue",
        label="default cvxpy (CLARABEL->SCS)", m=m, ax=ax,
        max_weight=shared_max, filled=False,
    )
    plot_support_points_ybar_and_S(
        sup_sr[:, 0], sup_sr[:, 1], wt_sr, color="tab:orange",
        label="support-reduction", m=m, ax=ax,
        max_weight=shared_max, filled=True,
    )
    ax.set_xlabel(r"$\bar{y}$  (alpha axis)")
    ax.set_ylabel(r"S  at  E[S|theta]=(m-1)theta")
    ax.set_title(f"NPMLE support overlay, {r['gm']}x{r['gs']} grid"
                 f"  (W2^2={r['w2sq']:.3g})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ybar, S, m = make_data(seed=0)
    print(f"data: n={len(ybar)}, m={m}, seed=0")

    res20 = run_grid(ybar, S, m, 20, 20)
    res40 = run_grid(ybar, S, m, 40, 40)

    overlay_figure(res20, m, os.path.join(OUT, "support_overlay_20x20.png"))
    overlay_figure(res40, m, os.path.join(OUT, "support_overlay_40x40.png"))

    # Print raw atom coordinates for the 40x40 divergence table.
    for tag, r in (("20x20", res20), ("40x40", res40)):
        print(f"\n--- {tag} default pruned atoms (alpha, theta, w) ---")
        order = np.argsort(-r["wt_def"])
        for i in order:
            print(f"  alpha={r['sup_def'][i,0]:+.4f} theta={r['sup_def'][i,1]:.4f} w={r['wt_def'][i]:.4f}")
        print(f"--- {tag} SR pruned atoms (alpha, theta, w) ---")
        order = np.argsort(-r["wt_sr"])
        for i in order:
            print(f"  alpha={r['sup_sr'][i,0]:+.4f} theta={r['sup_sr'][i,1]:.4f} w={r['wt_sr'][i]:.4f}")


if __name__ == "__main__":
    main()
