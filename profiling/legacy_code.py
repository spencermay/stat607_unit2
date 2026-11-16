"""
Regression tests will use this untouched code for solving the npmle
"""

from abc import ABC, abstractmethod

import numpy as np
import cvxpy as cp
from scipy.stats import norm, gamma, t

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
# print(SRC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.data_generating_processes import GaussianMixture, MixJointMeanVariance

def get_joint_mixing_distribution_npmle(ybar, S, m,
                                        gridsize_mean=40, gridsize_S=40,
                                        prune_tol=1e-4,
                                        cvxpy_solve_wrapper=_solve_npmle_problem):
    """
    NPMLE estimate of the JOINT mixing distribution H(alpha, theta), WITHOUT
    assuming alpha and theta are independent.

    Unlike get_mixing_distribution_npmle -- which estimates the alpha-marginal
    and theta-marginal separately and takes their outer product (valid only
    under independence) -- this solves one convex maximization over a single
    joint pmf on the Cartesian grid of (alpha_k, theta_l) atoms. That joint pmf
    need not factor, so correlated (alpha, theta) can be recovered.

    Likelihood of observation i under atom (alpha_k, theta_l), using the same
    sufficient-statistic factorization as the oracle (ybar _||_ S | theta):

        L_i(k, l) = phi(ybar_i; alpha_k, theta_l / m)        # mean part
                    * Gamma(S_i; (m-1)/2, 2 theta_l)         # variance part

    We maximize sum_i log( sum_{k,l} A[i, (k,l)] f_{kl} ) over the joint simplex
    {f >= 0, sum f = 1}. The problem has gridsize_mean * gridsize_S variables
    (one per joint atom) and an (n, gridsize_mean*gridsize_S) likelihood matrix,
    so it is much heavier than the two small independent solves -- hence the
    coarser default grid. Pruning keeps the returned support small regardless.

    Parameters
    ----------
    ybar, S, m : as in get_mixing_distribution_npmle.
    gridsize_mean, gridsize_S : int
        Atoms per axis; the joint grid has gridsize_mean * gridsize_S points.
        Defaults are deliberately coarse because the joint solve scales with
        the product.
    prune_tol : float
        Relative mass threshold for dropping negligible joint atoms (see
        _prune_pmf); prune_tol <= 0 disables pruning.

    Returns
    -------
    GaussianMixture wrapping a MixJointMeanVariance over the (pruned) joint
    support -- a joint distribution, consumed identically to the
    independent NPMLE by bayes_rule / marginal_bayes_rule / the metrics.
    """
    ybar = np.asarray(ybar, dtype=float)
    S = np.asarray(S, dtype=float)
    r = (m - 1) / 2

    # Axis grids (same endpoint conventions as the independent version).
    mean_axis = np.linspace(min(ybar), max(ybar), gridsize_mean)
    variance_axis = np.linspace(min(S / r), max(S / r), gridsize_S)

    # Flat joint atoms in C-order: atom index p = k * gridsize_S + l maps to
    # (mean_axis[k], variance_axis[l]). This matches the flattening used
    # elsewhere (Simulation, MixingDistribution.mean_and_variance_joint_approx_pmf).
    alpha_atoms = np.repeat(mean_axis, gridsize_S)          # (P,)
    theta_atoms = np.tile(variance_axis, gridsize_mean)     # (P,)
    n_atoms = alpha_atoms.size                              # P = K * L

    # Likelihood matrix A[i, p] = L_i(atom_p), shape (N, P).
    #   mean part: phi(ybar_i; alpha_p, theta_p/m)
    #   var  part: Gamma(S_i; (m-1)/2, scale=2 theta_p)
    ybar_var = theta_atoms / m                                      # (P,)
    mean_part = norm.pdf(ybar[:, None], loc=alpha_atoms, scale=np.sqrt(ybar_var))
    var_part = gamma.pdf(S[:, None], a=r, scale=2 * theta_atoms)    # (N, P)
    A_joint = mean_part * var_part                                  # (N, P)

    # Single joint convex maximization over the joint simplex.
    f_joint = cp.Variable(n_atoms)
    max_likelihood = cp.Maximize(cp.sum(cp.log(A_joint @ f_joint)))
    constraints = [f_joint >= 0, cp.sum(f_joint) == 1]
    problem = cp.Problem(max_likelihood, constraints)
    cvxpy_solve_wrapper(problem)

    weights = np.clip(np.asarray(f_joint.value).ravel(), 0, None)
    weights = weights / weights.sum()

    # Prune negligible joint atoms (keep alpha/theta aligned with weights).
    keep = weights > prune_tol * weights.max() if prune_tol > 0 else slice(None)
    alpha_keep = alpha_atoms[keep]
    theta_keep = theta_atoms[keep]
    weights_keep = weights[keep]
    weights_keep = weights_keep / weights_keep.sum()

    return GaussianMixture(
        MixJointMeanVariance(alpha_keep, theta_keep, weights_keep)
    )