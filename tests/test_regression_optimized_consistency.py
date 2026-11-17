"""
Ensure that the optimized code is consistent with the old code.
"""

import numpy as np
import pytest
import ot

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
PROF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiling")

for filepath in (SRC, PROF):
    if filepath not in sys.path:
        sys.path.insert(0, filepath)

from src.stats_methods import get_joint_mixing_distribution_npmle as joint_npmle_new
from profiling.legacy_code import get_joint_mixing_distribution_npmle as joint_npmle_legacy
from tests.test_stats_methods import _draw_correlated, M


def _support_and_weights(gm):
    means, variances, weights = gm.mixing_distribution.mean_and_variance_joint_approx_pmf()
    support = np.column_stack([np.asarray(means, float), np.asarray(variances, float)])
    return support, np.asarray(weights, float)


def test_joint_npmle_solver_regression():
    """
    Check that the production support-reduction joint-NPMLE fit AGREES with the
    legacy cvxpy convex solve to within ~1e-3 Wasserstein distance -- not that
    it reproduces it exactly.

    The production solver was swapped from the dense cvxpy convex maximization
    to fixed-grid support reduction, so the two are different algorithms and the
    fitted mixtures no longer land on exactly the same support with exactly the
    same mass. We therefore compare the two discrete joint distributions by the
    squared Wasserstein (optimal-transport) distance, which is robust to the
    support pruning differing between versions, and require it to be small
    rather than zero. Measured across seeds 0,1,2 at the 40x40 default grid the
    distance is ~5e-5 to ~3.6e-4 and does not shrink with start_floor, so the
    tolerance is atol=1e-3.
    """
    for seed in range(3):
        ybar, S, _ = _draw_correlated(seed)

        gotGM = joint_npmle_new(ybar, S, M)
        wantGM = joint_npmle_legacy(ybar, S, M)

        got_support, got_weights = _support_and_weights(gotGM)
        want_support, want_weights = _support_and_weights(wantGM)

        # Squared Euclidean ground cost between joint (mean, variance) atoms.
        M_sq_euclid = ot.dist(got_support, want_support, metric="sqeuclidean")

        # Exact minimum transport cost (squared Wasserstein distance).
        min_cost = ot.emd2(got_weights, want_weights, M_sq_euclid)

        assert np.isclose(min_cost, 0, rtol=0, atol=1e-3), (
            f"seed {seed}: support-reduction vs legacy cvxpy Wasserstein^2 "
            f"distance {min_cost:.3e} exceeds 1e-3"
        )
