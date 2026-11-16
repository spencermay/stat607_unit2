import numpy as np
import pytest
from scipy.stats import gamma
import ot

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
PROF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiling")
# print(SRC)
for filepath in (SRC, PROF):
    if filepath not in sys.path:
        sys.path.insert(0, filepath)

from src.stats_methods import get_joint_mixing_distribution_npmle as joint_npmle_new
from profiling.legacy_code import get_joint_mixing_distribution_npmle as joint_npmle_legacy

def test_joint_npmle_solver_regression():
    """
    We want to test whether the distributions are the same but
    in general because we prune the support and I suspect the solver
    will not in general have the same mass on exactly the same support
    in different versions, I use the optimal transport cost to determine
    if the resulting discrete distributions are the same.
    """
    gotGM = joint_npmle_new()
    wantGM = joint_npmle_legacy()

    got_support_means, got_support_vars, got_weights = gotGM.mixing_distribution.mean_and_variance_joint_approx_pmf()
    want_support_means, want_support_vars, want_weights = wantGM.mixing_distribution.mean_and_variance_joint_approx_pmf()
    
    # 1. Squared Euclidean distance (default)
    M_sq_euclid = ot.dist((got_support_means, got_support_vars), (want_support_means, want_support_vars), metric='sqeuclidean')

    # Exact minimum cost (Wasserstein distance)
    min_cost = ot.emd2(got_weights, want_weights, M_sq_euclid)

    assert np.isclose(min_cost, 0, rtol=0, atol=1e-12)