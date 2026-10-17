from abc import ABC, abstractmethod

import numpy as np
import cvxpy as cp
from scipy.stats import norm, gamma

from data_generating_processes import GaussianMixture


def solve_npmle():
    

# NPMLE by Kiefer-Wolfowitz (at least as far as I can gather from the paper)
def get_mixing_distribution_npmle(ybar, S, m, gridsize_mean=1000, gridsize_S=1001):
    """
    Get the NPMLE estimate of the mixing distributions for the mean/variance
    """
    # First, solve for a discrete pmf approximation to the variance max-likelihood distribution
    
    # Shape constant for variance gamma distribution
    r = (m-1)/2

    # The max-likelihood mixing distribution is supported on [min(data), max(data)]
    grid_endpoints = (min(S / r), max(S / r))
    grid_start, grid_end = grid_endpoints
    # The pmf for variance is supported on this grid
    variance_grid = (np.linspace(grid_start, grid_end, num=gridsize_S)
                     .reshape((gridsize_S, 1)))

    # Make a cvxpy variable for the pmf to find argmax of loglikelihood 
    f_variance = cp.Variable(gridsize_S)

    # Compute the A matrix for the variance problem (a bunch of gamma pdfs)
    A_variance = gamma(r, variance_grid).pdf(S)
    
    max_likelihood = cp.Maximize(cp.sum(cp.log(A_variance @ f_variance)))
    
    constraints = [f_variance >=0, cp.sum(f_variance) == gridsize_S]
    
    problem = cp.Problem(max_likelihood, constraints)

    problem.solve()

    pmf_variance = f_variance.value



    # Next, solve for a discrete pmf approximation to the *mean* max-likelihood distribution
    
    grid_endpoints = (min(ybar / r), max(ybar / r))
    grid_start, grid_end = grid_endpoints
    # The pmf for variance is supported on this grid
    mean_grid = (np.linspace(grid_start, grid_end, num=gridsize_mean)
                 .reshape((gridsize_mean, 1)))

    A_mean = np.sqrt(m/variance_grid) \
                * norm.pdf(np.sqrt(m) \
                * (ybar - mean_grid)/np.sqrt(variance_grid)) \
                * pmf_variance

    # Solve for max-likelihood pmf
    f_mean = cp.Variable(gridsize_mean)

    max_likelihood = cp.Maximize(cp.sum(cp.log(A_mean @ f_mean)))
    
    constraints = [f_mean >=0, cp.sum(f_mean) == gridsize_S]
    
    problem = cp.Problem(max_likelihood, constraints)

    problem.solve()

    pmf_mean = f_mean.value

    return {"mean_grid": mean_grid,
            "pmf_mean": pmf_mean,
            "variance_grid": variance_grid,
            "pmf_variance": pmf_variance}


class BayesRuleEstimator(ABC):
    @abstractmethod
    def bayes_rule(self, ybar, S) -> tuple:
        """Compute the Bayes rule for (mean,variance) given
        ybar and standard error for n samples of a Gaussian mixture"""

class NaiveEstimator(BayesRuleEstimator):
    """The no-shrinkage estimator"""
    def bayes_rule(self, ybar, S):
        return ybar

class OracleBayesRule(BayesRuleEstimator):
    def __init__(self, true_dgp: GaussianMixture) -> None:
        self.true_dgp = true_dgp

    def bayes_rule(self, ybar, S):
        raise NotImplementedError

class NPMLE_BayesRule(BayesRuleEstimator):
    def bayes_rule(self, ybar, S):
        raise NotImplementedError