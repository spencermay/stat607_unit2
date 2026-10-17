from abc import ABC, abstractmethod

import numpy as np
import cvxpy as cp
from scipy.stats import norm, gamma

from data_generating_processes import GaussianMixture


def solve_npmle():
    

# NPMLE by Kiefer-Wolfowitz (at least as far as I can gather from the paper)
def get_mixing_distribution_npmle(ybar, S, m, gridsize=1000):
    """
    Get the NPMLE estimate of the mixing distributions for the mean/variance
    """
    # First, solve for a discrete pmf approximation to the variance max-likelihood distribution
    r = (m-1)/2

    # The max-likelihood mixing distribution is supported on [min(data), max(data)]
    grid_endpoints = (min(S / r), max(S / r))
    grid_start, grid_end = grid_endpoints
    grid_locations = np.linspace(grid_start, grid_end, gridsize).reshape((gridsize, 1))

    # Make a cvxpy variable to find argmax of loglikelihood 
    pmf_variance = cp.Variable(gridsize)

    # Compute the A matrix for the variance problem (a bunch of gamma pdfs)
    A_variance = gamma(, ).pdf(S)
    
    max_likelihood = cp.Maximize(cp.sum(cp.log(A @ pmf_variance)))
    
    constraints = [pmf_variance >=0, cp.sum(pmf_variance) == gridsize]
    
    problem = cp.Problem(max_likelihood, constraints)


    A_mean = norm.pdf(np.sqrt(m) @ (ybar - ))

    # The max likelihood mixing distribution is supported on [min(ybar), max(ybar)] because vibes
    grid_endpoints = (min(ybar), max(ybar))
    grid_start, grid_end = grid_endpoints

    grid_locations = np.linspace(grid_start, grid_end, gridsize).reshape((gridsize, 1))

    # Solve for max-likelihood pmf variance first
    pmf_mean = cp.Variable(gridsize)


    
    bias_start, bias_end = bias_range

    x_bias = np.linspace(bias_start, bias_end, N).reshape((N,1))

    

    # Find normal cdfs for (-cva pm x_bias) for all biases
    norm_cdfs = norm.cdf(-cva - x_bias) + norm.cdf(-cva + x_bias)

    # Initalize normed bias variable
    bias = cp.Variable(N)

    max_risk = cp.Maximize(norm_cdfs.T @ bias)
    constraints = [bias >=0, cp.sum(bias)==1, x_bias.T @ bias == first_moment, np.square(x_bias).T @ bias == second_moment]

    problem = cp.Problem(max_risk, constraints)

    problem.solve()

    return bias.value, max_risk.value



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