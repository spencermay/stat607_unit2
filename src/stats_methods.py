from abc import ABC, abstractmethod

import numpy as np
import cvxpy as cp

from data_generating_processes import GaussianMixture


# NPMLE by Kiefer-Wolfowitz (at least as far as I can gather from the paper)
def get_mixing_distribution_npmle(ybar, S):
    raise NotImplementedError

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