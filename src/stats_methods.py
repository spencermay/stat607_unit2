from abc import ABC, abstractmethod

import numpy as np
import cvxpy as cp
from scipy.stats import norm, gamma, t

from data_generating_processes import GaussianMixture


# def maximize_pmf_convex(likelihood: callable,
#                         grid_bounds: tuple,
#                         other_estimated_pmfs: dict,
#                         **data):
#     """
#     Helper function for the NPMLE estimate of the mixing distribution.
    
#     Maximize Integral[ 
#                     likelihood(
#                         this_estimate,
#                         other_estimates,
#                         data)
#                     ]
#                     d(this_estimate)

#     (over all discrete pmfs supported on grid_interval with N_GRID support
#         points equally spaced)
    
#     Parameters
#     ----------
#     likelihood: function (to be maximized) taking (other_estimates, data)
#                 and returning a scalar
#     other_estimates:
#         dict(
#             "grid": spacing,
#             "pmf": probability mass distributed on that grid,
#             )
#     data:
#         all the other variables that determine the likelihood like (S, ybar, m)
#         for the mixture of gaussians setting.
#     """

#     grid_endpoints = (min(S / r), max(S / r))
#     grid_start, grid_end = grid_endpoints
#     # The pmf for variance is supported on this grid
#     variance_grid = (np.linspace(grid_start, grid_end, num=gridsize_S)
#                         .reshape((gridsize_S, 1)))

#     # Make a cvxpy variable for the pmf to find argmax of loglikelihood 
#     f_variance = cp.Variable(gridsize_S)

#     # Compute the A matrix for the variance problem (a bunch of gamma pdfs)
#     A_variance = gamma(r, variance_grid).pdf(S)
    
#     max_likelihood = cp.Maximize(cp.sum(cp.log(A_variance @ f_variance)))
    
#     constraints = [f_variance >=0, cp.sum(f_variance) == gridsize_S]
    
#     problem = cp.Problem(max_likelihood, constraints)

#     problem.solve()

#     pmf_variance = f_variance.value



# NPMLE by Kiefer-Wolfowitz (at least as far as I can gather from the paper)
def get_mixing_distribution_npmle(ybar, S, m, gridsize_mean=1000, gridsize_S=1001):
    """
    Get the NPMLE estimate of the mixing distributions for the mean/variance.

    Parameters
    ----------
    ybar: observed sample mean
    S: observed standard error
    gridsizes: the number of support points to include in each discrete pmf approximation

    Returns
    -------
    Dictionary:
            "mean_grid": the support points for the pmf of
                    the max-likelihood mean mixing distribution
            "pmf_mean": the mass assigned to each of those same support
                    points for the mean mixing distribution
            "variance_grid": the support points for the pmf of
                    the max-likelihood variance mixing distribution
            "pmf_variance": the mass assigned to each of those same support
                    points for the variance mixing distribution
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

    # use the t-statistic pivot strategy proposed in the paper
    # Note: I think neither is actually fully justified!!!
    # A_mean = np.sqrt(m / S) \
    #                 * t.pdf(np.sqrt(m / S) \
    #                         * (ybar - mean_grid))


    # Solve for max-likelihood pmf
    f_mean = cp.Variable(gridsize_mean)

    max_likelihood = cp.Maximize(cp.sum(cp.log(A_mean @ f_mean)))
    
    constraints = [f_mean >=0, cp.sum(f_mean) == gridsize_mean]
    
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

def _gaussian_pdf(x, mu, var):
    """Exact Gaussian density phi(x; mu, var) evaluated at x (no randomness)."""
    # This function was AI-generated
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def _gaussian_pdf_derivative(x, mu, var):
    """
    Exact derivative d/dx phi(x; mu, var).

    For a Gaussian, phi'(x) = -((x - mu) / var) * phi(x).
    """
    # This function was AI-generated
    return -((x - mu) / var) * _gaussian_pdf(x, mu, var)


class OracleBayesRule(BayesRuleEstimator):
    """
    Oracle posterior-mean estimator for alpha, E[alpha | ybar], under the true
    DGP, via Tweedie's formula:

        E[alpha | ybar] = ybar + (var_ybar) * f'(ybar) / f(ybar)

    where f is the marginal density of ybar. Because ybar | (alpha, theta) ~
    N(alpha, theta/m), the marginal is a finite mixture over the joint support
    of (alpha, theta):

        f(ybar)  = sum_k w_k * phi(ybar; alpha_k, theta_k / m)
        f'(ybar) = sum_k w_k * phi'(ybar; alpha_k, theta_k / m)

    Both f and f' are evaluated exactly (closed form), so no RNG stream is
    touched.

    Heteroscedastic note: the naive single-multiplier Tweedie form
    ``ybar + v * f'/f`` is only exact when every component shares the same
    sampling variance. Here theta (hence theta/m) is random, so we use the
    exact per-component identity. Within component k,

        alpha_k * phi_k(ybar) = ybar * phi_k(ybar) + (theta_k/m) * phi_k'(ybar),

    which follows from phi_k'(ybar) = -((ybar - alpha_k)/(theta_k/m)) phi_k.
    Summing over the mixture gives the exact derivative-over-density form

        E[alpha | ybar] = ybar + ( sum_k w_k (theta_k/m) phi_k'(ybar) ) / f(ybar),

    i.e. the Tweedie correction is the variance-scaled sum of the component
    density derivatives, divided by the marginal density.
    """

    def __init__(self, true_dgp: GaussianMixture, m: int) -> None:
        self.true_dgp = true_dgp
        self.m = m
        # Precompute the (fixed) joint support and weights of the mixing dist.
        means, variances, weights = (
            true_dgp.mixing_distribution.mean_and_variance_joint_approx_pmf()
        )
        self._means = np.asarray(means)
        self._variances = np.asarray(variances)
        self._weights = np.asarray(weights)
        # Sampling variance of ybar for each joint support point.
        self._ybar_variances = self._variances / self.m

    def bayes_rule(self, ybar, S):
        """
        Exact oracle posterior mean E[alpha | ybar] (S is unused: the oracle
        marginalizes over the true variance mixing distribution rather than
        plugging in an observed standard error).

        Accepts a scalar or array-like `ybar`; returns the matching shape.
        """
        ybar = np.asarray(ybar, dtype=float)

        # Evaluate each component's N(alpha_k, theta_k/m) density and derivative
        # at every ybar. Broadcast ybar over the component axis.
        x = ybar[..., None]  # shape (..., 1)
        mu = self._means  # shape (K,)
        var = self._ybar_variances  # shape (K,)

        comp_pdf = _gaussian_pdf(x, mu, var)  # (..., K)
        comp_pdf_deriv = _gaussian_pdf_derivative(x, mu, var)  # (..., K)

        # Marginal density f(ybar) = sum_k w_k phi_k(ybar).
        f = np.sum(self._weights * comp_pdf, axis=-1)  # (...)

        # Exact Tweedie correction for heteroscedastic components:
        #   sum_k w_k (theta_k/m) phi_k'(ybar) / f(ybar).
        correction_numer = np.sum(
            self._weights * var * comp_pdf_deriv, axis=-1
        )  # (...)

        result = ybar + correction_numer / f
        return result

class NPMLE_BayesRule(BayesRuleEstimator):
    def bayes_rule(self, ybar, S):
        raise NotImplementedError