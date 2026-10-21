from abc import ABC, abstractmethod

import numpy as np
import cvxpy as cp
from scipy.stats import norm, gamma, t

from data_generating_processes import GaussianMixture, MixIndependentMeanVariance, DiscreteDistribution


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
def get_mixing_distribution_npmle(ybar, S, m, gridsize_mean=200, gridsize_S=200):
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

    # Compute the A matrix for the variance problem (a bunch of gamma pdfs).
    # S = sum_t (y_t - ybar)^2 and S | theta ~ theta * chi^2_{m-1}
    #   = Gamma(shape=(m-1)/2, scale=2*theta), so the per-grid-point scale is
    #   2*variance_grid. The second positional arg to scipy's gamma is `loc`
    #   (a shift), NOT the scale, so it must be passed as `scale=`.
    A_variance = gamma(r, scale=2 * variance_grid).pdf(S)
    
    max_likelihood = cp.Maximize(cp.sum(cp.log(A_variance.T @ f_variance)))
    
    constraints = [f_variance >=0, cp.sum(f_variance) == gridsize_S]
    
    problem = cp.Problem(max_likelihood, constraints)

    problem.solve()

    pmf_variance = f_variance.value



    # Next, solve for a discrete pmf approximation to the *mean* max-likelihood distribution
    
    
    grid_endpoints = (min(ybar), max(ybar))
    grid_start, grid_end = grid_endpoints
    # The pmf for variance is supported on this grid
    mean_grid = (np.linspace(grid_start, grid_end, num=gridsize_mean)
                 .reshape((gridsize_mean, 1)))

    # A_mean[k, i] = f(ybar_i | alpha_k), the likelihood of the i-th observed
    # mean under mean grid point alpha_k = mean_grid[k], marginalized over the
    # (already estimated) variance mixing distribution:
    #
    #   f(ybar_i | alpha_k) = sum_j pmf_variance[j]
    #                          * phi(ybar_i; alpha_k, theta_j / m)
    #
    # with theta_j = variance_grid[j] and
    #   phi(ybar; alpha, theta/m) = sqrt(m/theta) * norm.pdf(sqrt(m)(ybar-alpha)/sqrt(theta)).
    #
    # Three axes are involved -- mean grid (k), variance grid (j), data (i) --
    # so we build the (k, j, i) tensor and contract (sum) over the variance
    # axis j. Grids are stored as (gridsize, 1) columns; flatten them here and
    # place each quantity on its own axis via broadcasting.
    alpha_k = mean_grid.reshape(gridsize_mean, 1, 1)       # (K, 1, 1)
    theta_j = variance_grid.reshape(1, gridsize_S, 1)      # (1, J, 1)
    ybar_i = np.asarray(ybar).reshape(1, 1, -1)            # (1, 1, N)
    pmf_var_j = np.asarray(pmf_variance).reshape(1, gridsize_S, 1)  # (1, J, 1)

    component = (
        np.sqrt(m / theta_j)
        * norm.pdf(np.sqrt(m) * (ybar_i - alpha_k) / np.sqrt(theta_j))
    )  # (K, J, N) = phi(ybar_i; alpha_k, theta_j/m)

    # Marginalize over the variance grid -> (K, N), matching A_variance's
    # (grid x data) orientation so A_mean.T @ f_mean below is unchanged.
    A_mean = np.sum(pmf_var_j * component, axis=1)  # (K, N)

    # use the t-statistic pivot strategy proposed in the paper
    # Note: I think neither is actually fully justified!!!
    # A_mean = np.sqrt(m / S) \
    #                 * t.pdf(np.sqrt(m / S) \
    #                         * (ybar - mean_grid))


    # Solve for max-likelihood pmf
    f_mean = cp.Variable(gridsize_mean)

    max_likelihood = cp.Maximize(cp.sum(cp.log(A_mean.T @ f_mean)))
    
    constraints = [f_mean >=0, cp.sum(f_mean) == gridsize_mean]
    
    problem = cp.Problem(max_likelihood, constraints)

    problem.solve()

    pmf_mean = f_mean.value

    # The cvxpy problems constrain sum(f) == gridsize (not 1), and the solver
    # can return tiny negatives, so normalize into proper pmfs before handing
    # them to DiscreteDistribution (which requires nonnegative weights summing
    # to 1). Also flatten grids/pmfs to 1-D so their shapes match (grids are
    # built as (gridsize, 1) columns; pmfs come back as (gridsize,)).
    mean_grid = np.asarray(mean_grid).ravel()
    variance_grid = np.asarray(variance_grid).ravel()

    pmf_mean = np.clip(np.asarray(pmf_mean).ravel(), 0, None)
    pmf_mean = pmf_mean / pmf_mean.sum()
    pmf_variance = np.clip(np.asarray(pmf_variance).ravel(), 0, None)
    pmf_variance = pmf_variance / pmf_variance.sum()

    return GaussianMixture(
            MixIndependentMeanVariance(
             DiscreteDistribution(mean_grid, pmf_mean),
             DiscreteDistribution(variance_grid, pmf_variance)
            )
           )


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



def bayes_rule(gaussian_mixture, ybar, S, m):
    """
    Exact oracle posterior mean E[alpha | ybar, S].

    Parameters
    ----------
    gaussian_mixture : GaussianMixture
        Carries the (true or estimated) mixing distribution whose joint support
        (alpha_j, theta_j, w_j) drives the Bayes rule, accessed via
        gaussian_mixture.mixing_distribution.mean_and_variance_joint_approx_pmf().
    ybar : scalar or array-like
        Per-unit sample mean, m^{-1} sum_t y_t.
    S : scalar or array-like
        Per-unit deviation sum of squares, sum_t (y_t - ybar)^2. Must
        broadcast against `ybar`.

    Returns
    -------
    ndarray (or scalar) of E[alpha | ybar, S], matching the broadcast shape
    of (ybar, S).
    """
    # Written by Claude, with my direction about what math should be happening
    # I had to take particular pains on Proposition 2 in the paper

    ybar = np.asarray(ybar, dtype=float)
    S = np.asarray(S, dtype=float)

    # Flat joint support of H: parallel arrays over joint points (alpha_j, theta_j).
    means, variances, weights = (
        gaussian_mixture.mixing_distribution.mean_and_variance_joint_approx_pmf()
    )
    _means = np.asarray(means, dtype=float)
    _variances = np.asarray(variances, dtype=float)
    _weights = np.asarray(weights, dtype=float)
    # Sampling variance of ybar at each joint support point.
    _ybar_variances = _variances / m

    # Distinct theta values (the variance posterior lives on these), plus a
    # map from each joint support point to its distinct-theta index.
    _theta_values, _theta_index = np.unique(
        _variances, return_inverse=True
    )
    # Gamma parameters for S | theta: S ~ theta * chi^2_{m-1}
    #   => Gamma(shape=(m-1)/2, scale=2*theta).
    _gamma_shape = (m - 1) / 2.0
    _gamma_scales = 2.0 * _theta_values  # per distinct theta

    # Broadcast observations over the joint-support axis (last axis = K points).
    x = ybar[..., None]              # (..., 1)
    mu = _means                 # (K,)
    var = _ybar_variances       # (K,)

    # phi(ybar; alpha_j, theta_j/m) at every joint point.
    comp_pdf = _gaussian_pdf(x, mu, var)  # (..., K)

    # ---- Factor 1: E[alpha | ybar, theta_k] for each DISTINCT theta_k ----
    # Numerator/denominator of the fixed-theta posterior mean, accumulated
    # per distinct theta over the joint points that share that theta.
    n_theta = len(_theta_values)
    batch_shape = comp_pdf.shape[:-1]

    num_fixed = np.zeros(batch_shape + (n_theta,))  # sum_j w_jk alpha_j phi
    den_fixed = np.zeros(batch_shape + (n_theta,))  # sum_j w_jk phi  = f(ybar|theta_k)
    w_phi = _weights * comp_pdf                 # (..., K)
    np.add.at(
        np.moveaxis(num_fixed, -1, 0),
        _theta_index,
        np.moveaxis(w_phi * _means, -1, 0),
    )
    np.add.at(
        np.moveaxis(den_fixed, -1, 0),
        _theta_index,
        np.moveaxis(w_phi, -1, 0),
    )
    # E[alpha | ybar, theta_k]; den_fixed > 0 wherever any mass is present.
    with np.errstate(invalid="ignore", divide="ignore"):
        cond_mean_alpha = num_fixed / den_fixed  # (..., n_theta)

    # ---- Factor 2: posterior over theta given (ybar, S) ------------------
    # f(ybar | theta_k) is exactly den_fixed (marginal of alpha at theta_k).
    # f(S | theta_k) is the Gamma density on the deviation SS. Prior
    # p(theta_k) is already baked into den_fixed via the joint weights w_jk,
    # so the unnormalized posterior is den_fixed * Gamma(S; shape, scale_k).
    gamma_S = gamma.pdf(
        S[..., None], a=_gamma_shape, scale=_gamma_scales
    )  # (..., n_theta)

    post_theta_unnorm = den_fixed * gamma_S  # (..., n_theta)
    post_norm = np.sum(post_theta_unnorm, axis=-1, keepdims=True)
    post_theta = post_theta_unnorm / post_norm  # (..., n_theta)

    # ---- Combine --------------------------------------------------------
    # Where a theta carries no alpha mass at this ybar, cond_mean_alpha is
    # nan but its posterior weight is 0; zero those products out safely.
    contrib = np.where(post_theta > 0, cond_mean_alpha * post_theta, 0.0)
    result = np.sum(contrib, axis=-1)
    return result



class OracleBayesRule(BayesRuleEstimator):
    """
    Oracle Bayes rule for alpha.

    Model (per unit, m draws y_t ~ N(alpha, theta)):
        ybar = m^{-1} sum_t y_t             sufficient stat 1
        S    = sum_t (y_t - ybar)^2         sufficient stat 2 (deviation SS)
    For which, given (alpha, theta),
        ybar ~ N(alpha, theta/m),   S ~ theta * chi^2_{m-1},   ybar _||_ S,
    and S depends on theta only (free of alpha).

    Per Proposition 2, use the decomposition:
        E[alpha | ybar, S] = sum_k E[alpha | ybar, theta_k] * p(theta_k | ybar, S)

    Factor 1 -- posterior mean of alpha at fixed theta_k (homoscedastic, so the
    simple ratio-of-sums / Tweedie form is exact):
        E[alpha | ybar, theta_k]
            = sum_j w_{jk} alpha_j phi(ybar; alpha_j, theta_k/m)
              / sum_j w_{jk} phi(ybar; alpha_j, theta_k/m)
    where the sum over j runs over joint support points sharing variance
    theta_k.

    Factor 2 -- posterior over theta, using both statistics (ybar _||_ S | theta):
        p(theta_k | ybar, S)
            ∝ f(S | theta_k) * f(ybar | theta_k)
            = Gamma(S; shape=(m-1)/2, scale=2 theta_k)
              * ( sum_j w_{jk} phi(ybar; alpha_j, theta_k/m) )
    normalized over the distinct theta_k.
    """
    # ^^ Docstring by Claude

    def __init__(self,
                 true_dgp: GaussianMixture,
                 m: int) -> None:
        self.true_dgp = true_dgp
        self.m = m

        # Flat joint support of H: parallel arrays over joint points (alpha_j, theta_j).
        means, variances, weights = (
            true_dgp.mixing_distribution.mean_and_variance_joint_approx_pmf()
        )
        self._means = np.asarray(means, dtype=float)
        self._variances = np.asarray(variances, dtype=float)
        self._weights = np.asarray(weights, dtype=float)
        # Sampling variance of ybar at each joint support point.
        self._ybar_variances = self._variances / self.m

        # Distinct theta values (the variance posterior lives on these), plus a
        # map from each joint support point to its distinct-theta index.
        self._theta_values, self._theta_index = np.unique(
            self._variances, return_inverse=True
        )
        # Gamma parameters for S | theta: S ~ theta * chi^2_{m-1}
        #   => Gamma(shape=(m-1)/2, scale=2*theta).
        self._gamma_shape = (self.m - 1) / 2.0
        self._gamma_scales = 2.0 * self._theta_values  # per distinct theta

    def bayes_rule(self, ybar, S, m):
        """
        Exact oracle posterior mean E[alpha | ybar, S].

        Parameters
        ----------
        ybar : scalar or array-like
            Per-unit sample mean, m^{-1} sum_t y_t.
        S : scalar or array-like
            Per-unit deviation sum of squares, sum_t (y_t - ybar)^2. Must
            broadcast against `ybar`.

        Returns
        -------
        ndarray (or scalar) of E[alpha | ybar, S], matching the broadcast shape
        of (ybar, S).
        """
        # Written by Claude, with my direction about what math should be happening
        # I had to take particular pains on Proposition 2 in the paper

        return bayes_rule(self.true_dgp, ybar=ybar, S=S, m=m)

class NPMLE_BayesRule(BayesRuleEstimator):
    def bayes_rule(self, ybar, S, m, y_test, S_test, m_test):
        """
        Evaluate the bayes rule given data (ybar, S, m) at the
        test point (y_test, S_test, m_test)
        """
        npmle_mix = get_mixing_distribution_npmle(ybar=ybar, S=S, m=m)

        return bayes_rule(gaussian_mixture=npmle_mix, ybar=y_test, S=S_test, m=m_test)