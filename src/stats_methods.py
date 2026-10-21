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
    Oracle Bayes rule for alpha under the true DGP H(alpha, theta), conditioning
    on the FULL sufficient statistics (ybar, S) rather than ybar alone.

    Model (per unit, m draws y_t ~ N(alpha, theta)):
        ybar = m^{-1} sum_t y_t          sufficient stat 1
        S    = m^{-1} sum_t y_t^2        sufficient stat 2 (raw second moment)

    Because S uses raw squares (not deviations from ybar), ybar and S are NOT
    conditionally independent. The clean split uses the deviation sum of
    squares
        W = sum_t (y_t - ybar)^2 = m*S - m*ybar^2 = m*(S - ybar^2),
    for which, given (alpha, theta),
        ybar ~ N(alpha, theta/m),   W ~ theta * chi^2_{m-1},   ybar _||_ W,
    and W depends on theta only (free of alpha). (ybar, S) <-> (ybar, W) is an
    invertible transform, so conditioning on either carries the same info.

    Decomposition (exact, no randomness):
        E[alpha | ybar, S] = sum_k E[alpha | ybar, theta_k] * p(theta_k | ybar, S)

    Factor 1 -- posterior mean of alpha at FIXED theta_k (homoscedastic, so the
    simple ratio-of-sums / Tweedie form is exact):
        E[alpha | ybar, theta_k]
            = sum_j w_{jk} alpha_j phi(ybar; alpha_j, theta_k/m)
              / sum_j w_{jk} phi(ybar; alpha_j, theta_k/m)
    where the sum over j runs over joint support points sharing variance
    theta_k.

    Factor 2 -- posterior over theta, using BOTH statistics (ybar _||_ W | theta):
        p(theta_k | ybar, S)
            ∝ f(W | theta_k) * f(ybar | theta_k)
            = Gamma(W; shape=(m-1)/2, scale=2 theta_k)
              * ( sum_j w_{jk} phi(ybar; alpha_j, theta_k/m) )
    normalized over the distinct theta_k. The constant (in theta) Jacobian of
    the (ybar, W) <-> (ybar, S) map cancels in the normalization.

    Everything is a finite sum over the discrete support of H, evaluated in
    closed form, so no RNG stream is touched. This reduces to the independent
    /ybar-only behavior exactly when H factors and S is uninformative about
    alpha.
    """

    def __init__(self, true_dgp: GaussianMixture, m: int) -> None:
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
        # Gamma parameters for W | theta: W ~ theta * chi^2_{m-1}
        #   => Gamma(shape=(m-1)/2, scale=2*theta).
        self._gamma_shape = (self.m - 1) / 2.0
        self._gamma_scales = 2.0 * self._theta_values  # per distinct theta

    def bayes_rule(self, ybar, S):
        """
        Exact oracle posterior mean E[alpha | ybar, S].

        Parameters
        ----------
        ybar : scalar or array-like
            Per-unit sample mean, m^{-1} sum_t y_t.
        S : scalar or array-like
            Per-unit raw second moment, m^{-1} sum_t y_t^2. Must broadcast
            against `ybar`.

        Returns
        -------
        ndarray (or scalar) of E[alpha | ybar, S], matching the broadcast shape
        of (ybar, S).
        """
        ybar = np.asarray(ybar, dtype=float)
        S = np.asarray(S, dtype=float)

        # Deviation sum of squares recovered from (ybar, S): W = m (S - ybar^2).
        # Clip tiny negatives from float error; W is nonnegative by construction.
        W = self.m * (S - ybar**2)
        W = np.maximum(W, 0.0)

        # Broadcast observations over the joint-support axis (last axis = K points).
        x = ybar[..., None]              # (..., 1)
        mu = self._means                 # (K,)
        var = self._ybar_variances       # (K,)

        # phi(ybar; alpha_j, theta_j/m) at every joint point.
        comp_pdf = _gaussian_pdf(x, mu, var)  # (..., K)

        # ---- Factor 1: E[alpha | ybar, theta_k] for each DISTINCT theta_k ----
        # Numerator/denominator of the fixed-theta posterior mean, accumulated
        # per distinct theta over the joint points that share that theta.
        n_theta = len(self._theta_values)
        batch_shape = comp_pdf.shape[:-1]

        num_fixed = np.zeros(batch_shape + (n_theta,))  # sum_j w_jk alpha_j phi
        den_fixed = np.zeros(batch_shape + (n_theta,))  # sum_j w_jk phi  = f(ybar|theta_k)
        w_phi = self._weights * comp_pdf                 # (..., K)
        np.add.at(
            np.moveaxis(num_fixed, -1, 0),
            self._theta_index,
            np.moveaxis(w_phi * self._means, -1, 0),
        )
        np.add.at(
            np.moveaxis(den_fixed, -1, 0),
            self._theta_index,
            np.moveaxis(w_phi, -1, 0),
        )
        # E[alpha | ybar, theta_k]; den_fixed > 0 wherever any mass is present.
        with np.errstate(invalid="ignore", divide="ignore"):
            cond_mean_alpha = num_fixed / den_fixed  # (..., n_theta)

        # ---- Factor 2: posterior over theta given (ybar, S) ------------------
        # f(ybar | theta_k) is exactly den_fixed (marginal of alpha at theta_k).
        # f(W | theta_k) is the Gamma density. Prior p(theta_k) is already baked
        # into den_fixed via the joint weights w_jk, so the unnormalized posterior
        # is den_fixed * Gamma(W; shape, scale_k).
        gamma_W = gamma.pdf(
            W[..., None], a=self._gamma_shape, scale=self._gamma_scales
        )  # (..., n_theta)

        post_theta_unnorm = den_fixed * gamma_W  # (..., n_theta)
        post_norm = np.sum(post_theta_unnorm, axis=-1, keepdims=True)
        post_theta = post_theta_unnorm / post_norm  # (..., n_theta)

        # ---- Combine --------------------------------------------------------
        # Where a theta carries no alpha mass at this ybar, cond_mean_alpha is
        # nan but its posterior weight is 0; zero those products out safely.
        contrib = np.where(post_theta > 0, cond_mean_alpha * post_theta, 0.0)
        result = np.sum(contrib, axis=-1)
        return result

class NPMLE_BayesRule(BayesRuleEstimator):
    def bayes_rule(self, ybar, S):
        raise NotImplementedError