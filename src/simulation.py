"""
Run the simulation study!
"""

import numpy as np

from data_generating_processes import GaussianMixture, MixJointMeanVariance


class Simulation():
    def __init__(self,
                 n: int,
                 m: int,
                 mean_support_pts: np.ndarray,
                 variance_support_pts: np.ndarray,
                 joint_pmf: np.ndarray,):
        """
        Specify a setup by the joint distribution on the (/a possibly approximate)
        pmf that describes the joint distribution of (alpha,theta)=mean, variance of
        mixing distribution.

        joint_pmf[i,j] = pmf(mean_support_pts[i], variance_support_pts[j])

        Parameters
        ----------
        n : int
            Number of units (independent (alpha_i, theta_i) draws) per trial.
        m : int
            Number of observations per unit. Must be >= 2 so the deviation sum
            of squares S has positive degrees of freedom (m - 1).
        mean_support_pts : np.ndarray, shape (I,)
            Support points for alpha.
        variance_support_pts : np.ndarray, shape (J,)
            Support points for theta (all strictly positive).
        joint_pmf : np.ndarray, shape (I, J)
            joint_pmf[i, j] = P(alpha = mean_support_pts[i],
                                theta = variance_support_pts[j]).
        """
        self.mean_support_pts = np.asarray(mean_support_pts, dtype=float)
        self.variance_support_pts = np.asarray(variance_support_pts, dtype=float)
        self.joint_pmf = np.asarray(joint_pmf, dtype=float)

        if not (isinstance(n, (int, np.integer)) and n >= 1):
            raise ValueError(f"n must be a positive integer; got {n!r}")
        if not (isinstance(m, (int, np.integer)) and m >= 2):
            raise ValueError(
                f"m must be an integer >= 2 (need m-1 > 0 dof for S); got {m!r}"
            )
        if self.mean_support_pts.ndim != 1 or self.variance_support_pts.ndim != 1:
            raise ValueError("support point arrays must be 1-D")
        expected_shape = (len(self.mean_support_pts), len(self.variance_support_pts))
        if self.joint_pmf.shape != expected_shape:
            raise ValueError(
                "joint_pmf shape must be (len(mean_support_pts), "
                f"len(variance_support_pts)) = {expected_shape}; got "
                f"{self.joint_pmf.shape}"
            )

        self.n = int(n)
        self.m = int(m)

        # The following is very much Claude's doing.
        # Build the true data-generating process from the flattened joint pmf.
        # Flatten in C-order so that entry k = i * J + j corresponds to
        # (mean_support_pts[i], variance_support_pts[j]), matching the documented
        # joint_pmf[i, j] convention and the flattening used elsewhere
        # (OracleBayesRule / MixingDistribution.mean_and_variance_joint_approx_pmf).
        n_var = len(self.variance_support_pts)
        n_mean = len(self.mean_support_pts)
        means_flat = np.repeat(self.mean_support_pts, n_var)
        variances_flat = np.tile(self.variance_support_pts, n_mean)
        weights_flat = self.joint_pmf.ravel()  # C-order

        self.mixing_distribution = MixJointMeanVariance(
            means_flat, variances_flat, weights_flat
        )
        self.dgp = GaussianMixture(self.mixing_distribution)

    def generate_statistics(self, rng):
        """
        Draw one trial of n units (m observations each) and return the
        sufficient statistics used by the estimators.

        Returns
        -------
        ybar : np.ndarray, shape (n,)
            Per-unit sample mean, m^{-1} sum_t y_{it}.
        S : np.ndarray, shape (n,)
            Per-unit deviation sum of squares, sum_t (y_{it} - ybar_i)^2.
            (S | theta ~ theta * chi^2_{m-1}, the convention the NPMLE and the
            oracle Bayes rule expect.)
        """
        y = self.dgp.generate(rng, self.n, self.m)  # (n, m)
        ybar = y.mean(axis=1)                        # (n,)
        S = ((y - ybar[:, None]) ** 2).sum(axis=1)   # (n,) deviation SS
        return ybar, S


class IndependentMVSimulation(Simulation):
    def __init__(self,
                     n: int,
                     m: int,
                     mean_support_pts: np.ndarray,
                     variance_support_pts: np.ndarray,
                     mean_pmf: np.ndarray,
                     variance_pmf: np.ndarray):
        # Independent mean/variance => joint pmf is the outer product:
        #   joint_pmf[i, j] = mean_pmf[i] * variance_pmf[j]
        super().__init__(
            n,
            m,
            mean_support_pts,
            variance_support_pts,
            joint_pmf=np.outer(mean_pmf, variance_pmf),
        )
