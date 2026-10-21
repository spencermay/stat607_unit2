"""
Simulation setup and data generation for the Gaussian location-scale mixture
study.
"""

import json
import os

import numpy as np

from data_generating_processes import GaussianMixture, MixJointMeanVariance

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_PROJECT_ROOT, "results", "raw")


def _hash_numpy(arr):
    return hash((arr.tobytes(), arr.shape, arr.dtype))


def _descriptor_to_json(sim_descriptor):
    """Serialize a simulation descriptor dict to a one-line JSON string,
    converting numpy arrays/scalars to plain Python so json can handle them."""
    # This helper by Claude (i didn't how to easily add the descriptor string)
    def _to_builtin(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return x

    return json.dumps({k: _to_builtin(v) for k, v in sim_descriptor.items()})


def _write_raw_csv(ybar, S, sim_descriptor, fname):
    """
    Write the generated (ybar, S) to a CSV, labeled with the simulation
    descriptor.

    The first line is a comment (prefixed with '#') holding the descriptor
    dict serialized as JSON, so the file records exactly which simulation
    setup produced it. The remaining lines are a normal two-column CSV:

        # {"id": ..., "n": 400, "m": 11, "mean_support_pts": [...], ...}
        ybar,S
        <ybar_0>,<S_0>
        ...

    The '#'-prefixed header is skipped by numpy.genfromtxt / pandas.read_csv
    (comment='#'), so the data stays readable while keeping provenance.

    Parameters
    ----------
    ybar, S : array-like, shape (n,)
        Per-unit sufficient statistics to record.
    sim_descriptor : dict
        Describes the simulation (id, n, m, support points, joint pmf). Values
        may be numpy arrays/scalars.
    fname : str
        Destination path. Parent directories are created if needed.
    """
    # helper boilerplate also written by claude
    ybar = np.asarray(ybar)
    S = np.asarray(S)
    if ybar.shape != S.shape:
        raise ValueError(
            f"ybar and S must have the same shape; got {ybar.shape}, {S.shape}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)

    data = np.column_stack([ybar, S])
    header = f"{_descriptor_to_json(sim_descriptor)}\nybar,S"
    # comments='# ' puts the descriptor behind a comment marker; the second
    # header line (ybar,S) is also commented out by np.savetxt, which is the
    # conventional CSV column header for '#'-comment-aware readers.
    np.savetxt(fname, data, delimiter=",", header=header, comments="# ")

class Simulation():
    def __init__(self,
                 n: int,
                 m: int,
                 mean_support_pts: np.ndarray,
                 variance_support_pts: np.ndarray,
                 joint_pmf: np.ndarray,
                 ):
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

        self._id = hash((n,m,_hash_numpy(self.mean_support_pts), _hash_numpy(self.variance_support_pts), _hash_numpy(self.joint_pmf)))
        self._descriptor = {"id": self._id,
                            "n": n,
                            "m": m,
                            "mean_support_pts": (self.mean_support_pts),
                            "variance_support_pts": (self.variance_support_pts),
                            "joint_pmf": (self.joint_pmf),
                            }
        
    @property
    def id(self):
        """Stable hash identifying this simulation setup (used in filenames)."""
        return self._id

    @property
    def descriptor(self):
        """Dict describing this simulation setup (id, n, m, support, pmf)."""
        return self._descriptor

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
        fname = os.path.join(RAW_DIR, f"raw_{self._descriptor['id']}.csv")
        _write_raw_csv(ybar, S, sim_descriptor=self._descriptor, fname=fname)
        return ybar, S

class IndependentMVSimulation(Simulation):
    def __init__(self,
                     n: int,
                     m: int,
                     mean_support_pts: np.ndarray,
                     variance_support_pts: np.ndarray,
                     mean_pmf: np.ndarray,
                     variance_pmf: np.ndarray,
                     ):
        # Independent mean/variance => joint pmf is the outer product:
        #   joint_pmf[i, j] = mean_pmf[i] * variance_pmf[j]
        super().__init__(
            n,
            m,
            mean_support_pts,
            variance_support_pts,
            joint_pmf=np.outer(mean_pmf, variance_pmf),
        )