"""
Run the simulation study!
"""

import numpy as np
from data_generating_processes import GaussianMixture

class Simulation():
    def __init__(self,
                 n: int,
                 m: int | np.ndarray,
                 mean_support_pts: np.ndarray,
                 variance_support_pts: np.ndarray,
                 joint_pmf: np.ndarray,):
        """
        Specify a setup by the joint distribution on the (/a possibly approximate)
        pmf that describes the joint distribution of (alpha,theta)=mean, variance of
        mixing distribution.
        """
        self.n=n
        self.m=m
        self.mean_support_pts = mean_support_pts
        self.variance_support_pts = variance_support_pts
        self.joint_pmf = joint_pmf

class IndependentMVSimulation(Simulation):
    def __init__(self,
                     n: int,
                     m: int | np.ndarray,
                     mean_support_pts: np.ndarray,
                     variance_support_pts: np.ndarray,
                     mean_pmf: np.ndarray,
                     variance_pmf: np.ndarray):
        self.n=n
        self.m=m
        self.mean_support_pts = mean_support_pts
        self.variance_support_pts = variance_support_pts
        self.mean_pmf = mean_pmf
        self.variance_pmf = variance_pmf
        return super().__init__(n, m, mean_support_pts, variance_support_pts, joint_pmf=np.outer(mean_pmf, variance_pmf))