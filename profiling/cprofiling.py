from cProfile import Profile
from pstats import SortKey, Stats

import os
import sys
import subprocess

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
# print(SRC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import src.figures

import numpy as np
from src.simulation import IndependentMVSimulation, Simulation
from src.metrics import NPMLE_BayesRule

def get_cprofile(func: callable, filename:str|None=None, folder="profiling/profiles/", verbose=True):
    with Profile() as profile:
        func()

        if verbose:
            (Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.TIME)
                .print_stats(15))
        if filename is not None:
            if filename[-6:] == ".prof": filename = filename[:-5]
            profile.dump_stats(os.path.join(
                        folder, filename + _get_git_hash()+".prof"
                    ))

def _get_git_hash() -> str:
    # This git-hash retrieving helper is by Gemini (at my request)
    #  so I can better track what code corresponds to before and after 
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


#############



def independent_mv_simulation_setup():
    return IndependentMVSimulation(
            n=400,
            m=11,
            mean_support_pts=np.array([-0.5, 1.0, 3.0]),
            variance_support_pts=np.array([0.5, 2.0, 4.0]),
            mean_pmf=np.full(3, 1 / 3),
            variance_pmf=np.full(3, 1 / 3),
        )

def dependent_mv_simulation_setup():
    mean_support = np.array([-0.5, 1.0, 3.0])
    variance_support = np.array([0.5, 2.0, 4.0])
    # joint_pmf[i, j] = P(alpha = mean_support[i], theta = variance_support[j]).
    # Diagonal-heavy -> positive correlation; small off-diagonal mass keeps the
    # support fully explored and both marginals non-degenerate.
    correlated_joint_pmf = np.array([
        [0.28, 0.04, 0.01],   # alpha = -0.5 mostly with theta = 0.5
        [0.04, 0.25, 0.04],   # alpha =  1.0 mostly with theta = 2.0
        [0.01, 0.04, 0.29],   # alpha =  3.0 mostly with theta = 4.0
    ])
    correlated_joint_pmf = correlated_joint_pmf / correlated_joint_pmf.sum()
    return Simulation(
            n=400,
            m=11,
            mean_support_pts=mean_support,
            variance_support_pts=variance_support,
            joint_pmf=correlated_joint_pmf,
        )

def generate_data_from_simulation_setup(simulation, seed=0):
    ybar, S = simulation.generate_statistics(seed)
    m = simulation.m
    return ybar, S, m

def npmle_calculations(ybar, S, m, assume_independent_mv=True):
    npmle_estimator = NPMLE_BayesRule(ybar, S=S, m=m, assume_independent_mv=assume_independent_mv)
    return npmle_estimator



if __name__ == "__main__":
    # get_cprofile(src.figures.main, filename="before__src.figures.main"+_get_git_hash()+".prof")

    independent_data = generate_data_from_simulation_setup(independent_mv_simulation_setup())
    dependent_data = generate_data_from_simulation_setup(dependent_mv_simulation_setup())

    get_cprofile(lambda: npmle_calculations(*independent_data, assume_independent_mv=True),
                 filename="before_independent_calculation")
    get_cprofile(lambda: npmle_calculations(*dependent_data, assume_independent_mv=False),
                 filename="before_dependent_calculation")