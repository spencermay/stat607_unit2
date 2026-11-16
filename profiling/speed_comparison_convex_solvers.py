"""
Upon consulting https://www.cvxpy.org/tutorial/solvers/index.html, it appears
I have more solvers available that may be better suited to the tasks I want to
solve.

For instance I have these solvers but I've only been using the first 2:
>>> cp.installed_solvers()
['CLARABEL', 'SCS', 'SCIPY', 'HIGHS', 'OSQP']
"""

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
# print(SRC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from src.simulation import IndependentMVSimulation, Simulation
from src.metrics import NPMLE_BayesRule
from src.stats_methods import get_joint_mixing_distribution_npmle


