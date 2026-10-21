"""
Put the project's src/ directory on sys.path so tests can import the modules
(data_generating_processes, stats_methods, simulation) directly.

Learned this "separating the silly import src code from the rest of the test cases"
trick from AI, I like it a lot and will be using it in the future.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
