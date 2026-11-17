# Profiling and optimization for the Unit 2 project

### Baseline

The first baseline profile is of the whole end-to-end figure creation script, with profile at [before__src*.prof](profiles/before__src.figures.mainf5bb7e7.prof). Even including all the figure creation steps which I was worried would muddy the picture, the main culprit is far and away the NPMLE calculation with the cvxpy CLARABEL solver.

I made this before picture more granular (see [before_dependent_calculations*.prof](profiles/before_dependent_calculationa49d7cb.prof)) by considering the two separate use cases for this method (mean and variance are drawn independently vs dependently). In the latter case, it naturally took much longer to solve the problem (~7s per call vs .06s in the solver!!!) but in both cases the function call that took the most time was the solver.

One interesting thing I found in this was that for the quicker independent mean/variance case, the cvxpy solver took about of a 4th of its time reducing the problem to something a conic solver could handle. So not merely is the actual solving of the convex optimization problem important but if we cared mostly about the independent case then we would also want to maybe speed up the time it takes the solver to run its conic solver reduction.


### Profiling the different solver options

Per the [cvxpy webpage](https://www.cvxpy.org/tutorial/solvers/index.html), I found my installed solvers with:
```
>>> cp.installed_solvers()
['CLARABEL', 'SCS', 'SCIPY', 'HIGHS', 'OSQP']
```

Unfortunately, when I tried seeing which of these improves the time to solve the problem (in speed_comparison_convex_solvers.py), I found that CLARABEL (the original solver I had used) was the only viable option:

------------------------------------------------------------------------
|solver         | best time (s) |     accuracy dist | status|
| ---------|--------------|--------------|----------------------------------
CLARABEL        |      7.1590 |        0.000e+00 | ok
SCS          |             --          |      -- | TIMEOUT (exceeded 15s)
SCIPY         |            --           |     -- | FAILED: SolverError: The solver SCIPY cannot solve this problem.
HIGHS       |              --            |    -- | FAILED: SolverError: The solver HIGHS cannot solve this problem.
OSQP        |              --           |     -- | FAILED: SolverError: The solver OSQP cannot solve this problem.
------------------------------------------------------------------------

So, I tried a new approach. One option is to use a naive Scipy solver (I thought, who knows, maybe it would work), and another is to use active support reduction; because I want a quick comparison, I had GPT write these implementations and I tested in speed_comparison_convex_solvers.py (on a reduced gridsize of 20x20 because 40x40 was too large):

|method|time|error|within tolerance?|
|-----|------|------|------|
| support-reduction            |       0.0328      |   2.482e-06 | ok |
| default (CLARABEL->SCS)      |       1.4711       |  0.000e+00 | ok |
| scipy-SLSQP            |             5.2674    |     9.751e-06 | ok |

From this, support reduction seems like a promising method.

### Frank–Wolfe

### Numerical stability: log likelihood with normalization

$$
\begin{aligned}
\mathrm{argmin}_{f>0, \sum f_i = 1} [1.T @ log(A @ f)] &= argmin_{f>0, \sum f_i = 1} [1.T @ log(A @ f)-c log(1.T @ f)]\\
& = argmin_{f>0, \sum f_i = 1} [1.T @ log(A @ f / 1.T @ f)]
\end{aligned}
$$