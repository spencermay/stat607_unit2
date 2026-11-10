# Profiling and optimization for the Unit 2 project

The first baseline profile is of the whole end-to-end figure creation script, with profile at [text](profiles/before__src.figures.mainf5bb7e7.prof). Even including all the figure creation steps which I was worried would muddy the picture, the main culprit is far and away the NPMLE calculation with the cvxpy CLARABEL solver.

I made this before picture more granular by considering each relevant case (mean and variance are drawn independently vs dependently). In the latter case, it naturally took much longer to solve the problem (~7s per call vs .034s in the solver!!!) but in both cases the function call that took the most time was the solver.