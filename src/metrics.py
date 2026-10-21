"""
Monte Carlo risk of a Bayes-rule estimator.

We estimate the mean squared error of an estimator's rule for alpha by
repeatedly drawing from the true DGP, building the estimator on each draw,
and averaging squared errors. Two targets are supported:

  * target="oracle": compare the estimator's rule to the oracle Bayes rule
    E[alpha | ybar, S]. This is the excess risk / regret over the optimal rule
    (how much is lost by not knowing the true mixing distribution).
  * target="alpha": compare the estimator's rule to the true latent alpha that
    generated each unit. This is the total risk of estimating alpha.

Both the point estimate (MSE) and the Monte Carlo standard error of that
estimate are returned. The independent replication unit is the simulation
draw, so the MC standard error is computed across draws (units within a draw
share the same dataset / fitted estimator and are not independent).
"""

import numpy as np
from tqdm import tqdm

from simulation import Simulation
from stats_methods import BayesRuleEstimator, OracleBayesRule


def mc_mse_bayes_rule(
                simulation: Simulation,
                make_estimator,
                oracle_bayes_rule: OracleBayesRule,
                *,
                n_sim: int = 100,
                seed: int = 0,
                target: str = "oracle",
                progress: bool = True,
                ):
    """
    Monte Carlo MSE of an estimator's Bayes rule, with its MC standard error.

    For each of n_sim draws (ybar, S, alpha) ~ DGP:
      est = make_estimator(ybar, S, m).bayes_rule(ybar, S, m)
      per-draw MSE = mean_i (est_i - target_i)^2
    where target_i is the oracle rule E[alpha | ybar_i, S_i] (target="oracle")
    or the true latent alpha_i (target="alpha").

    Parameters
    ----------
    simulation : Simulation
        Carries the true DGP, n, and m, and draws (ybar, S, alpha) through its
        _draw helper (no per-draw CSV written).
    make_estimator : callable
        make_estimator(ybar, S, m) -> BayesRuleEstimator, fitted/ready to call
        .bayes_rule(ybar, S, m). This factory form handles estimators that fit
        on the data in their constructor (e.g. NPMLE_BayesRule) as well as those
        that don't (oracle, naive) uniformly. Example:
            lambda yb, s, m: NPMLE_BayesRule(yb, s, m)
            lambda yb, s, m: NaiveEstimator()
    oracle_bayes_rule : OracleBayesRule
        The oracle, used as the comparison target when target="oracle".
    n_sim : int, default 100
        Number of Monte Carlo draws.
    seed : int, default 0
        Base seed; draw k uses an independent child stream of SeedSequence(seed).
    target : {"oracle", "alpha"}, default "oracle"
        Comparison target (see module docstring).
    progress : bool, default True
        Show a tqdm progress bar over the draws. Set False for quiet runs
        (e.g. inside tests).

    Returns
    -------
    mse : float
        Monte Carlo estimate of the MSE (mean of per-draw MSEs).
    mc_standard_error : float
        Standard error of `mse` across the n_sim draws:
        std(per-draw MSEs, ddof=1) / sqrt(n_sim).
    """
    if target not in ("oracle", "alpha"):
        raise ValueError(f"target must be 'oracle' or 'alpha'; got {target!r}")
    if n_sim < 1:
        raise ValueError(f"n_sim must be >= 1; got {n_sim}")

    m = simulation.m
    # Independent child RNG streams, one per draw, reproducible from `seed`.
    child_seeds = np.random.SeedSequence(seed).spawn(n_sim)

    per_draw_mse = np.empty(n_sim)
    draws = tqdm(
        enumerate(child_seeds),
        total=n_sim,
        disable=not progress,
        desc=f"MC MSE (target={target})",
    )
    for k, child in draws:
        rng = np.random.default_rng(child)
        ybar, S, alpha, _theta = simulation._draw(rng)

        estimator = make_estimator(ybar, S, m)
        est = np.asarray(estimator.bayes_rule(ybar, S, m))

        if target == "oracle":
            comparison = np.asarray(oracle_bayes_rule.bayes_rule(ybar, S, m))
        else:  # target == "alpha"
            comparison = alpha

        per_draw_mse[k] = np.mean((est - comparison) ** 2)

    mse = float(np.mean(per_draw_mse))
    if n_sim == 1:
        mc_standard_error = float("nan")  # undefined with a single draw
    else:
        mc_standard_error = float(
            np.std(per_draw_mse, ddof=1) / np.sqrt(n_sim)
        )
    return mse, mc_standard_error
