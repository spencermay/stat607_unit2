"""
Monte Carlo risk of a Bayes-rule estimator (out-of-sample).

We estimate the mean squared error of an estimator's rule for alpha by, in each
replicate, fitting the estimator on one draw from the true DGP (the TRAIN set)
and evaluating its rule on a second, INDEPENDENT draw (the TEST set), then
averaging squared errors. Train and test are separate draws because the risk
of interest is generalization to fresh data: evaluating on the training draw
would measure in-sample fit, which is optimistically biased for estimators that
fit the mixing distribution to the data (e.g. NPMLE). Two targets are
supported:

  * target="oracle": compare the estimator's rule to the oracle Bayes rule
    E[alpha | ybar, S] on the TEST set. This is the excess risk / regret over
    the optimal rule (how much is lost by not knowing the true mixing
    distribution).
  * target="alpha": compare the estimator's rule to the true latent alpha of
    the TEST set. This is the total risk of estimating alpha.

Both the point estimate (MSE) and the Monte Carlo standard error of that
estimate are returned. The independent replication unit is the (train, test)
replicate, so the MC standard error is computed across replicates (test units
within a replicate share the same fitted estimator and are not independent).
"""

import numpy as np
from tqdm import tqdm

from simulation import Simulation, IndependentMVSimulation
from stats_methods import (
    BayesRuleEstimator,
    OracleBayesRule,
    NaiveEstimator,
    NPMLE_BayesRule,
)


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

    For each of n_sim replicates:
      train draw (ybar_tr, S_tr) ~ DGP -> estimator = make_estimator(ybar_tr, S_tr, m)
      test draw  (ybar_te, S_te, alpha_te) ~ DGP   (independent of train)
      est = estimator.bayes_rule(ybar_te, S_te, m)
      per-replicate MSE = mean_i (est_i - target_i)^2
    where target_i is the oracle rule E[alpha | ybar_te_i, S_te_i]
    (target="oracle") or the true latent alpha_te_i (target="alpha"), both from
    the test draw.

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
    # Independent child RNG streams, one per replicate, reproducible from
    # `seed`. Each replicate spawns two further independent sub-streams: one
    # for the TRAIN draw (used to fit the estimator) and one for the TEST draw
    # (fresh data on which the fitted rule's risk is measured). Keeping them
    # independent is the point: evaluating on the training draw would measure
    # in-sample fit, which is optimistically biased for estimators that fit the
    # mixing distribution to the data (e.g. NPMLE).

    # This updated method was my idea, but Claude wrote the boilerplate
    child_seeds = np.random.SeedSequence(seed).spawn(n_sim)

    per_draw_mse = np.empty(n_sim)
    draws = tqdm(
        enumerate(child_seeds),
        total=n_sim,
        disable=not progress,
        desc=f"MC MSE (target={target})",
    )
    for k, child in draws:
        train_seq, test_seq = child.spawn(2)

        # TRAIN: draw data and fit the estimator on it.
        train_rng = np.random.default_rng(train_seq)
        train_ybar, train_S, _train_alpha, _train_theta = simulation._draw(train_rng)
        estimator = make_estimator(train_ybar, train_S, m)

        # TEST: independent fresh draw; evaluate the fitted rule here.
        test_rng = np.random.default_rng(test_seq)
        test_ybar, test_S, test_alpha, _test_theta = simulation._draw(test_rng)
        est = np.asarray(estimator.bayes_rule(test_ybar, test_S, m))

        if target == "oracle":
            # Oracle rule evaluated at the TEST inputs.
            comparison = np.asarray(
                oracle_bayes_rule.bayes_rule(test_ybar, test_S, m)
            )
        else:  # target == "alpha"
            comparison = test_alpha  # true latent alpha of the TEST draw

        per_draw_mse[k] = np.mean((est - comparison) ** 2)

    mse = float(np.mean(per_draw_mse))
    if n_sim == 1:
        mc_standard_error = float("nan")  # undefined with a single draw
    else:
        mc_standard_error = float(
            np.std(per_draw_mse, ddof=1) / np.sqrt(n_sim)
        )
    return mse, mc_standard_error


if __name__ == "__main__":
    # For the pretty formatting in the output table, thank AI.

    # Compare the Monte Carlo risk of the naive, oracle, and NPMLE Bayes rules
    # in two DGP settings: alpha independent of theta, and alpha positively
    # correlated with theta. For each estimator we report both risk targets:
    #   regret  = MSE vs the oracle rule E[alpha | ybar, S]  (target="oracle")
    #   total   = MSE vs the true latent alpha               (target="alpha")
    #
    # NPMLE refits on every replicate (and the joint fit solves a convex problem
    # taking seconds), so it uses a smaller n_sim than the closed-form rules.

    SEED = 0
    N_SIM_FAST = 100   # oracle / naive: cheap closed-form rules
    # NPMLE refits per replicate and the JOINT fit solves a convex problem
    # taking seconds, so keep this small. Each npmle row is
    #   n_sim * 2 targets * 2 draws-per-replicate  convex solves; bump it up
    # (e.g. 50-100) for a publication-quality estimate, accepting the runtime.
    N_SIM_NPMLE = 10

    mean_support = np.array([-0.5, 1.0, 3.0])
    variance_support = np.array([0.5, 2.0, 4.0])

    # Setting 1: independent alpha, theta (uniform marginals -> outer product).
    independent_sim = IndependentMVSimulation(
        n=400, m=11,
        mean_support_pts=mean_support,
        variance_support_pts=variance_support,
        mean_pmf=np.full(3, 1 / 3),
        variance_pmf=np.full(3, 1 / 3),
    )

    # Setting 2: positively correlated alpha, theta via a diagonal-heavy joint
    # pmf (does NOT factor), with the same (uniform) marginals.
    correlated_joint_pmf = np.array([
        [0.28, 0.04, 0.01],
        [0.04, 0.25, 0.04],
        [0.01, 0.04, 0.29],
    ])
    correlated_joint_pmf = correlated_joint_pmf / correlated_joint_pmf.sum()
    correlated_sim = Simulation(
        n=400, m=11,
        mean_support_pts=mean_support,
        variance_support_pts=variance_support,
        joint_pmf=correlated_joint_pmf,
    )

    settings = [
        ("independent", independent_sim),
        ("correlated", correlated_sim),
    ]

    # (name, make_estimator factory, n_sim). The NPMLE factory uses the joint
    # fit in the correlated setting and the independent fit otherwise -- bound
    # per setting below.
    def build_estimators(sim):
        oracle = OracleBayesRule(sim.dgp, m=sim.m)
        return oracle, [
            ("naive", lambda yb, s, m: NaiveEstimator(), N_SIM_FAST),
            ("oracle", lambda yb, s, m: OracleBayesRule(sim.dgp, m=m), N_SIM_FAST),
            ("npmle(indep)",
             lambda yb, s, m: NPMLE_BayesRule(yb, s, m, assume_independent_mv=True),
             N_SIM_NPMLE),
            ("npmle(joint)",
             lambda yb, s, m: NPMLE_BayesRule(yb, s, m, assume_independent_mv=False),
             N_SIM_NPMLE),
        ]

    # Human-readable labels and descriptions for each estimator row.
    ESTIMATOR_LABELS = {
        "naive": "Naive (no shrinkage, alpha = ybar)",
        "oracle": "Oracle (knows true H)",
        "npmle(indep)": "NPMLE (independence assumed)",
        "npmle(joint)": "NPMLE (joint alpha, theta)",
    }
    SETTING_LABELS = {
        "independent": "Setting A: alpha independent of theta",
        "correlated": "Setting B: alpha positively correlated with theta",
    }

    # --- run all (setting, estimator) cells, collecting results -------------
    results = {}  # (setting_name, est_name) -> dict of risks + n_sim
    for setting_name, sim in settings:
        oracle, estimators = build_estimators(sim)
        for est_name, make_est, n_sim in estimators:
            regret, regret_se = mc_mse_bayes_rule(
                sim, make_est, oracle,
                n_sim=n_sim, seed=SEED, target="oracle", progress=False,
            )
            total, total_se = mc_mse_bayes_rule(
                sim, make_est, oracle,
                n_sim=n_sim, seed=SEED, target="alpha", progress=False,
            )
            results[(setting_name, est_name)] = dict(
                regret=regret, regret_se=regret_se,
                total=total, total_se=total_se, n_sim=n_sim,
            )

    # --- render a labeled, self-documenting report -------------------------
    COL_EST, COL_RISK = 36, 22
    TABLE_WIDTH = COL_EST + 2 * COL_RISK + 4

    def fmt_risk(mean, se):
        return f"{mean:.4f} ± {se:.4f}"

    print()
    print("=" * TABLE_WIDTH)
    print("Monte Carlo risk of Bayes-rule estimators for the mean (alpha)".center(TABLE_WIDTH))
    print("Gaussian location-scale mixture;  n = 400 units,  m = 11 obs/unit".center(TABLE_WIDTH))
    print("=" * TABLE_WIDTH)
    print()
    print("Each cell is mean squared error ± Monte Carlo standard error, estimated")
    print("out-of-sample (estimator fit on a training draw, evaluated on an")
    print("independent test draw). Two risk measures are reported:")
    print()
    print("  Regret vs oracle : MSE of the estimator's rule against the oracle rule")
    print("                     E[alpha | ybar, S]  -- excess risk from not knowing H.")
    print("  Total vs alpha   : MSE against the true latent alpha -- total estimation risk.")
    print()
    print("Lower is better; the oracle attains zero regret by definition.")
    print()

    for setting_name, _sim in settings:
        print(SETTING_LABELS[setting_name])
        print("-" * TABLE_WIDTH)
        print(f"{'Estimator':<{COL_EST}}  {'Regret vs oracle':<{COL_RISK}}  {'Total vs alpha':<{COL_RISK}}")
        print("-" * TABLE_WIDTH)
        for est_name in ESTIMATOR_LABELS:
            res = results[(setting_name, est_name)]
            print(
                f"{ESTIMATOR_LABELS[est_name]:<{COL_EST}}  "
                f"{fmt_risk(res['regret'], res['regret_se']):<{COL_RISK}}  "
                f"{fmt_risk(res['total'], res['total_se']):<{COL_RISK}}"
            )
        print()

    # Footnote: which rows used how many Monte Carlo replicates.
    n_fast = results[("independent", "oracle")]["n_sim"]
    n_npmle = results[("independent", "npmle(indep)")]["n_sim"]
    print("-" * TABLE_WIDTH)
    print(f"Monte Carlo replicates: closed-form rules (naive, oracle) n_sim = {n_fast};")
    print(f"NPMLE rows n_sim = {n_npmle} (each refits per replicate, so kept smaller).")
    print("=" * TABLE_WIDTH)