"""
Tests for the Gaussian location-scale mixture Bayes rules and the NPMLE solver.

Covers the use-cases exercised while building these:
  * the oracle Bayes rule E[alpha | ybar, S] under a correlated joint H,
  * that the oracle actually uses S (not just ybar),
  * agreement with a brute-force Monte Carlo posterior mean,
  * that the NPMLE solver runs end to end and returns a valid GaussianMixture,
  * Simulation input validation.

Conventions (must match src/):
  ybar = m^{-1} sum_t y_t                  ~ N(alpha, theta/m)
  S    = sum_t (y_t - ybar)^2              ~ theta * chi^2_{m-1}
with ybar _||_ S | (alpha, theta).

Began test suite with Claude.
"""
import numpy as np
import pytest
from scipy.stats import gamma

from data_generating_processes import (
    GaussianMixture,
    MixJointMeanVariance,
)
from stats_methods import (
    OracleBayesRule,
    NaiveEstimator,
    NPMLE_BayesRule,
    get_mixing_distribution_npmle,
    get_joint_mixing_distribution_npmle,
    bayes_rule,
    marginal_bayes_rule,
    _gaussian_pdf,
)
from simulation import Simulation, IndependentMVSimulation


# --- shared fixtures / helpers ---------------------------------------------

M = 11  # observations per unit (matches the paper / ADEMP)


def make_correlated_dgp():
    """A correlated joint H(alpha, theta): larger alpha <-> larger theta."""
    means = np.array([-0.5, 1.0, 3.0])
    variances = np.array([0.5, 2.0, 4.0])
    weights = np.array([0.5, 0.3, 0.2])
    mix = MixJointMeanVariance(means, variances, weights)
    return GaussianMixture(mix), means, variances, weights


def direct_posterior_mean_alpha(means, variances, weights, ybar, S, m):
    """
    Brute-force closed-form E[alpha | ybar, S] over the discrete joint support,
    derived independently of OracleBayesRule's implementation:
        p(k | ybar, S) ∝ w_k * phi(ybar; alpha_k, theta_k/m) * Gamma(S; (m-1)/2, 2 theta_k)
    """
    ybar = np.atleast_1d(np.asarray(ybar, float))
    S = np.atleast_1d(np.asarray(S, float))
    out = np.empty_like(ybar)
    for i in range(len(ybar)):
        py = _gaussian_pdf(ybar[i], means, variances / m)
        ps = gamma.pdf(S[i], a=(m - 1) / 2, scale=2 * variances)
        post = weights * py * ps
        post = post / post.sum()
        out[i] = np.sum(post * means)
    return out


# --- oracle: exactness vs an independent closed form -----------------------

def test_oracle_matches_independent_closed_form():
    dgp, means, variances, weights = make_correlated_dgp()
    oracle = OracleBayesRule(dgp, m=M)

    ybar = np.array([-1.0, 0.0, 0.5, 1.5, 3.0])
    # S is a deviation sum of squares (strictly positive, no ybar coupling here)
    S = np.array([3.0, 8.0, 5.0, 12.0, 20.0])

    got = oracle.bayes_rule(ybar, S, m=M)
    want = direct_posterior_mean_alpha(means, variances, weights, ybar, S, M)

    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_oracle_free_function_and_method_agree():
    """OracleBayesRule.bayes_rule delegates to the module-level bayes_rule."""
    dgp, *_ = make_correlated_dgp()
    oracle = OracleBayesRule(dgp, m=M)
    ybar = np.array([0.2, 1.1])
    S = np.array([4.0, 9.0])
    np.testing.assert_allclose(
        oracle.bayes_rule(ybar, S, m=M),
        bayes_rule(dgp, ybar=ybar, S=S, m=M),
        rtol=0, atol=0,
    )


# --- oracle actually uses S (the whole point of the joint case) ------------

def test_oracle_depends_on_S():
    """Same ybar, different S must give different estimates when alpha and theta
    are correlated (S informs the theta posterior, hence alpha)."""
    dgp, *_ = make_correlated_dgp()
    oracle = OracleBayesRule(dgp, m=M)

    ybar = np.array([1.0, 1.0])
    S = np.array([3.0, 25.0])  # small vs large deviation SS
    est = oracle.bayes_rule(ybar, S, m=M)
    assert not np.isclose(est[0], est[1]), (
        "estimate should change with S under correlated H"
    )


def test_oracle_scalar_and_array_consistent():
    dgp, *_ = make_correlated_dgp()
    oracle = OracleBayesRule(dgp, m=M)
    scalar = oracle.bayes_rule(0.7, 5.0, m=M)
    arr = oracle.bayes_rule(np.array([0.7]), np.array([5.0]), m=M)
    np.testing.assert_allclose(np.asarray(scalar).reshape(()), arr[0])


# --- correlated sampling preserves the alpha/theta pairing -----------------

def test_correlated_sampling_preserves_pairing():
    """A diagonal joint pmf means each alpha is drawn only with its paired theta."""
    means = np.array([-0.5, 1.0, 3.0])
    variances = np.array([0.5, 2.0, 4.0])
    weights = np.array([0.4, 0.3, 0.3])
    mix = MixJointMeanVariance(means, variances, weights)

    rng = np.random.default_rng(0)
    a, th = mix.generate_mean_and_variance(rng, 50_000)

    for mu, var in zip(means, variances):
        paired = th[np.isclose(a, mu)]
        assert np.all(np.isclose(paired, var)), (
            f"alpha={mu} should pair only with theta={var}"
        )


# --- oracle vs brute-force Monte Carlo (statistical correctness) -----------

@pytest.mark.slow
def test_oracle_matches_monte_carlo():
    dgp, means, variances, weights = make_correlated_dgp()
    oracle = OracleBayesRule(dgp, m=M)

    rng = np.random.default_rng(12345)
    N = 2_000_000
    idx = rng.choice(len(weights), size=N, p=weights)
    a = means[idx]
    th = variances[idx]
    y = a[:, None] + np.sqrt(th)[:, None] * rng.standard_normal((N, M))
    ybar = y.mean(axis=1)
    S = ((y - ybar[:, None]) ** 2).sum(axis=1)  # deviation sum of squares

    orc = oracle.bayes_rule(ybar, S, m=M)

    # Bias-free comparison: within each (ybar, S) cell, the mean of the true
    # alpha should match the mean of the oracle estimate (removes the
    # cell-center approximation). Compare only well-populated cells.
    yb_edges = np.quantile(ybar, np.linspace(0, 1, 13))
    S_edges = np.quantile(S, np.linspace(0, 1, 13))
    iy = np.clip(np.digitize(ybar, yb_edges) - 1, 0, len(yb_edges) - 2)
    iS = np.clip(np.digitize(S, S_edges) - 1, 0, len(S_edges) - 2)

    max_err, compared = 0.0, 0
    for by in range(len(yb_edges) - 1):
        for bs in range(len(S_edges) - 1):
            sel = (iy == by) & (iS == bs)
            if sel.sum() < 2000:
                continue
            err = abs(a[sel].mean() - orc[sel].mean())
            max_err = max(max_err, err)
            compared += 1

    assert compared > 20, "too few populated cells to trust the comparison"
    assert max_err < 0.05, f"oracle vs MC worst-cell error too large: {max_err}"


# --- NPMLE solver end to end -----------------------------------------------

def test_npmle_solver_returns_valid_mixture():
    """The convex solver runs and returns a GaussianMixture with proper pmfs."""
    sim = IndependentMVSimulation(
        n=400, m=M,
        mean_support_pts=np.array([-0.5, 1.0, 3.0]),
        variance_support_pts=np.array([0.5, 2.0, 4.0]),
        mean_pmf=np.full(3, 1 / 3),
        variance_pmf=np.full(3, 1 / 3),
    )
    ybar, S = sim.generate_statistics(seed=7)

    mix = get_mixing_distribution_npmle(ybar, S, M, gridsize_mean=200, gridsize_S=201)
    assert isinstance(mix, GaussianMixture)

    md = mix.mixing_distribution
    mp = md.mean_distribution.weights
    vp = md.var_distribution.weights
    mg = md.mean_distribution.support_points
    vg = md.var_distribution.support_points

    # pmfs are valid: nonnegative, sum to 1, shape matches their support
    for g, p in ((mg, mp), (vg, vp)):
        assert p.shape == g.shape
        assert np.all(p >= 0)
        assert np.isclose(p.sum(), 1.0)

    # variance mean is recovered to a reasonable tolerance (true E[theta] = 2.167)
    assert abs(np.sum(vg * vp) - (0.5 + 2.0 + 4.0) / 3) < 0.5


def test_npmle_bayes_rule_runs_and_is_finite():
    sim = IndependentMVSimulation(
        n=300, m=M,
        mean_support_pts=np.array([-0.5, 1.0, 3.0]),
        variance_support_pts=np.array([0.5, 2.0, 4.0]),
        mean_pmf=np.full(3, 1 / 3),
        variance_pmf=np.full(3, 1 / 3),
    )
    ybar, S = sim.generate_statistics(seed=3)

    # Fit on the training data (constructor), evaluate at test points.
    estimator = NPMLE_BayesRule(ybar=ybar, S=S, m=M)
    est = estimator.bayes_rule(ybar=ybar, S=S, m=M)
    est = np.asarray(est)
    assert est.shape == ybar.shape
    assert np.all(np.isfinite(est))


# --- naive estimator -------------------------------------------------------

def test_naive_is_identity_on_ybar():
    ybar = np.array([0.1, -2.0, 3.3])
    np.testing.assert_array_equal(NaiveEstimator().bayes_rule(ybar, S=None), ybar)


# --- Simulation wiring + validation ----------------------------------------

def test_simulation_builds_dgp_and_statistics():
    sim = IndependentMVSimulation(
        n=50, m=M,
        mean_support_pts=np.array([-0.5, 1.0, 3.0]),
        variance_support_pts=np.array([0.5, 2.0, 4.0]),
        mean_pmf=np.full(3, 1 / 3),
        variance_pmf=np.full(3, 1 / 3),
    )
    assert np.isclose(sim.joint_pmf.sum(), 1.0)
    assert sim.joint_pmf.shape == (3, 3)

    ybar, S = sim.generate_statistics(seed=0)
    assert ybar.shape == (50,)
    assert S.shape == (50,)
    assert np.all(S >= 0)  # deviation sum of squares is nonnegative


def test_simulation_rejects_bad_m():
    with pytest.raises(ValueError):
        Simulation(
            n=10, m=1,  # m must be >= 2 for S to have positive dof
            mean_support_pts=np.array([0.0, 1.0]),
            variance_support_pts=np.array([1.0, 2.0]),
            joint_pmf=np.full((2, 2), 0.25),
        )


def test_simulation_rejects_mismatched_joint_pmf_shape():
    with pytest.raises(ValueError):
        Simulation(
            n=10, m=M,
            mean_support_pts=np.array([0.0, 1.0, 2.0]),   # 3 means
            variance_support_pts=np.array([1.0, 2.0]),     # 2 variances
            joint_pmf=np.full((2, 2), 0.25),               # wrong: should be (3, 2)
        )


# --- marginal Bayes rule E[alpha | ybar] -----------------------------------

def test_oracle_marginal_matches_closed_form():
    """
    marginal_bayes_rule (S integrated out over the discrete theta support)
    equals a brute-force closed-form E[alpha | ybar] computed directly from the
    joint support -- an independent derivation.
    """
    dgp, means, variances, weights = make_correlated_dgp()
    ybar = np.array([-1.0, 0.0, 0.5, 1.5, 3.0])

    # Direct E[alpha | ybar]: posterior over joint support weighted by
    # f(ybar | alpha_k, theta_k) only (S integrated out).
    want = np.empty_like(ybar)
    for i in range(len(ybar)):
        py = _gaussian_pdf(ybar[i], means, variances / M)
        post = weights * py
        post = post / post.sum()
        want[i] = np.sum(post * means)

    got = marginal_bayes_rule(dgp, ybar, m=M)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    # method on the oracle agrees with the module-level function
    np.testing.assert_allclose(
        OracleBayesRule(dgp, m=M).marginal_bayes_rule(ybar, m=M), got
    )


@pytest.mark.slow
def test_npmle_marginal_runs_across_seeds():
    """The NPMLE marginal Bayes rule fits and evaluates (finite, right shape)
    across several seeds -- guards against solver flakiness."""
    y_test = np.linspace(-2.0, 5.0, 30)

    for seed in range(4):
        sim = IndependentMVSimulation(
            n=300, m=M,
            mean_support_pts=np.array([-0.5, 1.0, 3.0]),
            variance_support_pts=np.array([0.5, 2.0, 4.0]),
            mean_pmf=np.full(3, 1 / 3),
            variance_pmf=np.full(3, 1 / 3),
        )
        ybar, S = sim.generate_statistics(seed=seed)

        # Fit on training data (constructor), evaluate the marginal at test ybar.
        estimator = NPMLE_BayesRule(ybar=ybar, S=S, m=M)
        marg = estimator.marginal_bayes_rule(ybar=y_test, m=M)
        marg = np.asarray(marg)
        assert marg.shape == y_test.shape, f"seed {seed}: bad shape {marg.shape}"
        assert np.all(np.isfinite(marg)), f"seed {seed}: non-finite output"


# --- joint NPMLE (correlated alpha, theta) ---------------------------------

def _cov_alpha_theta(mixing_distribution):
    """Cov(alpha, theta) under a mixing distribution's joint pmf."""
    a, t, w = mixing_distribution.mean_and_variance_joint_approx_pmf()
    a, t, w = np.asarray(a), np.asarray(t), np.asarray(w)
    return float(np.sum(w * a * t) - np.sum(w * a) * np.sum(w * t))


def _draw_correlated(seed, n=400, m=M):
    """Draw (ybar, S) from a POSITIVELY correlated joint DGP (large alpha paired
    with large theta), returning also the true Cov(alpha, theta)."""
    dgp, means, variances, weights = make_correlated_dgp()
    rng = np.random.default_rng(seed)
    y, _alpha, _theta = dgp.generate(rng, n, m, return_latents=True)
    ybar = y.mean(axis=1)
    S = ((y - ybar[:, None]) ** 2).sum(axis=1)
    true_cov = (
        np.sum(weights * means * variances)
        - np.sum(weights * means) * np.sum(weights * variances)
    )
    return ybar, S, float(true_cov)


def test_joint_npmle_returns_joint_mixture():
    """The joint NPMLE returns a MixJointMeanVariance over a nonempty support."""
    ybar, S, _ = _draw_correlated(0)
    mix = get_joint_mixing_distribution_npmle(ybar, S, M)
    assert isinstance(mix, GaussianMixture)
    assert isinstance(mix.mixing_distribution, MixJointMeanVariance)
    a, t, w = mix.mixing_distribution.mean_and_variance_joint_approx_pmf()
    assert len(a) == len(t) == len(w) >= 1
    assert np.isclose(np.sum(w), 1.0)


@pytest.mark.slow
def test_joint_npmle_recovers_positive_correlation():
    """
    On a positively-correlated DGP, the joint NPMLE recovers positive
    Cov(alpha, theta), while the independent NPMLE cannot (its outer-product
    construction forces Cov == 0 regardless of the data).
    """
    for seed in range(3):
        ybar, S, true_cov = _draw_correlated(seed)
        assert true_cov > 0  # sanity: the DGP really is positively correlated

        joint_mix = get_joint_mixing_distribution_npmle(ybar, S, M)
        joint_cov = _cov_alpha_theta(joint_mix.mixing_distribution)

        indep_mix = get_mixing_distribution_npmle(ybar, S, M)
        indep_cov = _cov_alpha_theta(indep_mix.mixing_distribution)

        # Joint fit picks up the positive dependence...
        assert joint_cov > 0.1, (
            f"seed {seed}: joint Cov={joint_cov:.3f} should be clearly positive"
        )
        # ...and does so far better than the independent fit, which is ~0.
        assert abs(indep_cov) < 1e-6, (
            f"seed {seed}: independent Cov={indep_cov:.2e} should be ~0"
        )
        assert joint_cov > indep_cov
