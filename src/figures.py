"""
Figure-producing functions for the Gaussian location-scale mixture study.

Organization
------------
  * ``plot_*`` functions DRAW onto an Axes and return ``(fig, ax)``. They take
    an optional ``ax=None`` so they can be used standalone (they make their own
    figure) or composed into a multi-panel figure (pass an existing Axes).
    They never save to disk.

  * ``figure_*`` functions COMPOSE several ``plot_*`` panels into one complete
    figure and return it (still no disk I/O).

  * ``save_figure`` is the only helper that knows about the results/ directory
    layout and file naming.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from stats_methods import (
                OracleBayesRule,
                NPMLE_BayesRule,
                get_mixing_distribution_npmle,
                NaiveEstimator
                )
from simulation import IndependentMVSimulation, Simulation


# Anchor output to the project root (parent of src/) so results land in the
# repo's results/ regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "results", "figures")


def _ensure_ax(ax):
    """Return (fig, ax), creating a new figure/axes if ax is None."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def save_figure(fig, name, sim_id):
    """
    Save a figure under results/figures/ with a provenance-tagged filename.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Short descriptive name, e.g. "paper_figures".
    sim_id : hashable
        The simulation's id (Simulation.id), embedded in the filename so a
        figure can be traced back to the setup that produced it.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}_{sim_id}.pdf")
    fig.savefig(path)
    return path


# --- individual panels (plot_*) --------------------------------------------

def plot_mean_pmf(mixing_distribution, *, ax=None, pmf_color="blue"):
    """
    Stem plot of the marginal pmf of alpha (the mean mixing distribution),
    collapsing the joint support onto its distinct mean values.
    """
    fig, ax = _ensure_ax(ax)

    means, _variances, weights = (
        mixing_distribution.mean_and_variance_joint_approx_pmf()
    )
    means = np.asarray(means)
    weights = np.asarray(weights)

    unique_means = np.unique(means)
    # Total mass on each distinct mean (sum weights of joint points sharing it).
    mass = np.array([weights[np.isclose(means, mu)].sum() for mu in unique_means])

    ax.vlines(unique_means, ymin=0, ymax=mass, color=pmf_color)
    ax.set_ylabel("P(alpha)")
    ax.set_xlabel("alpha")
    return fig, ax


def plot_data_rug(ybar, *, ax=None, ymin=-0.1, ymax=-0.01, color="black"):
    """Rug of observed ybar values along the bottom of an axes (overlay)."""
    fig, ax = _ensure_ax(ax)
    ax.vlines(ybar, ymin=ymin, ymax=ymax, color=color)
    return fig, ax


def plot_ybar_hist(ybar, *, ax=None, bins=50):
    """Histogram of the observed per-unit means ybar."""
    fig, ax = _ensure_ax(ax)
    ax.hist(ybar, bins=bins)
    ax.set_xlabel("ybar")
    ax.set_ylabel("count")
    return fig, ax

def _evaluate_bayes_rule_grid(estimate, ybar_range, S_range, n_grid):
    """
    Evaluate a Bayes rule on the Cartesian grid ybar_range x S_range.

    Parameters
    ----------
    estimate : callable
        estimate(ybar, S) -> E[alpha | ybar, S], vectorized over broadcastable
        arrays (e.g. ``lambda yb, s: oracle.bayes_rule(yb, s, m=11)``). A plain
        callable is used rather than a BayesRuleEstimator because the concrete
        estimators take extra arguments (m; the NPMLE needs training data) that
        a uniform interface can't supply.
    ybar_range, S_range : (low, high) tuples
        Grid extents for ybar and S.
    n_grid : int
        Number of grid points per axis.

    Returns
    -------
    ybar_axis : (n_grid,) the ybar grid values (axis 0 of the surface)
    S_axis    : (n_grid,) the S grid values   (axis 1 of the surface)
    surface   : (n_grid, n_grid) with surface[i, j] = E[alpha | ybar_i, S_j]
    """
    ybar_axis = np.linspace(ybar_range[0], ybar_range[1], n_grid)
    S_axis = np.linspace(S_range[0], S_range[1], n_grid)

    # Flatten the full product so the estimator sees matched (ybar, S) pairs:
    #   repeat ybar over the S axis, tile S over the ybar axis (C-order), so
    #   pair k = i*n_grid + j is (ybar_i, S_j). Reshape back to (n_grid, n_grid)
    #   with ybar on axis 0, S on axis 1.
    ybar_flat = np.repeat(ybar_axis, n_grid)
    S_flat = np.tile(S_axis, n_grid)
    values = np.asarray(estimate(ybar_flat, S_flat))
    surface = values.reshape(n_grid, n_grid)
    return ybar_axis, S_axis, surface


def plot_bayes_rule_marginal_on_ybar(marginal_estimate,
                                     ybar_range,
                                     *,
                                     ax=None,
                                     n_grid=200):
    """
    Plot the marginal Bayes rule E[alpha | ybar] as a function of ybar.

    S is marginalized out exactly over the discrete theta support (see
    stats_methods.marginal_bayes_rule): no S grid, no S_range, no truncation.
    The caller supplies that marginalized rule directly.

    Parameters
    ----------
    marginal_estimate : callable
        marginal_estimate(ybar) -> E[alpha | ybar], vectorized over an array of
        ybar values, e.g. ``lambda yb: oracle.marginal_bayes_rule(yb, m=11)``.
    ybar_range : (low, high) tuple
        Extent of the ybar axis to plot over.
    """
    fig, ax = _ensure_ax(ax)

    ybar_axis = np.linspace(ybar_range[0], ybar_range[1], n_grid)
    marginal = np.asarray(marginal_estimate(ybar_axis))

    ax.plot(ybar_axis, marginal)
    ax.set_xlabel("ybar")
    ax.set_ylabel("E[alpha | ybar]")
    return fig, ax


def plot_compare_oracle_npmle_marginal_ybar_bayes_rule(
        simulation, ybar, S, m, ax=None, npmle_estimator=None):
    """
    Overlay the marginal E[alpha | ybar] for the oracle, the NPMLE, and the
    naive rule.

    Pass a pre-fit `npmle_estimator` (NPMLE_BayesRule) to avoid re-solving the
    NPMLE optimization here; if None, one is fit from (ybar, S, m).
    """
    fig, ax = _ensure_ax(ax)

    oracle_estimator = OracleBayesRule(simulation.dgp, simulation.m)

    plot_bayes_rule_marginal_on_ybar(marginal_estimate=lambda yb: oracle_estimator.marginal_bayes_rule(yb, simulation.m),
                                     ybar_range=(min(ybar), max(ybar)),
                                     ax=ax,
                                    )

    if npmle_estimator is None:
        npmle_estimator = NPMLE_BayesRule(ybar, S=S, m=m)

    plot_bayes_rule_marginal_on_ybar(marginal_estimate=lambda yb: npmle_estimator.marginal_bayes_rule(yb, simulation.m),
                                     ybar_range=(min(ybar), max(ybar)),
                                     ax=ax,
                                    )

    naive_estimator = NaiveEstimator()
    
    plot_bayes_rule_marginal_on_ybar(marginal_estimate=lambda x:x,
                                        ybar_range=(min(ybar), max(ybar)),
                                        ax=ax,
                                    )

    return fig, ax

def plot_support_points_ybar_and_S(means, variances, weights, color, label,
                                   m, *, ax=None, max_weight=None, filled=True):
    """
    Scatter the support of a mixing distribution onto the (ybar, S) plane,
    using ONE fixed color per series and marker AREA proportional to each
    atom's weight, to overlay on the Bayes-rule contour.

    Encoding rationale: two support sets (e.g. NPMLE vs true) are overlaid, so
    color is used for SERIES IDENTITY (one color each) and SIZE for weight.
    Sizes use a shared absolute scale (via `max_weight`) so equal marker areas
    mean equal probability mass across both series -- directly comparable,
    unlike per-series colormaps.

    Coordinate mapping: the support lives in (alpha, theta) but the axes are
    (ybar, S). Each atom is placed via the mean of its sufficient statistics:
      * alpha -> ybar axis directly (ybar is an unbiased estimate of alpha),
      * theta -> S axis at E[S | theta] = (m - 1) * theta, since
        S | theta ~ Gamma((m-1)/2, 2 theta) has mean (m - 1) * theta.

    Parameters
    ----------
    means, variances, weights : array-like, shape (K,)
        Joint support atoms (alpha, theta) and their probability mass.
    color : matplotlib color
        A single color identifying this series (e.g. "tab:orange").
    label : str
        Legend label for this support set.
    m : int
        Observations per unit, for the theta -> E[S|theta] = (m-1) theta mapping.
    ax : matplotlib Axes, optional
        Axes to draw on (the contour's axes, so the support overlays it).
    max_weight : float, optional
        Weight that maps to the largest marker. Pass the SAME value for every
        series on a shared axes so sizes are comparable; defaults to this
        series' own max (only correct when one series is drawn).
    """
    fig, ax = _ensure_ax(ax)

    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Map atoms into (ybar, S) coordinates (see docstring).
    x = means
    y = (m - 1) * variances

    if max_weight is None:
        max_weight = weights.max()

    # Marker AREA proportional to weight on a shared scale; a small floor keeps
    # tiny atoms visible.
    MAX_AREA = 400.0
    sizes = 15.0 + MAX_AREA * (weights / max_weight)

    if filled:
        # Solid markers (e.g. the estimate). Drawn above the hollow series.
        ax.scatter(
            x, y,
            s=sizes, color=color,
            edgecolors="black", linewidths=0.5,
            alpha=0.85, label=label, zorder=4,
        )
    else:
        # Hollow outline markers (e.g. the truth) so a filled series underneath
        # stays visible where atoms coincide -- avoids occlusion.
        ax.scatter(
            x, y,
            s=sizes, facecolors="none", edgecolors=color,
            linewidths=1.6, label=label, zorder=3,
        )
    ax.legend(loc="best", fontsize="small")
    return fig, ax

def plot_bayes_rule_contour_ybar_and_S(estimate,
                                       ybar_range,
                                       S_range,
                                       *,
                                       ax=None,
                                       n_grid=200):
    """
    Filled contour of the full Bayes rule surface E[alpha | ybar, S] over the
    (ybar, S) grid. `estimate` is a callable estimate(ybar, S) -> E[alpha|ybar,S].
    """
    fig, ax = _ensure_ax(ax)

    ybar_axis, S_axis, surface = _evaluate_bayes_rule_grid(
        estimate, ybar_range, S_range, n_grid
    )

    # contourf expects Z indexed [row=y, col=x]; put S on the y-axis, ybar on x.
    cs = ax.contourf(ybar_axis, S_axis, surface.T, levels=20)
    fig.colorbar(cs, ax=ax, label="E[alpha | ybar, S]")
    ax.set_xlabel("ybar")
    ax.set_ylabel("S")
    return fig, ax

def plot_contours_and_mixing_support(npmle_estimator: NPMLE_BayesRule,
                                     simulation: Simulation,
                                     ybar,
                                     S,
                                     m,
                                     ax=None):
    """
    Display bayes rule contours and the support of the mean/variance joint distribution.
    """
    
    # The NPMLE mixture is pruned to its nontrivial atoms (a handful of support
    # points), so bayes_rule's (n_grid^2, support) intermediate is small and the
    # contour can use a fine grid.
    fig, ax = plot_bayes_rule_contour_ybar_and_S(
        lambda yb, _S: npmle_estimator.bayes_rule(yb, _S, m=m),
        ybar_range=(min(ybar), max(ybar)),
        S_range=(min(S), max(S)),
        ax=ax,
        n_grid=50,
    )

    npmle_mvw = npmle_estimator.npmle_mix.mixing_distribution.mean_and_variance_joint_approx_pmf()
    true_mvw = simulation.mixing_distribution.mean_and_variance_joint_approx_pmf()

    # Shared size scale across both series so equal marker areas mean equal
    # probability mass. Fixed, distinct colors identify the series; marker size
    # encodes weight.
    shared_max_w = max(np.max(npmle_mvw[2]), np.max(true_mvw[2]))

    plot_support_points_ybar_and_S(
        *npmle_mvw, color="tab:orange",
        label=f"NPMLE ({'alpha indep. theta' if npmle_estimator.assume_independent_mv else 'joint alpha, theta'})",
        m=m, ax=ax, max_weight=shared_max_w,
    )

    plot_support_points_ybar_and_S(
        *true_mvw, color="black",
        label="True mixing distribution",
        m=m, ax=ax, max_weight=shared_max_w, filled=False,
    )

    return fig, ax


# --- composite figures (figure_*) ------------------------------------------

def figure_paper(simulation:Simulation, ybar, S, m):
    """
    Reproduce the paper-style multi-panel figure:
      panel 0: mean pmf (stems) with a rug of the observed ybar beneath,
      panel 1: histogram of ybar.
      panel 2: bayes rule with a rug of observed ybar beneath

    Takes the data (ybar) as an argument rather than generating it, so the
    caller controls the RNG / whether statistics are regenerated.
    """
    fig, axs = plt.subplots(3, sharex=True)

    # Fit the NPMLE ONCE and reuse it across every panel. The NPMLE solve is
    # the expensive part (a convex optimization taking seconds); it depends
    # only on the training data (ybar, S, m), so re-fitting per panel -- as the
    # old code did (one bare solve for panel 0, one inside the panel-2 helper,
    # and one inside the panel-3 contour lambda) -- repeated the identical solve
    # 3x and was the cause of the slow figure.
    npmle_estimator = NPMLE_BayesRule(ybar, S=S, m=m)

    plot_mean_pmf(simulation.mixing_distribution, ax=axs[0], pmf_color="blue")
    plot_mean_pmf(npmle_estimator.npmle_mix.mixing_distribution, ax=axs[0], pmf_color="orange")
    plot_data_rug(ybar, ax=axs[0])

    plot_ybar_hist(ybar, ax=axs[1])

    plot_compare_oracle_npmle_marginal_ybar_bayes_rule(
        simulation=simulation, ybar=ybar, S=S, m=m, ax=axs[2],
        npmle_estimator=npmle_estimator,
    )
    plot_data_rug(ybar, ax=axs[2], ymin=-1.3, ymax=-1)


    fig.tight_layout()
    return fig


# Additional figure
def figure_contours(simulation:Simulation, seed):
    fig, axs = plt.subplots(2, sharex=True)

    ybar, S = simulation.generate_statistics(seed)
    m = simulation.m

    npmle_estimator = NPMLE_BayesRule(ybar, S=S, m=m, assume_independent_mv=True)

    plot_contours_and_mixing_support(npmle_estimator=npmle_estimator,
                                     simulation=simulation,
                                     ybar=ybar,
                                     S=S,
                                     m=m,
                                     ax=axs[0])

    
    npmle_estimator = NPMLE_BayesRule(ybar, S=S, m=m, assume_independent_mv=False)

    plot_contours_and_mixing_support(npmle_estimator=npmle_estimator,
                                     simulation=simulation,
                                     ybar=ybar,
                                     S=S,
                                     m=m,
                                     ax=axs[1])

    fig.tight_layout()
    return fig


def make_papers_figures(simulation, seed=0):
    """
    Draw one trial from `simulation`, build the paper
    figure, save it, and return (fig, path).
    """
    ybar, S = simulation.generate_statistics(seed)
    fig = figure_paper(simulation, ybar, S, simulation.m)
    path = save_figure(fig, "paper_figures", f"{simulation.id}_seed{seed}")
    return fig, path

def make_contour_figure(simulation, seed=0):
    fig = figure_contours(simulation, seed)
    path = save_figure(fig, "contour_figure", f"{simulation.id}_seed{seed}")
    return fig, path



if __name__ == "__main__":

    SEED = 0

    figure4_simulation = IndependentMVSimulation(
        n=400,
        m=11,
        mean_support_pts=np.array([-0.5, 1.0, 3.0]),
        variance_support_pts=np.array([0.5, 2.0, 4.0]),
        mean_pmf=np.full(3, 1 / 3),
        variance_pmf=np.full(3, 1 / 3),
    )
    _, out_path = make_papers_figures(figure4_simulation, seed=SEED)
    print(f"wrote {out_path}")

    # --- Correlated (alpha, theta) DGP for the contour / joint-NPMLE figure ---
    # Correlation is encoded by a joint pmf that does NOT factor as an outer
    # product. Here mass sits on the diagonal of the (alpha, theta) grid, so
    # small alpha pairs with small theta and large alpha with large theta
    # (positive dependence) -- something only the JOINT NPMLE can recover; the
    # independent fit forces Cov(alpha, theta) = 0. We use the base Simulation
    # directly (IndependentMVSimulation only accepts factorized marginals).
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

    correlated_simulation = Simulation(
        n=400,
        m=11,
        mean_support_pts=mean_support,
        variance_support_pts=variance_support,
        joint_pmf=correlated_joint_pmf,
    )
    _, out_path = make_contour_figure(correlated_simulation, seed=SEED)
    print(f"wrote {out_path}")
