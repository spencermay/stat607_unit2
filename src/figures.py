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


def plot_compare_oracle_npmle_marginal_ybar_bayes_rule(simulation, ybar, S, m, ax=None):
    fig, ax = _ensure_ax(ax)

    oracle_estimator = OracleBayesRule(simulation.dgp, simulation.m)

    plot_bayes_rule_marginal_on_ybar(marginal_estimate=lambda yb: oracle_estimator.marginal_bayes_rule(yb, simulation.m),
                                     ybar_range=(min(ybar), max(ybar)),
                                     ax=ax,
                                    )

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
    fig, axs = plt.subplots(3)

    plot_mean_pmf(simulation.mixing_distribution, ax=axs[0], pmf_color="blue")
    plot_mean_pmf(get_mixing_distribution_npmle(ybar=ybar, S=S, m=m).mixing_distribution,ax=axs[0], pmf_color="orange")
    plot_data_rug(ybar, ax=axs[0])

    plot_ybar_hist(ybar, ax=axs[1])

    plot_compare_oracle_npmle_marginal_ybar_bayes_rule(simulation=simulation, ybar=ybar, S=S, m=m, ax=axs[2])
    plot_data_rug(ybar, ax=axs[2], ymin=-1.3, ymax=-1)

    fig.tight_layout()
    return fig


def make_papers_figures(simulation, seed=0):
    """
    Draw one trial from `simulation`, build the paper
    figure, save it, and return (fig, path).
    """
    ybar, S = simulation.generate_statistics(seed)
    fig = figure_paper(simulation, ybar, S=S, m=simulation.m)
    # Encode both the setup id and the seed so figures from different draws of
    # the same setup don't overwrite each other.
    path = save_figure(fig, "paper_figures", f"{simulation.id}_seed{seed}")
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
