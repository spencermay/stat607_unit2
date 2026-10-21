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
from scipy.stats import gamma

from simulation import IndependentMVSimulation
from stats_methods import _gaussian_pdf


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

def plot_true_mean_pmf(mixing_distribution, *, ax=None):
    """
    Stem plot of the true marginal pmf of alpha (the mean mixing distribution),
    collapsing the joint support onto its distinct mean values.
    """
    fig, ax = _ensure_ax(ax)

    means = np.asarray(mixing_distribution.means)
    weights = np.asarray(mixing_distribution.weights)

    unique_means = np.unique(means)
    # Total mass on each distinct mean (sum weights of joint points sharing it).
    mass = np.array([weights[np.isclose(means, mu)].sum() for mu in unique_means])

    ax.vlines(unique_means, ymin=0, ymax=mass)
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


def make_conditional_S_density(gaussian_mixture, m):
    """
    Build the conditional density f(S | ybar) under a GaussianMixture DGP, for
    use as the marginalization weight in plot_bayes_rule_marginal_on_ybar.

    Since ybar _||_ S | (alpha, theta) and S | theta ~ theta * chi^2_{m-1}:

        f(S | ybar) = sum_k p(theta_k | ybar) * Gamma(S; (m-1)/2, 2 theta_k)

    where p(theta_k | ybar) ∝ f(ybar | theta_k) * p(theta_k) is the variance
    posterior given ybar alone (f(ybar | theta_k) marginalizes alpha over the
    conditional prior of alpha given theta_k).

    Returns
    -------
    callable weight(ybar, S) -> f(S | ybar), vectorized over broadcastable
    ``ybar`` and ``S`` (the component axis is summed internally).
    """
    means, variances, weights = (
        gaussian_mixture.mixing_distribution.mean_and_variance_joint_approx_pmf()
    )
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    weights = np.asarray(weights, dtype=float)

    theta_values, theta_index = np.unique(variances, return_inverse=True)
    ybar_variances = variances / m          # Var(ybar) per joint support point
    gamma_shape = (m - 1) / 2.0
    gamma_scales = 2.0 * theta_values       # S | theta ~ Gamma(shape, 2 theta)
    n_theta = len(theta_values)

    def weight(ybar, S):
        ybar = np.asarray(ybar, dtype=float)
        S = np.asarray(S, dtype=float)

        # p(theta_k | ybar) ∝ sum_j w_jk phi(ybar; alpha_j, theta_j/m), grouped
        # by distinct theta. Accumulate f(ybar | theta_k) per distinct theta.
        comp_pdf = _gaussian_pdf(ybar[..., None], means, ybar_variances)  # (...,K)
        w_phi = weights * comp_pdf                                        # (...,K)

        f_ybar_given_theta = np.zeros(ybar.shape + (n_theta,))
        np.add.at(
            np.moveaxis(f_ybar_given_theta, -1, 0),
            theta_index,
            np.moveaxis(w_phi, -1, 0),
        )
        post_theta = f_ybar_given_theta / np.sum(
            f_ybar_given_theta, axis=-1, keepdims=True
        )  # (..., n_theta), p(theta_k | ybar)

        # f(S | ybar) = sum_k p(theta_k | ybar) Gamma(S; shape, 2 theta_k).
        gamma_S = gamma.pdf(
            S[..., None], a=gamma_shape, scale=gamma_scales
        )  # (..., n_theta)
        return np.sum(post_theta * gamma_S, axis=-1)

    return weight


def plot_bayes_rule_marginal_on_ybar(estimate,
                                     ybar_range,
                                     S_range,
                                     conditional_S_density,
                                     *,
                                     ax=None,
                                     n_grid=200):
    """
    Plot the model marginal Bayes rule E[alpha | ybar] as a function of ybar.

    Parameters
    ----------
    estimate : callable
        estimate(ybar, S) -> E[alpha | ybar, S]; see _evaluate_bayes_rule_grid.
    ybar_range, S_range : (low, high) tuples
    conditional_S_density : callable
        weight(ybar, S) -> f(S | ybar), the S-marginalization weight. Use
        make_conditional_S_density(gaussian_mixture, m) to build it from a DGP.
    """
    fig, ax = _ensure_ax(ax)

    ybar_axis, S_axis, surface = _evaluate_bayes_rule_grid(
        estimate, ybar_range, S_range, n_grid
    )

    # Weights f(S_j | ybar_i): shape (n_grid, n_grid), row = ybar, col = S.
    yb_mesh = ybar_axis[:, None]           # (n_grid, 1)
    S_mesh = S_axis[None, :]               # (1, n_grid)
    w = conditional_S_density(yb_mesh, S_mesh)  # (n_grid, n_grid)

    # Weighted average over S (axis 1), renormalized per ybar row so the
    # truncation to S_range doesn't change the scale.
    marginal = np.sum(surface * w, axis=1) / np.sum(w, axis=1)

    ax.plot(ybar_axis, marginal)
    ax.set_xlabel("ybar")
    ax.set_ylabel("E[alpha | ybar]")
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

def figure_paper(simulation, ybar, *, S=None):
    """
    Reproduce the paper-style multi-panel figure:
      panel 0: true mean pmf (stems) with a rug of the observed ybar beneath,
      panel 1: histogram of ybar.

    Takes the data (ybar) as an argument rather than generating it, so the
    caller controls the RNG / whether statistics are regenerated.
    """
    fig, axs = plt.subplots(3)

    plot_true_mean_pmf(simulation.mixing_distribution, ax=axs[0])
    plot_data_rug(ybar, ax=axs[0])
    plot_data_rug(ybar, ax=axs[2])

    plot_ybar_hist(ybar, ax=axs[1])



    fig.tight_layout()
    return fig


def make_papers_figures(simulation, seed=0):
    """
    Draw one trial from `simulation`, build the paper
    figure, save it, and return (fig, path).
    """
    rng = np.random.default_rng(seed)
    ybar, S = simulation.generate_statistics(rng)
    fig = figure_paper(simulation, ybar, S=S)
    path = save_figure(fig, "paper_figures", simulation.id)
    return fig, path


if __name__ == "__main__":
    figure4_simulation = IndependentMVSimulation(
        n=400,
        m=11,
        mean_support_pts=np.array([-0.5, 1.0, 3.0]),
        variance_support_pts=np.array([0.5, 2.0, 4.0]),
        mean_pmf=np.full(3, 1 / 3),
        variance_pmf=np.full(3, 1 / 3),
    )
    _, out_path = make_papers_figures(figure4_simulation)
    print(f"wrote {out_path}")
