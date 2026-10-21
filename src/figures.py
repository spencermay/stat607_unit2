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

from simulation import IndependentMVSimulation


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

    plot_ybar_hist(ybar, ax=axs[1])

    

    fig.tight_layout()
    return fig


def make_papers_figures(simulation, seed=0):
    """
    Convenience runner: draw one trial from `simulation`, build the paper
    figure, save it, and return (fig, path). Equivalent to the previous
    Simulation.make_papers_figures, now as a free function.
    """
    rng = np.random.default_rng(seed)
    ybar, S = simulation.generate_statistics(rng)
    fig = figure_paper(simulation, ybar, S=S)
    path = save_figure(fig, "paper_figures", simulation.id)
    return fig, path


if __name__ == "__main__":
    figure4_simulation = IndependentMVSimulation(
        n=400, m=11,
        mean_support_pts=np.array([-0.5, 1.0, 3.0]),
        variance_support_pts=np.array([0.5, 2.0, 4.0]),
        mean_pmf=np.full(3, 1 / 3),
        variance_pmf=np.full(3, 1 / 3),
    )
    _, out_path = make_papers_figures(figure4_simulation)
    print(f"wrote {out_path}")
