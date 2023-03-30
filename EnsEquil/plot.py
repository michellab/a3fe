"""Plotting functions"""

import matplotlib.pyplot as _plt
from math import ceil as _ceil
import numpy as _np
import scipy.stats as _stats
import seaborn as _sns
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

from .process_grads import GradientData
from ._utils import read_overlap_mat as _read_overlap_mat

def general_plot(x_vals: _np.ndarray, y_vals: _np.ndarray, x_label: str, y_label: str,
                 outfile: str, vline_val: _Optional[float] = None,
                 hline_val: _Optional[float] = None) -> None:
    """ 
    Plot several sets of y_vals against one set of x vals, and show confidence
    intervals based on inter-y-set deviations (assuming normality).

    Parameters
    ----------
    x_vals : np.ndarray
        1D array of x values.
    y_vals : np.ndarray
        1 or 2D array of y values, with shape (n_sets, n_vals). Assumes that
        the sets of data are passed in the same order as the runs.
    x_label : str
        Label for the x axis.
    y_label : str
        Label for the y axis.
    outfile : str
        Name of the output file.
    vline_val : float, Optional
        x value to draw a vertical line at, for example the time taken for
        equilibration.
    hline_val : float, Optional
        y value to draw a horizontal line at.
    """
    y_avg = _np.mean(y_vals, axis=0)
    conf_int = _stats.t.interval(0.95, len(y_vals[:, 0])-1, loc=y_avg, scale=_stats.sem(y_vals, axis=0))  # 95 % C.I.

    fig, ax = _plt.subplots(figsize=(8, 6))
    ax.plot(x_vals, y_avg, label="Mean", linewidth=2)
    for i, entry in enumerate(y_vals):
        ax.plot(x_vals, entry, alpha=0.5, label=f"run {i+1}")
    if vline_val is not None:
        ax.axvline(x=vline_val, color='red', linestyle='dashed')
    if hline_val is not None:
        ax.axhline(y=hline_val, color='black', linestyle='dashed')
    # Add confidence intervals
    ax.fill_between(x_vals, conf_int[0], conf_int[1], alpha=0.5, facecolor='#ffa500')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    fig.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    # Close the figure to avoid memory leaks
    _plt.close(fig)


def plot_gradient_stats(gradients_data: GradientData, output_dir: str, plot_type: str) -> None:
    """ 
    Plot the variance of the gradients for a list of lambda windows.
    If equilibrated is True, only data after equilibration is used.

    Parameters
    ----------
    gradients_data : GradientData
        GradientData object containing the gradient data.
    output_dir : str
        Directory to save the plot to.
    plot_type : str
        Type of plot to make. Can be "mean", "variance", "sem", "stat_ineff", or "integrated_sem".

    Returns
    -------
    None
    """
    # Check plot_type is valid
    plot_type = plot_type.lower()
    plot_types = ["mean", "intra_run_variance", "sem", "stat_ineff", "integrated_sem"]
    if not plot_type in plot_types:
        raise ValueError(f"'plot_type' must be one of {plot_types}, not {plot_type}")
    
    # Make plots of variance of gradients
    fig, ax = _plt.subplots(figsize=(8, 6))

    if plot_type == "mean":
        ax.bar(gradients_data.lam_vals,
               gradients_data.means,
               width=0.02, edgecolor='black',
               yerr=gradients_data.sems_overall)
        ax.set_ylabel(r"$\langle \frac{\mathrm{d}h}{\mathrm{d}\lambda}\rangle _{\lambda} $ / kcal mol$^{-1}$"),

    elif plot_type == "intra_run_variance":
        ax.bar(gradients_data.lam_vals,
               gradients_data.vars_intra,
               width=0.02, edgecolor='black' )
        ax.set_ylabel(r"Mean Intra-Run Var($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $) / kcal$^{2}$ mol$^{-2}$"),

    elif plot_type == "sem":
        ax.bar(gradients_data.lam_vals,
               gradients_data.sems_intra,
               width=0.02, edgecolor='black', label="Intra-Run")
        ax.bar(gradients_data.lam_vals,
               gradients_data.sems_inter,
               bottom=gradients_data.sems_intra,
               width=0.02, edgecolor='black', label="Inter-Run")
        ax.set_ylabel(r"SEM($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $) / kcal mol$^{-1}$"),
        ax.legend()

    elif plot_type == "stat_ineff":
        ax.bar(gradients_data.lam_vals,
               gradients_data.stat_ineffs,
               width=0.02, edgecolor='black')
        ax.set_ylabel(r"Statistical Inefficiency / ns")

    elif plot_type == "integrated_sem":
        handle1, *_ = ax.bar(gradients_data.lam_vals,
               gradients_data.sems_overall,
               label = "SEMs",
               width=0.02, edgecolor='black')
        ax.set_ylabel(r"SEM($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $) / kcal mol$^{-1}$"),
        ax.legend()
        # Get second y axis so we can plot on different scales
        ax2= ax.twinx()
        handle2, = ax2.plot(gradients_data.lam_vals,
               gradients_data.integrated_sems,
               label = "Integrated SEM", color='red', linewidth=2)
        # Add vertical lines to show optimal lambda windows
        delta_sem = 0.1
        integrated_sems = gradients_data.integrated_sems
        total_sem = integrated_sems[-1]
        sem_vals = _np.linspace(0, total_sem, int(total_sem/delta_sem) + 1)
        optimal_lam_vals = gradients_data.calculate_optimal_lam_vals(delta_sem = 0.1)
        # Add horizontal lines at sem vals
        for sem_val in sem_vals:
            ax2.axhline(y=sem_val, color='black', linestyle='dashed', linewidth=0.5)
        # Add vertical lines at optimal lambda vals
        for lam_val in optimal_lam_vals:
            ax2.axvline(x=lam_val, color='black', linestyle='dashed', linewidth=0.5)
        ax2.set_ylabel(r"Integrated SEM($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $) / kcal mol$^{-1}$"),
        ax2.legend()

    ax.set_xlabel(r"$\lambda$")
    
    name = f"{output_dir}/gradient_{plot_type}"
    if gradients_data.equilibrated:
        name += "_equilibrated"
    fig.savefig(name, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    _plt.close(fig)


def plot_gradient_hists(gradients_data: GradientData, output_dir: str) -> None:
    """ 
    Plot histograms of the gradients for a list of lambda windows.
    If equilibrated is True, only data after equilibration is used.

    Parameters
    ----------
    gradients_data : GradientData
        GradientData object containing the gradient data.
    output_dir : str
        Directory to save the plot to.
    equilibrated : bool
        If True, only equilibrated data is used.

    Returns
    -------
    None
    """
    # Plot mixed gradients for each window
    n_lams = len(gradients_data.lam_vals)
    fig, axs = _plt.subplots(nrows=_ceil(n_lams/8), ncols=8, figsize=(40, 5*(n_lams/8)))
    for i, ax in enumerate(axs.flatten()):
        if i < n_lams:
            # One histogram for each simulation
            for j, gradients in enumerate(gradients_data.gradients[i]):
                ax.hist(gradients, bins=50, density=True, alpha=0.5, label=f"Run {j+1}")
            ax.legend()
            ax.set_title(f"$\lambda$ = {gradients_data.lam_vals[i]}")
            ax.set_xlabel(r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$")
            ax.set_ylabel("Probability density")
            ax.text(0.05, 0.95, f"Std. dev. = {_np.std(gradients_data.gradients[i]):.2f}" + r" kcal mol$^{-1}$", transform=ax.transAxes)
            ax.text(0.05, 0.9, f"Mean = {_np.mean(gradients_data.gradients[i]):.2f}" + r" kcal mol$^{-1}$", transform=ax.transAxes)
    
    fig.tight_layout()
    name = f"{output_dir}/gradient_hists"
    if gradients_data.equilibrated:
        name += "_equilibrated"
    fig.savefig(name, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    _plt.close(fig)

def plot_equilibration_time(lam_windows: _List["LamWindows"], output_dir:str)->None:
    """
    Plot the equilibration time for each lambda window.

    Parameters
    ----------
    lam_windows : List[LamWindows]
        List of LamWindows objects.
    output_dir : str
        Directory to save the plot to.

    Returns
    -------
    None
    """
    fig, ax=_plt.subplots(figsize=(8, 6))
    # Plot the total time simulated per simulation, so we can see how efficient
    # the protocol is
    ax.bar([win.lam for win in lam_windows],
            [win.sims[0].tot_simtime for win in lam_windows],  # All sims at given lam run for same time
            width=0.02, edgecolor='black', label="Total time simulated per simulation")
    # Now plot the equilibration time
    ax.bar([win.lam for win in lam_windows],
            [win.equil_time for win in lam_windows],
            width=0.02, edgecolor='black', label="Equilibration time per simulation")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Time (ns)")
    fig.legend()
    fig.savefig(f"{output_dir}/equil_times", dpi=300,
                bbox_inches='tight', facecolor='white', transparent=False)

def plot_overlap_mat(ax: _plt.Axes, mbar_file: str, name: str) -> None:
    """
    Plot the overlap matrix for a given MBAR file on the supplied axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis on which to plot.
    mbar_file : str
        Path to MBAR file.
    name : str
        Name of the plot.

    Returns
    -------
    None
    """
    overlap_mat = _read_overlap_mat(mbar_file)
    _sns.heatmap(overlap_mat, ax=ax, square=True).figure
    ax.set_title(name)

def plot_overlap_mats(mbar_outfiles: _List[str], output_dir:str) -> None:
    """
    Plot the overlap matrices for all mbar outfiles supplied.
    
    Parameters
    ----------
    mbar_outfiles : List[str]
        List of MBAR outfiles. It is assumed that these are passed in the same
        order as the runs they correspond to.
    output_dir : str
        The directory to save the plot to.

    Returns
    -------
    None
    """
    n_runs = len(mbar_outfiles)
    fig, axs = _plt.subplots(1, n_runs, figsize=(4*n_runs, 4),dpi=1000)

    for i in range(n_runs):
        plot_overlap_mat(axs[i], mbar_outfiles[i], f"Run {i+1}")
        
    fig.tight_layout()
    fig.savefig(f"{output_dir}/overlap.png")
