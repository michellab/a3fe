"""Plotting functions"""

__all__ = [
    "general_plot",
    "p_plot",
    "plot_gradient_stats",
    "plot_gradient_hists",
    "plot_gradient_timeseries",
    "plot_equilibration_time",
    "plot_overlap_mat",
    "plot_mbar_pmf",
    "plot_against_exp",
    "plot_gelman_rubin_rhat",
    "plot_comparitive_convergence",
    "plot_comparitive_convergence_sem",
    "plot_normality",
    "plot_av_waters",
]

import glob as _glob
import os as _os
from math import ceil as _ceil
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import matplotlib.pyplot as _plt
import matplotlib.colors as _colors
import matplotlib.cm as _cm
import numpy as _np
import pandas as _pd
import scipy.stats as _stats
from scipy.stats import kruskal as _kruskal
import seaborn as _sns

from ..read._process_somd_files import read_mbar_pmf as _read_mbar_pmf
from ..read._process_somd_files import read_overlap_mat as _read_overlap_mat
from .process_grads import GradientData
from .rmsd import get_rmsd as _get_rmsd
from .compare import (
    get_comparitive_convergence_data as _get_comparitive_convergence_data,
)
from .waters import get_av_waters_stage as _get_av_waters_stage
from ..run._utils import SimulationRunnerIterator as _SimulationRunnerIterator


def general_plot(
    x_vals: _np.ndarray,
    y_vals: _np.ndarray,
    x_label: str,
    y_label: str,
    outfile: str,
    vline_val: _Optional[float] = None,
    hline_val: _Optional[float] = None,
    run_nos: _Optional[_List[int]] = None,
) -> None:
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
    run_nos : List[int], Optional
        List of of the numbers of the runs supplied. If None, the runs are
        numbered in the order supplied from 1.
    """
    # If the y values are 1D, add another axis so that the normal logic works
    y_vals_1d = True if len(y_vals.shape) == 1 else False
    if y_vals_1d:
        y_vals = y_vals[_np.newaxis, :]

    # Compute the mean and 95% confidence intervals
    y_avg = _np.mean(y_vals, axis=0)
    conf_int = _stats.t.interval(
        0.95, len(y_vals[:, 0]) - 1, loc=y_avg, scale=_stats.sem(y_vals, axis=0)
    )

    fig, ax = _plt.subplots(figsize=(8, 6))
    ax.plot(x_vals, y_avg, label="Mean", linewidth=2)
    for i, entry in enumerate(y_vals):
        ax.plot(x_vals, entry, alpha=0.5, label=f"run {run_nos[i] if run_nos else i+1}")
    if vline_val is not None:
        ax.axvline(x=vline_val, color="red", linestyle="dashed")
    if hline_val is not None:
        ax.axhline(y=hline_val, color="black", linestyle="dashed")
    # Add confidence intervals
    ax.fill_between(x_vals, conf_int[0], conf_int[1], alpha=0.5, facecolor="#ffa500")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # No point in adding a legend if there is only one set of data
    if not y_vals_1d:
        ax.legend()

    fig.savefig(
        outfile, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    # Close the figure to avoid memory leaks
    _plt.close(fig)


def p_plot(
    times: _np.ndarray,
    p_vals: _np.ndarray,
    outfile: str,
    p_cutoff: float = 0.4,
) -> None:
    """
    Plot the p value against time discarded from the start of the simulation.

    Parameters
    ----------
    times : np.ndarray
        1D array of times discarded from the start of the simulation. This is per
        run.
    p_vals : np.ndarray
        1D array of p values.
    outfile : str
        Name of the output file.
    p_cutoff : float, optional
        p value cutoff for significance. Default is 0.4. A horizontal line is
        drawn at this value.

    Returns
    -------
    None
    """
    fig, ax = _plt.subplots(figsize=(8, 6))
    ax.scatter(times, p_vals, s=10)
    ax.axhline(y=p_cutoff, color="red", linestyle="dashed")
    ax.set_xlabel("Time discarded from start of simulation per run / ns")
    ax.set_ylabel("p value")
    fig.savefig(
        outfile, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    # Close the figure to avoid memory leaks
    _plt.close(fig)


def plot_gradient_stats(
    gradients_data: GradientData, output_dir: str, plot_type: str
) -> None:
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
        Type of plot to make. Can be "mean", "variance", "sem", "stat_ineff", "integrated_sem",
        "sq_sem_sim_time" or "integrated_var".

    Returns
    -------
    None
    """
    # Check plot_type is valid
    plot_type = plot_type.lower()
    plot_types = [
        "mean",
        "intra_run_variance",
        "sem",
        "stat_ineff",
        "integrated_sem",
        "sq_sem_sim_time",
        "integrated_var",
        "pred_best_simtime",
    ]
    if not plot_type in plot_types:
        raise ValueError(f"'plot_type' must be one of {plot_types}, not {plot_type}")

    # Make plots of variance of gradients
    fig, ax = _plt.subplots(figsize=(8, 6))

    if plot_type == "mean":
        ax.bar(
            gradients_data.lam_vals,
            gradients_data.means,
            width=0.02,
            edgecolor="black",
            yerr=gradients_data.sems_overall,
        )
        ax.set_ylabel(
            r"$\langle \frac{\mathrm{d}h}{\mathrm{d}\lambda}\rangle _{\lambda} $ / kcal mol$^{-1}$"
        ),

    elif plot_type == "intra_run_variance":
        ax.bar(
            gradients_data.lam_vals,
            gradients_data.vars_intra,
            width=0.02,
            edgecolor="black",
        )
        ax.set_ylabel(
            r"Mean Intra-Run Var($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $) / kcal$^{2}$ mol$^{-2}$"
        ),

    elif plot_type == "sem":
        ax.bar(
            gradients_data.lam_vals,
            gradients_data.sems_intra,
            width=0.02,
            edgecolor="black",
            label="Intra-Run",
        )
        ax.bar(
            gradients_data.lam_vals,
            gradients_data.sems_inter,
            bottom=gradients_data.sems_intra,
            width=0.02,
            edgecolor="black",
            label="Inter-Run",
        )
        ax.set_ylabel(
            r"SEM($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $) / kcal mol$^{-1}$"
        ),
        ax.legend()

    elif plot_type == "stat_ineff":
        ax.bar(
            gradients_data.lam_vals,
            gradients_data.stat_ineffs,
            width=0.02,
            edgecolor="black",
        )
        ax.set_ylabel(r"Statistical Inefficiency / ns")

    elif plot_type == "integrated_sem":
        handle1, *_ = ax.bar(
            gradients_data.lam_vals,
            gradients_data.get_time_normalised_sems(origin="inter", smoothen=True),
            label="SEMs",
            width=0.02,
            edgecolor="black",
        )
        ax.set_ylabel(
            r"SEM($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $) / kcal mol$^{-1}$ ns$^{1/2}$"
        ),
        ax.legend()
        # Get second y axis so we can plot on different scales
        ax2 = ax.twinx()
        (handle2,) = ax2.plot(
            gradients_data.lam_vals,
            gradients_data.get_integrated_error(
                er_type="sem", origin="inter", smoothen=True
            ),
            label="Integrated SEM",
            color="red",
            linewidth=2,
        )
        # Add vertical lines to show optimal lambda windows
        delta_sem = 0.1
        integrated_sems = gradients_data.get_integrated_error(
            er_type="sem", origin="inter", smoothen=True
        )
        total_sem = integrated_sems[-1]
        sem_vals = _np.linspace(0, total_sem, int(total_sem / delta_sem) + 1)
        optimal_lam_vals = gradients_data.calculate_optimal_lam_vals(
            er_type="sem",
            # delta_er=delta_sem ,
            n_lam_vals=30,
            sem_origin="inter",
            smoothen_sems=True,
        )
        # Add horizontal lines at sem vals
        for sem_val in sem_vals:
            ax2.axhline(y=sem_val, color="black", linestyle="dashed", linewidth=0.5)
        # Add vertical lines at optimal lambda vals
        for lam_val in optimal_lam_vals:
            ax2.axvline(x=lam_val, color="black", linestyle="dashed", linewidth=0.5)
        ax2.set_ylabel(
            r"Integrated Standardised SEM($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $) / kcal mol$^{-1}$ ns$^{1/2}$"
        ),

    elif plot_type == "sq_sem_sim_time":
        ax.bar(
            gradients_data.lam_vals,
            gradients_data.sems_inter_delta_g**2
            / (gradients_data.total_times * len(gradients_data.run_nos)),
            width=0.02,
            edgecolor="black",
        )
        ax.set_ylabel(
            r"(SEM($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $))$^2$ / Total Sim Time / kcal$^2$ mol$^{-2}$ ns$^{-1}$"
        ),
        ax.legend()

    elif plot_type == "pred_best_simtime":
        # Calculate the predicted optimum simulation time
        relative_cost = gradients_data.relative_simulation_cost
        runtime_constant = gradients_data.runtime_constant
        n_runs = len(gradients_data.run_nos)
        time_normalised_sems = gradients_data.get_time_normalised_sems(
            origin="inter_delta_g", smoothen=False
        )
        pred_opt_simtime = (
            time_normalised_sems * 1 / _np.sqrt(runtime_constant * relative_cost)
        )
        # Get this as a per-run quantity
        pred_opt_simtime = pred_opt_simtime / n_runs

        ax.bar(
            gradients_data.lam_vals,
            pred_opt_simtime,
            width=0.02,
            edgecolor="black",
        )
        ax.set_ylabel(r"Predicted most efficient runtimes per run /  ns"),
        ax.legend()

    elif plot_type == "integrated_var":
        handle1, *_ = ax.bar(
            gradients_data.lam_vals,
            _np.sqrt(gradients_data.vars_intra),
            label="Variances",
            width=0.02,
            edgecolor="black",
        )
        ax.set_ylabel(
            r"(Var($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $))$^{1/2}$ / kcal mol$^{-1}$"
        ),
        ax.legend()
        # Get second y axis so we can plot on different scales
        ax2 = ax.twinx()
        (handle2,) = ax2.plot(
            gradients_data.lam_vals,
            gradients_data.get_integrated_error(er_type="root_var"),
            label="Integrated Sqr(Var)",
            color="red",
            linewidth=2,
        )
        # Add vertical lines to show optimal lambda windows
        delta_root_var = 1  # kcal mol^-1
        integrated_root_var = gradients_data.get_integrated_error(er_type="root_var")
        total_root_var = integrated_root_var[-1]
        root_var_vals = _np.linspace(
            0, total_root_var, int(total_root_var / delta_root_var) + 1
        )
        optimal_lam_vals = gradients_data.calculate_optimal_lam_vals(
            er_type="root_var",
            # delta_er=delta_root_var)
            n_lam_vals=30,
        )
        # Add horizontal lines at sem vals
        for root_var_val in root_var_vals:
            ax2.axhline(
                y=root_var_val, color="black", linestyle="dashed", linewidth=0.5
            )
        # Add vertical lines at optimal lambda vals
        for lam_val in optimal_lam_vals:
            ax2.axvline(x=lam_val, color="black", linestyle="dashed", linewidth=0.5)
        ax2.set_ylabel(
            r"Integrated (Var($\frac{\mathrm{d}h}{\mathrm{d}\lambda} $))$^{1/2}$ / kcal mol$^{-1}$"
        ),
        ax2.legend()
        ax2.legend()

    ax.set_xlabel(r"$\lambda$")

    name = f"{output_dir}/gradient_{plot_type}"
    if gradients_data.equilibrated:
        name += "_equilibrated"
    fig.savefig(
        name, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    _plt.close(fig)


def plot_gradient_hists(
    gradients_data: GradientData, output_dir: str, run_nos: _Optional[_List[int]] = None
) -> None:
    """
    Plot histograms of the gradients for a list of lambda windows.
    If equilibrated is True, only data after equilibration is used.

    Parameters
    ----------
    gradients_data : GradientData
        GradientData object containing the gradient data.
    output_dir : str
        Directory to save the plot to.
    run_nos : List[int], Optional, default: None
        The run numbers to use. If None, all runs will be used.

    Returns
    -------
    None
    """
    # Plot mixed gradients for each window
    n_lams = len(gradients_data.lam_vals)
    ensemble_size = len(
        gradients_data.gradients[0]
    )  # Check the length of the gradients data for the first window
    fig, axs = _plt.subplots(
        nrows=_ceil(n_lams / 8), ncols=8, figsize=(40, 5 * (n_lams / 8))
    )
    for i, ax in enumerate(axs.flatten()):  # type: ignore
        if i < n_lams:
            # One histogram for each simulation
            for j, gradients in enumerate(gradients_data.gradients[i]):
                ax.hist(
                    gradients,
                    bins=50,
                    density=True,
                    alpha=0.5,
                    label=f"Run {run_nos[j] if run_nos else j+1}",
                )
            ax.legend()
            ax.set_title(f"$\lambda$ = {gradients_data.lam_vals[i]}")
            ax.set_xlabel(r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$")
            ax.set_ylabel("Probability density")
            ax.text(
                0.05,
                0.95,
                f"Std. dev. = {_np.std(gradients_data.gradients[i]):.2f}"
                + r" kcal mol$^{-1}$",
                transform=ax.transAxes,
            )
            ax.text(
                0.05,
                0.9,
                f"Mean = {_np.mean(gradients_data.gradients[i]):.2f}"
                + r" kcal mol$^{-1}$",
                transform=ax.transAxes,
            )
            # Check if there is a significant difference between any of the sets of gradients, if we have more than one repeat
            # compare samples
            if ensemble_size > 1:
                stat, p = _kruskal(*gradients_data.subsampled_gradients[i])
                ax.text(
                    0.05, 0.85, f"Kruskal-Wallis p = {p:.2f}", transform=ax.transAxes
                )
                # If there is a significant difference, highlight the window
                if p < 0.05:
                    ax.tick_params(color="red")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red")
        # Hide redundant axes
        else:
            ax.remove()

    fig.tight_layout()
    name = f"{output_dir}/gradient_hists"
    if gradients_data.equilibrated:
        name += "_equilibrated"
    fig.savefig(
        name, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    _plt.close(fig)


def plot_gradient_timeseries(
    gradients_data: GradientData, output_dir: str, run_nos: _Optional[_List[int]] = None
) -> None:
    """
    Plot timeseries of the gradients for a list of lambda windows.
    If equilibrated is True, only data after equilibration is used.

    Parameters
    ----------
    gradients_data : GradientData
        GradientData object containing the gradient data.
    output_dir : str
        Directory to save the plot to.
    run_nos : List[int], Optional, default: None
        The run numbers to use. If None, all runs will be used.

    Returns
    -------
    None
    """
    # Plot mixed gradients for each window
    n_lams = len(gradients_data.lam_vals)
    fig, axs = _plt.subplots(
        nrows=_ceil(n_lams / 8), ncols=8, figsize=(40, 5 * (n_lams / 8))
    )
    for i, ax in enumerate(axs.flatten()):  # type: ignore
        if i < n_lams:
            # One histogram for each simulation
            for j, gradients in enumerate(gradients_data.gradients[i]):
                ax.plot(
                    gradients_data.times[i],
                    gradients,
                    alpha=0.5,
                    label=f"Run {run_nos[j] if run_nos else j+1}",
                )
            ax.legend()
            ax.set_title(f"$\lambda$ = {gradients_data.lam_vals[i]}")
            ax.set_xlabel("Time / ns")
            ax.set_ylabel(r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$")
            ax.text(
                0.05,
                0.95,
                f"Std. dev. = {_np.std(gradients_data.gradients[i]):.2f}"
                + r" kcal mol$^{-1}$",
                transform=ax.transAxes,
            )
            ax.text(
                0.05,
                0.9,
                f"Mean = {_np.mean(gradients_data.gradients[i]):.2f}"
                + r" kcal mol$^{-1}$",
                transform=ax.transAxes,
            )

    fig.tight_layout()
    name = f"{output_dir}/gradient_timeseries"
    if gradients_data.equilibrated:
        name += "_equilibrated"
    fig.savefig(
        name, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    _plt.close(fig)


def plot_equilibration_time(lam_windows: _List["LamWindows"], output_dir: str) -> None:  # type: ignore
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
    fig, ax = _plt.subplots(figsize=(8, 6))
    # Plot the total time simulated per simulation, so we can see how efficient
    # the protocol is
    ax.bar(
        [win.lam for win in lam_windows],
        [
            win.sims[0].tot_simtime for win in lam_windows
        ],  # All sims at given lam run for same time
        width=0.02,
        edgecolor="black",
        label="Total time simulated per simulation",
    )
    # Now plot the equilibration time
    ax.bar(
        [win.lam for win in lam_windows],
        [win.equil_time for win in lam_windows],
        width=0.02,
        edgecolor="black",
        label="Equilibration time per simulation",
    )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Time (ns)")
    fig.legend()
    fig.savefig(
        f"{output_dir}/equil_times",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )


def plot_overlap_mat(
    ax: _plt.Axes,
    name: str,
    mbar_file: _Optional[str] = None,
    predicted: bool = False,
    gradient_data: _Optional[GradientData] = None,
    color_bar_cutoffs=[0, 0.03, 0.1, 0.3, 1],
) -> None:
    """
    Plot the overlap matrix for a given MBAR file on the supplied axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis on which to plot.
    name : str
        Name of the plot.
    mbar_file : str, optional, default=None
        Path to MBAR file.
    predicted : bool, default=False
        If True, the overlap matrix is predicted from the variances of the
        gradient alone.
    gradient_data : GradientData, optional
        GradientData object containing the gradient data. Only required if
        predicted is True.

    Returns
    -------
    None
    """
    if predicted and not gradient_data:
        raise ValueError("GradientData object must be supplied if predicted is True.")
    if not predicted and not mbar_file:
        raise ValueError("MBAR file must be supplied if predicted is False.")
    if predicted:
        overlap_mat = gradient_data.get_predicted_overlap_mat()  # type: ignore
    else:
        overlap_mat = _read_overlap_mat(mbar_file)  # type: ignore

    # Tuple of colours and associated font colours.
    # The last and first colours are for the top and bottom of the scale
    # for the continuous colour bar, but are ignored for the discrete bar.
    all_colors = (
        ("#FBE8EB", "black"),  # Lighter pink
        ("#FFD3E0", "black"),
        ("#88CCEE", "black"),
        ("#78C592", "black"),
        ("#117733", "white"),
        ("#004D00", "white"),
    )  # Darker green

    # Set the colour map.
    # Create a color map using the extended palette and positions
    box_colors = [all_colors[i][0] for i in range(len(color_bar_cutoffs) + 1)]
    cmap = _colors.LinearSegmentedColormap.from_list(
        "CustomMap", list(zip(color_bar_cutoffs, box_colors))
    )

    # Normalise the same way each time so that plots are always comparable.
    norm = _colors.Normalize(vmin=0, vmax=1)

    # Create the heatmap. Separate the cells with white lines.
    im = ax.imshow(overlap_mat, cmap=cmap, norm=norm)
    num_rows = len(overlap_mat[0])
    for i in range(num_rows - 1):
        for j in range(num_rows - 1):
            # Make sure these are on the edges of the cells.
            ax.axhline(i + 0.5, color="white", linewidth=0.5)
            ax.axvline(j + 0.5, color="white", linewidth=0.5)

    # Label each cell with the overlap value.
    for i in range(num_rows):
        for j in range(num_rows):
            # Get the text colour based on the overlap value.
            overlap_val = overlap_mat[i][j]
            # Get the index of first color bound greater than the overlap value.
            for idx, bound in enumerate(color_bar_cutoffs):
                if bound > overlap_val:
                    break
            text_color = all_colors[1:-1][idx - 1][1]
            ax.text(
                j,
                i,
                "{:.2f}".format(overlap_mat[i][j]),
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    # Create a colorbar. Reduce the height of the colorbar to match the figure and remove the border.
    cbar = ax.figure.colorbar(im, ax=ax, cmap=cmap, norm=norm, shrink=0.7)
    cbar.outline.set_visible(False)

    # Set the axis labels.
    ax.set_xlabel(r"$\lambda$ Index")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(r"$\lambda$ Index")

    ticks = [x for x in range(0, num_rows)]

    # Set ticks every lambda window.
    ax.set_xticks(ticks)
    ax.xaxis.tick_top()
    ax.set_yticks(ticks)

    # Remove the borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title(name)


def plot_overlap_mats(
    output_dir: str,
    nlam: int,
    run_nos: _Optional[_List[int]] = None,
    mbar_outfiles: _Optional[_List[str]] = None,
    predicted: bool = False,
    gradient_data: _Optional[GradientData] = None,
) -> None:
    """
    Plot the overlap matrices for all mbar outfiles supplied.

    Parameters
    ----------
    output_dir : str
        The directory to save the plot to.
    nlam : int
        Number of lambda windows.
    run_nos : Optional[List[int]], default=None
        List of run numbers to use for MBAR. If None, all runs will be used.
    mbar_outfiles : Optional[List[str]], default=None
        List of MBAR outfiles. It is assumed that these are passed in the same
        order as the runs they correspond to. This is required if predicted
        is False (the default).
    predicted : bool, default=False
        If True, the overlap matrices are predicted from the variances of the
        gradient alone.
    gradient_data : GradientData
        GradientData object containing the gradient data. Only required if predicted
        is True.

    Returns
    -------
    None
    """
    # Check that the passed mbar outfiles, run_nos, and gradient_data are consistent

    if predicted:
        if not gradient_data:
            raise ValueError("GradientData object required if predicted is True.")
        n_runs = 1  # Only one plot if predicted
    else:
        if not mbar_outfiles:
            raise ValueError("MBAR outfiles required if predicted is False.")
        n_runs = len(mbar_outfiles)

    # Create the figure and axis. Use a default size for fewer than 16 windows,
    # otherwise scale the figure size to the number of windows.
    if nlam < 8:
        fig, axs = _plt.subplots(1, n_runs, figsize=(4 * n_runs, 4), dpi=300)
    else:
        fig, axs = _plt.subplots(
            1, n_runs, figsize=(n_runs * nlam / 2, nlam / 2), dpi=300
        )

    # Avoid not subscriptable errors when there is only one run
    if n_runs == 1:
        axs = [axs]

    for i in range(n_runs):
        plot_overlap_mat(
            ax=axs[i],
            name=f"Run {i+1}" if not predicted else "Predicted",
            mbar_file=mbar_outfiles[i] if mbar_outfiles else None,
            predicted=predicted,
            gradient_data=gradient_data,
        )

    fig.tight_layout()
    name = (
        f"{output_dir}/overlap_mats"
        if not predicted
        else f"{output_dir}/predicted_overlap_mats"
    )
    fig.savefig(name)


def plot_convergence(
    fracts: _np.ndarray,
    dgs: _np.ndarray,
    tot_simtime: float,
    equil_time: float,
    output_dir: str,
    n_runs: int,
) -> None:
    """
    Plot convergence of free energy estimate as a function of the total
    simulation time.

    Parameters
    ----------
    fracts : np.ndarray
        Array of fractions of the total equilibrated simulation time at which the dgs were calculated.
    dgs : np.ndarray
        Array of free energies at each fraction of the total equilibrated simulation time. This has
        ensemble size dimensions.
    tot_simtime : float
        Total simulation time for the runs included.
    equil_time : float
        Equilibration time (per run)
    output_dir : str
        Directory to save the plot to.
    n_runs : int
        Number of runs used to calculate the free energy estimate.
    """
    # Convert fraction of the equilibrated simulation time to total simulation time in ns
    tot_equil_time = equil_time * n_runs
    times = fracts * (tot_simtime - tot_equil_time) + tot_equil_time
    # Add zero time to the start
    times = _np.concatenate((_np.array([0]), times))

    # Add single Nan to correspond to zero time
    nans = _np.empty((dgs.shape[0], 1))
    nans[:] = _np.nan
    dgs = _np.hstack((nans, dgs))

    # Plot the free energy estimate as a function of the total simulation time
    name = "convergence"
    if equil_time == 0:
        name += "_no_equil"
    outfile = _os.path.join(output_dir, f"{name}.png")
    general_plot(
        times,
        dgs,
        "Total Simulation Time / ns",
        "Free energy / kcal mol$^{-1}$",
        outfile,
    )


def plot_sq_sem_convergence(
    fracts: _np.ndarray,
    dgs: _np.ndarray,
    tot_simtime: float,
    equil_time: float,
    output_dir: str,
    n_runs: int,
) -> None:
    """
    Plot convergence of the squared standard error of the mean of the free energy
    estimate as a function of the total simulation time.

    Parameters
    ----------
    fracts : np.ndarray
        Array of fractions of the total equilibrated simulation time at which the dgs were calculated.
    dgs : np.ndarray
        Array of free energies at each fraction of the total equilibrated simulation time. This has
        ensemble size dimensions.
    tot_simtime : float
        Total simulation time for the runs included.
    equil_time : float
        Equilibration time (per run)
    output_dir : str
        Directory to save the plot to.
    n_runs : int
        Number of runs used to calculate the free energy estimate.
    """
    # Convert fraction of the equilibrated simulation time to total simulation time in ns
    tot_equil_time = equil_time * n_runs
    times = fracts * (tot_simtime - tot_equil_time) + tot_equil_time
    # Add zero time to the start
    times = _np.concatenate((_np.array([0]), times))

    # Add single Nan to correspond to zero time
    nans = _np.empty((dgs.shape[0], 1))
    nans[:] = _np.nan
    dgs = _np.hstack((nans, dgs))

    # Get the squared standard error of the mean
    sq_sems = _np.square(_np.std(dgs, axis=0)) / dgs.shape[0]

    # Plot the free energy estimate as a function of the total simulation time
    name = "convergence_sq_sem"
    if equil_time == 0:
        name += "_no_equil"
    outfile = _os.path.join(output_dir, f"{name}.png")
    general_plot(
        times,
        sq_sems,
        "Total Simulation Time / ns",
        r"$\mathrm{SEM}^2$ / kcal$^{2}$ mol$^{-2}$",
        outfile,
    )


def _plot_mbar_gradient_convergence_single_run(
    ax: _plt.Axes,
    fracts: _np.ndarray,
    mbar_grads: _List[_Dict[str, _np.ndarray]],
    simtime: float,
    equil_time: float,
    run_name: str,
) -> _cm.ScalarMappable:
    """
    Plot the convergence of the gradients obtained from MBAR as a function of simulation
    time, on the axis supplied. Note that final gradients are subtracted so that changes
    are more obvious. Because of this, the integral of any given "PMF" gives the difference
    in the free energy estimate between the current and final times.

    Parameters
    ----------
    ax : matplotlib axis
        Axis on which to plot.
    fracts : np.ndarray
        Array of fractions of the total simulation time at which the gradients were calculated.
    mbar_grads : List[Dict[str, np.ndarray]]
        List of Dictionary of gradients obtained from MBAR. Each list corresponds to a given
        fraction of simulation time. The dict should contain the keys "lam_vals", "grads", and
        "grad_errs".
    simtime : float
        Total simulation time in ns.
    equil_time : float
        Equilibration time in ns.
    run_name: str
        The name of the run.

    Returns
    -------
    mapper : matplotlib.cm.ScalarMappable
        Colour mapper used to map simulation time to colour.
    """
    # Subtract the final gradient from each gradient
    final_grads = mbar_grads[-1]
    for mbar_grad in mbar_grads:
        mbar_grad["grads"] -= final_grads["grads"]

    # Get a list of times sampled
    times = fracts * (simtime - equil_time) + equil_time

    # Get a colour mapper to map simulation time to colour
    norm = _colors.Normalize(vmin=times[0], vmax=times[-1], clip=True)
    mapper = _cm.ScalarMappable(norm=norm, cmap=_cm.brg)

    # Plot the free energy estimate as a function of the total simulation time
    for i, mbar_grad in enumerate(mbar_grads):
        ax.plot(
            mbar_grad["lam_vals"],
            mbar_grad["grads"],
            color=mapper.to_rgba(times[i]),
        )

    # Labels
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(
        r"$\langle \frac{\mathrm{d}h}{\mathrm{d}\lambda}\rangle _{\lambda} $ / kcal mol$^{-1}$"
    ),
    ax.set_title(run_name)

    # Return the colour mapper so we can add it to the plot
    return mapper


def plot_mbar_gradient_convergence(
    fracts: _np.ndarray,
    mbar_grads: _List[_Dict[str, _Dict[str, _np.ndarray]]],
    simtime_per_run: float,
    equil_time_per_run: float,
    output_dir: str,
) -> None:
    """
    Plot the convergence of the gradients obtained from MBAR as a function of simulation
    time. Note that final gradients are subtracted so that changes are more obvious. Because
    of this, the integral of any given "PMF" gives the difference in the free energy estimate
    between the current and final times.

    Parameters
    ----------
    fracts : np.ndarray
        Array of fractions of the total simulation time at which the dgs were calculated.
    mbar_grads : Dict[str, Dict[str, np.ndarray]]
        List of Dictionary of gradients obtained from MBAR. Each list corresponds to a given
        fraction of simulation time. The first dict key is the name of the run. The inner dict
        should contain the keys "lam_vals", "grads", and "grad_errs".
    simtime_per_run : float
        Simulation time per run in ns.
    equil_time_per_run : float
        Equilibration time per run in ns.
    output_dir : str
        Directory to save the plot to.
    """
    n_runs = len(mbar_grads[0])
    fig, axs = _plt.subplots(1, n_runs, figsize=(5 * n_runs, 4), dpi=300)
    # Rearrange the dictionary for plotting individual runs
    mbar_grads_by_run = {run_name: [] for run_name in mbar_grads[0]}
    for mbar_grad in mbar_grads:
        for run_name in mbar_grad:
            mbar_grads_by_run[run_name].append(mbar_grad[run_name])

    for i, (run, grads) in enumerate(mbar_grads_by_run.items()):
        mapper = _plot_mbar_gradient_convergence_single_run(
            ax=axs[i],
            fracts=fracts,
            mbar_grads=grads,
            simtime=simtime_per_run,
            equil_time=equil_time_per_run,
            run_name=run.replace("_", " "),
        )
        # Add a colourbar
        fig.colorbar(mapper, ax=axs[i]).set_label("Simulation time / ns")

    name = "mbar_gradient_convergence"
    if equil_time_per_run == 0:
        name += "_no_equil"
    outfile = _os.path.join(output_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(
        outfile, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    _plt.close(fig)


def plot_mbar_pmf(outfiles: _List[str], output_dir: str) -> None:
    """
    Plot the PMF from MBAR for each run.

    Parameters
    ----------
    outfiles : List[str]
        List of MBAR output files. It is assumed that
        these are passed in the same order as the runs
        they correspond to.
    output_dir : str
        Directory to save the plot to.

    Returns
    -------
    None
    """
    lams_overall = []
    dgs_overall = []
    for i, out_file in enumerate(outfiles):
        lams, dgs, _ = _read_mbar_pmf(out_file)
        if i == 0:
            lams_overall = lams
        if len(lams) != len(lams_overall):
            raise ValueError("Lambda windows do not match between runs.")
        dgs_overall.append(dgs)

    general_plot(
        _np.array(lams_overall),
        _np.array(dgs_overall),
        r"$\lambda$",
        "Free energy / kcal mol$^{-1}$",
        outfile=f"{output_dir}/mbar_pmf.png",
    )


def plot_rmsds(
    lam_windows: _List["LamWindows"],
    output_dir: str,
    selection: str,
    group_selection: _Optional[str] = None,
) -> None:  # type: ignore
    """
    Plot the RMSDs for each lambda window. The reference used is the
    first frame of the trajectory in each case.

    Parameters
    ----------
    lam_windows : List[LamWindows]
        List of LamWindows objects.
    output_dir : str
        Directory to save the plot to.
    selection: str
        The selection, written using the MDAnalysis selection language, to
        use for the calculation of RMSD.
    group_selection: str, Optional, Default = None
        The selection, written using the MDAnalysis selection language, to
        use for the calculation of RMSD after alignment has been carried out
        according to "selection". If None, the "selection" selection
        passed to will be used to calculate RMSD as well as for alignment.

    Returns
    -------
    None
    """
    n_lams = len(lam_windows)
    ncols = 8 if n_lams > 8 else n_lams
    nrows = _ceil(n_lams / 8)
    figsize = (4 * ncols, 4 * nrows)
    fig, axs = _plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=300)
    axs = [axs] if n_lams == 1 else axs.flatten()

    # Take the overall reference as the first frame of the first simulation
    reference_traj = _os.path.join(
        lam_windows[0].sims[0].output_dir, "traj000000001.dcd"
    )

    for i, ax in enumerate(axs):  # type: ignore
        if i < n_lams:
            lam_window = lam_windows[i]
            # One set of RMSDS for each lambda window
            input_dirs = [sim.output_dir for sim in lam_windows[i].sims]
            rmsds, times = _get_rmsd(
                input_dirs=input_dirs,
                selection=selection,
                tot_simtime=lam_window.sims[0].tot_simtime,
                reference_traj=reference_traj,
                group_selection=group_selection,
            )  # Total simtime should be the same for all sims
            ax.legend()
            ax.set_title(f"$\lambda$ = {lam_window.lam}")
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel(r"RMSD ($\AA$)")
            for j, rmsd in enumerate(rmsds):
                ax.plot(times, rmsd, label=f"Run {j+1}")
            ax.legend()

            # If we have equilibration data, plot this
            if lam_window._equilibrated:  # Avoid triggering slow equilibration check
                ax.axvline(x=lam_window.equil_time, color="red", linestyle="dashed")

        # Hide redundant axes
        else:
            ax.remove()

    fig.tight_layout()

    group_selection_name = (
        "none" if not group_selection else group_selection.replace(" ", "")
    )
    name = f"{output_dir}/rmsd_{selection.replace(' ','')}_{group_selection_name}"  # Use selection string to make sure save name is unique
    fig.savefig(
        name, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    _plt.close(fig)


def plot_against_exp(
    all_results: _pd.DataFrame,
    output_dir: str,
    offset: bool = False,
    stats: _Optional[_Dict] = None,
) -> None:
    """
    Plot all results from a set of calculations against the
    experimental values.

    Parameters
    ----------
    all_results : _pd.DataFrame
        A DataFrame containing the experimental and calculated
        free energy changes and errors.
    output_dir : str
        Directory to save the plot to.
    offset: bool, Optional, Default = False
        Whether the calculated absolute binding free energies have been
        offset so that the mean experimental and calculated values are the same.
    stats: Dict, Optional, Default = None
        A dictionary of statistics, obtained using analyse.analyse_set.compute_stats
    """
    # Check that the correct columns have been supplied
    required_columns = [
        "calc_base_dir",
        "exp_dg",
        "exp_er",
        "calc_cor",
        "calc_dg",
        "calc_er",
    ]
    if list(all_results.columns) != required_columns:
        raise ValueError(
            f"The experimental values file must have the columns {required_columns} but has the columns {all_results.columns}"
        )

    # Create the plot
    fig, ax = _plt.subplots(1, 1, figsize=(6, 6), dpi=1000)
    ax.errorbar(
        x=all_results["exp_dg"],
        y=all_results["calc_dg"],
        xerr=all_results["exp_er"],
        yerr=all_results["calc_er"],
        ls="none",
        c="black",
        capsize=2,
        lw=0.5,
    )
    ax.scatter(x=all_results["exp_dg"], y=all_results["calc_dg"], s=50, zorder=100)
    ax.set_ylim([-18, 0])
    ax.set_xlim([-18, 0])
    ax.set_aspect("equal")
    ax.set_xlabel(r"Experimental $\Delta G^o_{\mathrm{Bind}}$ / kcal mol$^{-1}$")
    ax.set_ylabel(r"Calculated $\Delta G^o_{\mathrm{Bind}}$ / kcal mol$^{-1}$")
    # 1 kcal mol-1
    ax.fill_between(
        x=[-25, 0],
        y2=[-24, 1],
        y1=[-26, -1],
        lw=0,
        zorder=-10,
        alpha=0.5,
        color="darkorange",
    )
    # 2 kcal mol-1
    ax.fill_between(
        x=[-25, 0],
        y2=[-23, 2],
        y1=[-27, -2],
        lw=0,
        zorder=-10,
        color="darkorange",
        alpha=0.2,
    )

    # Add text, including number of ligands and stats if supplied
    n_ligs = len(all_results["calc_dg"])
    ax.text(0.03, 0.95, f"{n_ligs} ligands", transform=ax.transAxes)
    if stats:
        stats_text = ""
        for stat, label in zip(
            ["r2", "mue", "rho", "tau"],
            ["R$^2$", "MUE", r"Spearman $\rho$", r"Kendall $\tau$"],
        ):
            stats_text += f"{label}: {stats[stat][0]:.2f}$^{{{stats[stat][1]:.2f}}}_{{{stats[stat][2]:.2f}}}$\n"
        ax.text(0.55, 0, stats_text, transform=ax.transAxes)

    if offset:
        name = f"{output_dir}/overall_results_offset.png"
    else:
        name = f"{output_dir}/overall_results.png"
    fig.savefig(
        name, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    _plt.close(fig)


def plot_gelman_rubin_rhat(
    rhat_dict: _Dict[str, float],
    output_dir: str,
    cutoff: float = 1.1,
) -> None:
    """
    Plot the Gelman-Rubin Rhat statistic for each lambda window.

    Parameters
    ----------
    rhat_dict : Dict[str, float]
        A dictionary of the Rhat statistic for each lambda window.
    output_dir : str
        Directory to save the plot to.
    cutoff : float, Optional, Default = 1.1
        The cutoff for the Rhat statistic. The empirical 1.1 is default.

    Returns
    -------
    None
    """
    fig, ax = _plt.subplots(figsize=(8, 6))

    ax.bar(
        rhat_dict.keys(),
        rhat_dict.values(),
        width=0.02,
        edgecolor="black",
    )
    # This shouldn't be below 1, so don't show values below 1
    ax.set_ylim(bottom=0.98)

    # Set a horizontal line at the cutoff value
    ax.axhline(y=cutoff, color="red", linestyle="dashed")

    ax.set_ylabel(r"$\hat{R}$")
    ax.set_xlabel(r"$\lambda$")

    name = f"{output_dir}/gelman_rubin_rhat.png"
    fig.savefig(
        name, dpi=300, bbox_inches="tight", facecolor="white", transparent=False
    )
    _plt.close(fig)


def plot_comparitive_convergence(
    sim_runners: _SimulationRunnerIterator,
    output_dir: str = ".",
    equilibrated: bool = False,
    mode: str = "cumulative",
    name: _Optional[str] = None,
) -> None:
    """
    Plot the convergence of multiple simulation runners against each other.

    Parameters
    ----------
    sim_runners : List[sim_runner]
        The simulation runners to compare.
    output_dir : str, optional
        The directory to save the plot to. Defaults to the current directory.
    equilibrated : bool, optional, default=False
        Whether to use the equilibrated simulation time or the total simulation time. If False,
        all simulation data will be used, otherwise only the equilibrated data will be used.
    mode : str, optional, default="cumulative"
        "cumulative" or "block". The type of averaging to use. In both cases,
        20 MBAR evaluations are performed per simulation runner.
    name : str, optional
        The name of the plot. Defaults to "comparitive_convergence".

    Returns
    -------
    None
    """
    # Get the convergence data for each simulation runner
    convergence_data = _get_comparitive_convergence_data(
        sim_runners, equilibrated, mode
    )

    # Plot the convergence data
    fig, ax = _plt.subplots(figsize=(8, 6))
    for i, (times, dgs) in enumerate(convergence_data):
        # Select a single colour for each simulation runner
        color = _plt.cm.tab10(i)
        # For each of the replicates, plot the convergence data
        for j in range(dgs.shape[0]):
            ax.plot(times, dgs[j], color=color, alpha=0.5, linestyle="dashed")
        # Add the mean and 95 % CI
        y_avg = _np.mean(dgs, axis=0)
        y_err = _stats.t.interval(
            0.95, len(dgs) - 1, loc=y_avg, scale=_stats.sem(dgs, axis=0)
        )
        ax.plot(
            times,
            y_avg,
            label=f"{sim_runners.base_dirs[i]} mean",
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            times,
            y_err[0],
            y_err[1],
            alpha=0.2,
            color=color,
        )

    xlabel = (
        "Cumulative Total Sampling Time (Equilibration Ignored) / ns"
        if not equilibrated
        else "Cumulative Equilibrated Sampling Time / ns"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta G$ / kcal mol$^{-1}$")
    ax.legend(loc="best")
    name = name if name else "comparitive_convergence"
    fig.savefig(
        f"{output_dir}/{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    _plt.close(fig)


def plot_comparitive_convergence_sem(
    sim_runners: _SimulationRunnerIterator,
    output_dir: str = ".",
    equilibrated: bool = False,
    mode: str = "cumulative",
    name: _Optional[str] = None,
    color_indices: _Optional[_List[int]] = None,
) -> None:
    """
    Plot the convergence of the SEM of the free energy changes
    for simulation runners against each other.

    Parameters
    ----------
    sim_runners : List[sim_runner]
        The simulation runners to compare.
    output_dir : str, optional
        The directory to save the plot to. Defaults to the current directory.
    equilibrated : bool, optional, default=False
        Whether to use the equilibrated simulation time or the total simulation time. If False,
        all simulation data will be used, otherwise only the equilibrated data will be used.
    mode : str, optional, default="cumulative"
        "cumulative" or "block". The type of averaging to use. In both cases,
        20 MBAR evaluations are performed per simulation runner.
    name : str, optional
        The name of the plot. Defaults to "comparitive_convergence".
    color_indices : List[int], optional
        The color group to use for the simulation runners. If None, a
        different color will be used for each simulation runner.

    Returns
    -------
    None
    """
    if color_indices and len(color_indices) != len(sim_runners):
        raise ValueError(
            "If color_indices is supplied, it must have the same length as sim_runners."
        )

    # Get the convergence data for each simulation runner
    convergence_data = _get_comparitive_convergence_data(
        sim_runners, equilibrated, mode
    )

    # Plot the convergence data
    fig, ax = _plt.subplots(figsize=(8, 6))
    for i, (times, dgs) in enumerate(convergence_data):
        # Select a single colour for each simulation runner
        color = (
            _plt.cm.tab10(i) if not color_indices else _plt.cm.tab10(color_indices[i])
        )
        # Calculate the squared SEM at each time point
        sq_sems = (_np.std(dgs, axis=0)) / _np.sqrt(dgs.shape[0])
        ax.plot(
            times,
            sq_sems,
            label=f"{sim_runners.base_dirs[i]}",
            color=color,
        )

    xlabel = (
        "Cumulative Total Sampling Time (Equilibration Ignored) / ns"
        if not equilibrated
        else "Cumulative Equilibrated Sampling Time / ns"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\mathrm{SEM}$ / kcal mol$^{-1}$")
    ax.legend(loc="best")
    name = name if name else "comparitive_sem_convergence"
    fig.savefig(
        f"{output_dir}/{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    _plt.close(fig)


def plot_normality(data: _np.ndarray, output_dir: str) -> None:
    """
    Plot the histogram and QQ plot for a given set of data.

    Parameters
    ----------
    data : np.ndarray
        The data to plot.
    output_dir : str
        The directory to save the plot to.

    Returns
    -------
    None
    """
    # Plot the histogram and the QQ plot side-by-side
    fig, axs = _plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # Plot the histogram, kernel density estimate, and QQ plot
    axs[0].hist(data, edgecolor="black")
    _sns.kdeplot(data, ax=axs[1], color="black", linewidth=2)
    _stats.probplot(data, plot=axs[2])

    # Set the axis labels
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Histogram")
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Kernel Density Estimate")
    axs[2].set_xlabel("Theoretical Normal Quantiles")
    axs[2].set_ylabel("Ordered Values")
    axs[2].set_title("QQ Plot")

    # Compute the Shapiro-Wilk test and print the p value
    _, p_value = _stats.shapiro(data)
    axs[2].text(
        0.5,
        0.95,
        f"Shapiro-Wilk p-value: {p_value:.2f}",
        transform=axs[2].transAxes,
        horizontalalignment="center",
        verticalalignment="top",
    )

    # Stop the labels overlapping
    fig.tight_layout()
    fig.savefig(
        f"{output_dir}/normality_plot.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    _plt.close(fig)


def plot_av_waters(
    lam_windows: _List["LamWindow"],
    output_dir: str,
    percent_traj: float,
    index: int,
    length: float,
    index2: _Optional[int] = None,
    length2: _Optional[float] = None,
    run_nos: _Optional[_List[int]] = None,
) -> None:
    """
    Calculate average number of waters within given distance
    of an atom (or two atoms) with given index over the
    specified percentage of the end of the trajectory,
    for all simulations for all runs for all lambda windows
    supplied.

    Parameters
    ----------
    lam_windows : List[LamWindow]
        List of LamWindow objects for which to calculate average number of waters

    output_dir : str
        Directory to save the plot to.

    percent_traj : float
        percentage of trajectory (beginning from end) over
        which to average

    index : int
        Atom from which distance is calculated

    length : float
        Distance in Angstrom

    index2 : int, optional, default=None
        Optional. Index of second atom from which water must be within a specified distance

    length2 : float, optional, default=None
        Optional. Distance (Angstrom) from second atom which water must be within

    run_nos : Optional[List[int]], default=None
        Optional. List of run numbers to include in the analysis. If None, all runs will be included.

    Returns
    -------
    avg_close_waters: _np.ndarray
        Average number of waters within the specified distance(s) of the specified atom(s) for each
        lambda window for each run. Shape is (n_runs, n_lam_windows).
    """
    # Get the data
    lam_vals = _np.array([lam.lam for lam in lam_windows])
    av_waters = _get_av_waters_stage(
        lam_windows=lam_windows,
        percent_traj=percent_traj,
        index=index,
        length=length,
        index2=index2,
        length2=length2,
        run_nos=run_nos,
    )

    # Plot the data
    y_label = (
        f"Average number of waters within \n {length} "
        + r"$\mathrm{\AA}$ of atom "
        + f"{index}"
        if not index2
        else f"Average number of waters within\n {length} "
        + r"$\mathrm{\AA}$ of index "
        + f"{index} "
        + f"and {length2} "
        + r"$\mathrm{\AA}$ of index "
        + f"{index2}"
    )
    general_plot(
        x_vals=lam_vals,
        y_vals=av_waters,
        x_label=r"$\lambda$",
        y_label=y_label,
        outfile=f"{output_dir}/av_waters_{index}_{length}_{index2}_{length2}.png",
        run_nos=run_nos,
    )
