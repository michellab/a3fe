"""Plotting functions"""

import matplotlib.pyplot as _plt
from math import ceil as _ceil
import numpy as _np
import os as _os
import scipy.stats as _stats
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional


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


def get_gradient_data(lams: _List["LamWindow"], 
                      equilibrated: bool, 
                      inter_var: bool = True,
                      intra_var: bool = True, 
                    ) -> _Tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """ 
    Return the gradients, means, and variances of the gradients for each lambda
    window of a list of LamWindows.

    Parameters
    ----------
    lams : List[LamWindow]
        List of lambda windows.
    equilibrated : bool
        If True, only equilibrated data is used.
    inter_var : bool, optional, default=True
        If True, the inter-run variance is included.
    intra_var : bool, optional, default=True
        If True, the intra-run variance is included.

    Returns
    -------
    gradients_all_winds : np.ndarray
        Array of the gradients for each lambda window.
    means_all_winds : np.ndarray
        Array of the means of the gradients for each lambda window.
    variances_all_winds : np.ndarray
        Array of the variances of the gradients for each lambda window.
    """
    # Get mean and variance of gradients, including both intra-run and inter-run components if specified
    variances_all_winds = []
    means_all_winds = []
    gradients_all_winds = []
    for lam in lams:

        # Get all gradients
        gradients_wind = []
        for sim in lam.sims:
            gradients_wind.append(sim.read_gradients(equilibrated_only=equilibrated)[1])

        # Get intra-run quantities
        vars_intra = _np.var(gradients_wind, axis=1)
        var_intra = _np.mean(vars_intra)  # Mean of the variances - no need for roots as not SD
        means_intra = _np.mean(gradients_wind, axis=1)

        # Get inter-run quantities
        mean_overall = _np.mean(means_intra)
        var_inter = _np.var(means_intra)

        # Store the final results
        tot_var = 0
        if inter_var:
            tot_var += var_inter
        if intra_var:
            tot_var += var_intra
        # Convert to arrays for consistency
        variances_all_winds.append(_np.array(tot_var))
        means_all_winds.append(_np.array(mean_overall))
        gradients_all_winds.append(_np.array(gradients_wind))

    # Convert all to arrays for consistency
    return _np.array(gradients_all_winds), _np.array(means_all_winds), _np.array(variances_all_winds)


def plot_lam_gradient(lams: _List["LamWindow"], outdir: str, 
                       plot_mean: bool, plot_variance: bool,
                       equilibrated: bool, inter_var: bool = True,
                       intra_var: bool = True, log: bool = False,
                       ymax: _Optional[float] = None) -> None:
    """ 
    Plot the variance of the gradients for a list of lambda windows.
    If equilibrated is True, only data after equilibration is used.

    Parameters
    ----------
    lams : List[LamWindow]
        List of lambda windows.
    outdir : str
        Directory to save the plot to.
    plot_mean : bool
        If True, the mean of the gradients is plotted.
    plot_variance : bool
        If True, the variance of the gradients is plotted.
    equilibrated : bool
        If True, only equilibrated data is used.
    inter_var : bool, optional, default=True
        If True, the inter-run variance is included.
    intra_var : bool, optional, default=True
        If True, the intra-run variance is included.
    log : bool, optional, default=False
        If True, the y-axis is plotted on a log scale.
    ymax : None or float, optional, default=None
        If not None, the y-axis is limited to this value.

    Returns
    -------
    None
    """
    if not plot_mean and not plot_variance:
        raise ValueError("Must plot either the mean or variance of the gradients, or both.")
    
    _, means_all_winds, variances_all_winds = get_gradient_data(lams, equilibrated, inter_var, intra_var)

    # Make plots of variance of gradients
    fig, ax = _plt.subplots(figsize=(8, 6))

    if plot_mean and not plot_variance:
        ax.bar([win.lam for win in lams],
               means_all_winds,
               width=0.02, edgecolor='black',
               yerr=_np.sqrt(variances_all_winds))
        ax.set_ylabel(r"$\langle \frac{\mathrm{d}h}{\mathrm{d}\lambda}\rangle _{\lambda} $ / kcal mol$^{-1}$"),

    elif plot_variance and not plot_mean:
        ax.bar([win.lam for win in lams],
               variances_all_winds,
               width=0.02, edgecolor='black')
        ax.set_ylabel(r"Var($\langle \frac{\mathrm{d}h}{\mathrm{d}\lambda}\rangle _{\lambda} $) / kcal$^{2}$ mol$^{-2}$"),

    elif plot_mean and plot_variance:
        ax.bar([win.lam for win in lams],
               variances_all_winds,
               width=0.02, edgecolor="black", label="Variance")
        ax.bar([win.lam for win in lams[1:-1]],
               [abs(mean) for mean in means_all_winds],
               width=0.02, edgecolor='black', label="|Mean|", alpha=0.5)
        ax.set_ylabel(r"$\langle \frac{\mathrm{d}h}{\mathrm{d}\lambda}\rangle _{\lambda} $ / kcal mol$^{-1}$"),
        ax.legend()

    ax.set_xlabel(r"$\lambda$")
    if log:
        ax.set_yscale("log")
    if ymax is not None:
        ax.set_ylim(0, ymax)

    # Name uniquely based on options
    name = f"{outdir}/gradient"
    append_dict = {"_mean": plot_mean,
                   "_variance": plot_variance,
                   "_equilibrated": equilibrated,
                   "_inter": inter_var,
                   "_intra": intra_var,
                   "_ymax": bool(ymax),
                   "_log": log}
    for addition, use in append_dict.items():
        if use:
            name += addition

    fig.savefig(name, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    _plt.close(fig)


def plot_lam_gradient_hists(lams: _List["LamWindow"], outdir: str, equilibrated: bool,) -> None:
    """ 
    Plot histograms of the gradients for a list of lambda windows.
    If equilibrated is True, only data after equilibration is used.

    Parameters
    ----------
    lams : List[LamWindow]
        List of lambda windows.
    outdir : str
        Directory to save the plot to.
    equilibrated : bool
        If True, only equilibrated data is used.

    Returns
    -------
    None
    """
    gradients_all_winds, _, _ = get_gradient_data(lams, equilibrated, inter_var=False, intra_var=False)

    # Plot mixed gradients for each window
    n_lams = len(lams)
    fig, axs = _plt.subplots(nrows=_ceil(n_lams/8), ncols=8, figsize=(40, 5*(n_lams/8)))
    for i, ax in enumerate(axs.flatten()):
        if i < n_lams:
            # One histogram for each simulation
            for j, gradients in enumerate(gradients_all_winds[i]):
                ax.hist(gradients, bins=50, density=True, alpha=0.5, label=f"Run {j+1}")
            ax.legend()
            ax.set_title(f"$\lambda$ = {lams[i].lam}")
            ax.set_xlabel(r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$")
            ax.set_ylabel("Probability density")
            ax.text(0.05, 0.95, f"Std. dev. = {_np.std(gradients_all_winds[i]):.2f}" + r" kcal mol$^{-1}$", transform=ax.transAxes)
            ax.text(0.05, 0.9, f"Mean = {_np.mean(gradients_all_winds[i]):.2f}" + r" kcal mol$^{-1}$", transform=ax.transAxes)
    
    fig.tight_layout()
    name = f"{outdir}/gradient_hists"
    if equilibrated:
        name += "_equilibrated"
    fig.savefig(name, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    _plt.close(fig)
