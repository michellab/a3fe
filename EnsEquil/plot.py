"""Plotting functions"""

import matplotlib.pyplot as _plt
import numpy as _np
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
    # Get mean and variance of gradients, including both intra-run and inter-run components
    variances_all_winds = []
    means_all_winds = []
    for lam in lams:
        gradients_wind = []
        for sim in lam.sims:
            gradients_wind.append(sim.read_gradients(equilibrated_only=equilibrated))
        # Get intra-run quantities
        vars_intra = _np.var(gradients_wind, axis=1)
        var_intra = _np.mean(vars_intra)  # Mean of the variances - no need for roots as not SD
        means = _np.mean(gradients_wind, axis=1)
        # Get inter-run quantities
        mean_grads = _np.mean(means)
        means_all_winds.append(mean_grads)
        var_inter = _np.var(means)
        # Combine
        tot_var = 0
        if inter_var:
            tot_var += var_inter
        if intra_var:
            tot_var += var_intra
        variances_all_winds.append(tot_var)

    # Make plots of variance of gradients
    fig, ax = _plt.subplots(figsize=(8, 6))

    if plot_mean and not plot_variance:
        ax.bar([win.lam for win in lams],
               means_all_winds,
               width=0.02, edgecolor='black',
               yerr=_np.sqrt(variances_all_winds))
        ax.set_ylabel(r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$"),

    elif plot_variance and not plot_mean:
        ax.bar([win.lam for win in lams],
               variances_all_winds,
               width=0.02, edgecolor='black')
        ax.set_ylabel(r"Var($\frac{\mathrm{d}h}{\mathrm{d}\lambda}$) / kcal$^{2}$ mol$^{-2}$"),

    elif plot_mean and plot_variance:
        ax.bar([win.lam for win in lams],
               variances_all_winds,
               width=0.02, edgecolor="black", label="Variance")
        ax.bar([win.lam for win in lams],
               [abs(mean)*25 for mean in means_all_winds],
               width=0.02, edgecolor='black', label="|Mean|* 25", alpha=0.5)
        ax.set_ylabel(r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$"),
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
