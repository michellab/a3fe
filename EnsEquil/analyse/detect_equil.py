"""Functions for detecting equilibration based on an ensemble of simulations."""

import numpy as _np
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional
from pymbar import timeseries as _timeseries
import scipy.stats as _stats

from .plot import general_plot as _general_plot


def check_equil_block_gradient(lam_win:"LamWindow") -> _Tuple[bool, _Optional[float]]: # type: ignore
    """
    Check if the ensemble of simulations at the lambda window is
    equilibrated based on the ensemble gradient between averaged blocks.

    Parameters
    ----------
    lam_win : LamWindow
        Lambda window to check for equilibration.   

    Returns
    -------
    equilibrated : bool
        True if the simulation is equilibrated, False otherwise.
    equil_time : float
        Time taken to equilibrate, in ns.
    """
    # Get the gradient threshold and complain if it does not exist
    gradient_threshold = lam_win.gradient_threshold

    # Conversion between time and gradient indices.
    time_to_ind = 1 / (lam_win.sims[0].timestep * lam_win.sims[0].nrg_freq)
    idx_block_size = int(lam_win.block_size * time_to_ind)

    # Read dh/dl data from all simulations and calculate the gradient of the
    # gradient, d_dh_dl
    d_dh_dls = []
    dh_dls = []
    times, _ = lam_win.sims[0].read_gradients()
    equilibrated = False
    equil_time = None

    for sim in lam_win.sims:
        _, dh_dl = sim.read_gradients()  # Times should be the same for all sims
        dh_dls.append(dh_dl)
        # Create array of nan so that d_dh_dl has the same length as times irrespective of
        # the block size
        d_dh_dl = _np.full(len(dh_dl), _np.nan)
        # Compute rolling average with the block size
        rolling_av_dh_dl = lam_win._get_rolling_average(dh_dl, idx_block_size)
        for i in range(len(dh_dl)):
            if i < 2 * idx_block_size:
                continue
            else:
                d_dh_dl[i] = (rolling_av_dh_dl[i] - rolling_av_dh_dl[i - idx_block_size]) / \
                    lam_win.block_size  # Gradient of dh/dl in kcal mol-1 ns-1
        d_dh_dls.append(d_dh_dl)

    # Calculate the mean gradient
    mean_d_dh_dl = _np.mean(d_dh_dls, axis=0)

    # Check if the mean gradient has been below the threshold at any point, making
    # sure to exclude the initial nans
    last_grad = mean_d_dh_dl[2*idx_block_size]
    for i, grad in enumerate(mean_d_dh_dl[2*idx_block_size:]):
        if gradient_threshold:
            if _np.abs(grad) < gradient_threshold:
                equil_time = times[i]
                break
        # Check if gradient has passed through 0
        # If no gradient threshold is set, this is 
        # the only criterion for equilibration
        if _np.sign(last_grad) != _np.sign(grad):
            equil_time = times[i]
            break
        last_grad = grad

    if equil_time is not None:
        equilibrated = True

    # Write out data
    with open(f"{lam_win.output_dir}/equilibration_block_gradient.txt", "w") as ofile:
        ofile.write(f"Equilibrated: {equilibrated}\n")
        ofile.write(f"Equilibration time: {equil_time} ns\n")
        ofile.write(f"Block size: {lam_win.block_size} ns\n")

    # Change name of plots depending on whether a gradient threshold is set
    append_to_name = "_threshold" if gradient_threshold else ""

    # Save plots of dh/dl and d_dh/dl
    _general_plot(x_vals=times,
            y_vals=_np.array([lam_win._get_rolling_average(dh_dl, idx_block_size) for dh_dl in dh_dls]),
            x_label="Simulation Time per Window per Run / ns",
            y_label=r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$",
            outfile=f"{lam_win.output_dir}/dhdl_block_gradient" + append_to_name,
            # Shift the equilibration time by 2 * block size to account for the
            # delay in the block average calculation.
            vline_val=equil_time + 1 * lam_win.block_size if equil_time is not None else None)

    _general_plot(x_vals=times,
            y_vals=_np.array(d_dh_dls),
            x_label="Simulation Time per Window per Run / ns",
            y_label=r"$\frac{\partial}{\partial t}\frac{\partial H}{\partial \lambda}$ / kcal mol$^{-1}$ ns$^{-1}$",
            outfile=f"{lam_win.output_dir}/ddhdl_block_gradient" + append_to_name,
            vline_val=equil_time + 2 * lam_win.block_size if equil_time is not None else None,
            hline_val=0)

    return equilibrated, equil_time

def check_equil_shrinking_block_gradient(lam_win:"LamWindow") -> _Tuple[bool, _Optional[float]]: # type: ignore
    """
    Check if the ensemble of simulations at the lambda window is
    equilibrated based on the ensemble gradient between averaged blocks,
    where several sizes of block are used, starting from the largest possible.
    The window is marked as equilibrated if the absolute gradient is significantly
    below the gradient threshold. This option ignores the simulation block size.

    Parameters
    ----------
    lam_win : LamWindow
        Lambda window to check for equilibration.   

    Returns
    -------
    equilibrated : bool
        True if the simulation is equilibrated, False otherwise.
    equil_time : float
        Time taken to equilibrate, in ns.
    """
    # Get the gradient threshold and complain if it does not exist
    if not lam_win.gradient_threshold:
        raise ValueError("Gradient threshold not set. Please set the gradient threshold to "
                         "use check_equil_block_gradient for equilibration detection.")

    if not lam_win.lam_val_weight:
        raise ValueError("Lambda value weight not set. Please set the lambda value weight to "
                         "use check_equil_shrinking_block_gradient for equilibration detection.")
    
    # Scale the gradient threshold according to how each window contributes to the
    # total uncertainty, as given by TI and the trapezoidal rule.
    gradient_threshold = lam_win.gradient_threshold / lam_win.lam_val_weight

    # Conversion between time and gradient indices.
    time_to_ind = 1 / (lam_win.sims[0].timestep * lam_win.sims[0].nrg_freq)

    # For each fractional block size, check for equilibration
    frac_block_sizes = [1/2, 1/3, 1/4, 1/5] # Fraction of the total simulation time
    equilibrated = False
    equil_time = None
    equil_idx_block_size = None # Int
    equil_block_size = None # Frac
    equil_d_dh_dls = None
    largest_block_dh_dls = []
    largest_idx_block_size = None
    for block_size_ind, frac_block_size in enumerate(frac_block_sizes):
        idx_block_size = int((lam_win.tot_simtime / lam_win.ensemble_size) * frac_block_size * time_to_ind)

        # Block average the dh/dl data
        d_dh_dls = []
        dh_dls = []
        times, _ = lam_win.sims[0].read_gradients()

        if block_size_ind == 1:
            largest_idx_block_size = idx_block_size

        for sim in lam_win.sims:
            _, dh_dl = sim.read_gradients()  # Times should be the same for all sims
            dh_dls.append(dh_dl)
            # Create array of nan so that d_dh_dl has the same length as times irrespective of
            # the block size
            d_dh_dl = _np.full(len(dh_dl), _np.nan)
            # Compute rolling average with the block size
            rolling_av_dh_dl = lam_win._get_rolling_average(dh_dl, idx_block_size)
            for i in range(len(dh_dl)):
                if i < 2 * idx_block_size:
                    continue
                else:
                    d_dh_dl[i] = (rolling_av_dh_dl[i] - rolling_av_dh_dl[i - idx_block_size]) / \
                        lam_win.block_size  # Gradient of dh/dl in kcal mol-1 ns-1
            d_dh_dls.append(d_dh_dl)

        # Store the d_dh_dl with the largest block size
        if block_size_ind == 1:
            largest_block_dh_dls = d_dh_dls

        # Calculate the gradient of the block averaged dh/dl with standard deviation over the repeats
        mean_d_dh_dl = _np.mean(d_dh_dls, axis=0)
        ci_d_dh_dl = _stats.t.interval(0.95,
                                    len(d_dh_dls)-1,
                                    mean_d_dh_dl,
                                    scale=_stats.sem(d_dh_dls))[1] - mean_d_dh_dl

        # Check if the mean gradients has been simultaneously significantly below and above the threshold
        # at any point, making sure to exclude the initial nans
        for i, (mean_grad, ci) in enumerate(zip(mean_d_dh_dl[2*idx_block_size:], ci_d_dh_dl[2*idx_block_size:])):
            if mean_grad + ci < gradient_threshold and mean_grad - ci > -gradient_threshold:
                equil_time = times[i]
                equilibrated = True
                equil_d_dh_dls = d_dh_dls
                equil_block_size = frac_block_size * lam_win.tot_simtime / lam_win.ensemble_size
                equil_idx_block_size = idx_block_size
                break

    # Write out data
    with open(f"{lam_win.output_dir}/equilibration_shrinking_block_gradient.txt", "w") as ofile:
        ofile.write(f"Equilibrated: {equilibrated}\n")
        ofile.write(f"Equilibration time: {equil_time} ns\n")
        ofile.write(f"Successful block size (ns): {equil_block_size}\n")
        ofile.write(f"Gradient threshold: {gradient_threshold}\n")

    # Save plots of dh/dl and d_dh/dl


    idx_block_size = equil_idx_block_size if equil_idx_block_size is not None else largest_idx_block_size
    d_dh_dls =  equil_d_dh_dls if equil_block_size is not None else largest_block_dh_dls
    _general_plot(x_vals=times, # type: ignore
            # Use idx_block_size, the smallest block size tried, to plot the data
            y_vals=_np.array([lam_win._get_rolling_average(dh_dl, idx_block_size) for dh_dl in dh_dls]), # type: ignore
            x_label="Simulation Time per Window per Run / ns",
            y_label=r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$",
            outfile=f"{lam_win.output_dir}/dhdl_block_gradient",
            # Shift the equilibration time by 2 * block size to account for the
            # delay in the block average calculation.
            vline_val=equil_time + 1 * equil_block_size if equil_block_size is not None else None) # type: ignore

    # This will use the latest dh/dls generated with the final block size above
    _general_plot(x_vals=times, # type: ignore
            # The d_dh_dls are already block averaged with the final block size above
            y_vals=_np.array(d_dh_dls), # type: ignore
            x_label="Simulation Time per Window per Run / ns",
            y_label=r"$\frac{\partial}{\partial t}\frac{\partial H}{\partial \lambda} \mathrm{~/~ kcal~} \mathrm{mol}^{-1} \mathrm{ns}^{-1} $",
            outfile=f"{lam_win.output_dir}/ddhdl_block_gradient",
            vline_val=equil_time + 2 * equil_block_size if equil_block_size is not None else None, # type: ignore
            hline_val=0)

    return equilibrated, equil_time


def check_equil_chodera(lam_win:"LamWindow") -> _Tuple[bool, _Optional[float]]: # type: ignore
    """
    Check if the ensemble of simulations at the lambda window is
    equilibrated based Chodera's method of maximising the number
    of uncorrelated samples. This returns equilibrated = False and
    equil_time = None if the number of uncorrelated samples is
    less than 50.

    Please see the following paper for more details:
    J. Chem. Theory Comput. 2016, 12, 4, 1799–1805

    Parameters
    ----------
    lam_win : LamWindow
        Lambda window to check for equilibration.   

    Returns
    -------
    equilibrated : bool
        True if the simulation is equilibrated, False otherwise.
    equil_time : float
        Time taken to equilibrate, in ns.
    """
    # Conversion between time and gradient indices.
    time_to_ind = 1 / (lam_win.sims[0].timestep * lam_win.sims[0].nrg_freq)
    idx_block_size = int(lam_win.block_size * time_to_ind)

    # Read dh/dl data from all simulations
    dh_dls = []
    times, _ = lam_win.sims[0].read_gradients() # Times should be the same for all sims
    equilibrated = False
    equil_time = None

    for sim in lam_win.sims:
        _, dh_dl = sim.read_gradients()  
        dh_dls.append(dh_dl)

    # Calculate the mean gradient
    mean_dh_dl = _np.mean(dh_dls, axis=0)

    # Use Chodera's method on the ensemble average
    t0, g, Neff_max = _timeseries.detectEquilibration(mean_dh_dl)
    equil_time = times[t0]

    # Note that this method will always give an equilibration time
    if Neff_max < 50:
        equilibrated = False
        equil_time = None
    else:
        equilibrated = True

    # Write out data
    with open(f"{lam_win.output_dir}/equilibration_chodera.txt", "w") as ofile:
        ofile.write(f"Equilibrated: {equilibrated}\n")
        ofile.write(f"Equilibration time: {equil_time} ns\n")
        ofile.write(f"Number of uncorrelated samples: {Neff_max}\n")
        ofile.write(f"Staistical inefficiency: {g}\n")

    # Save plots of dh/dl and d_dh/dl
    # Use rolling average to smooth out the data
    rolling_av_time = 0.05 # ns
    rolling_av_block_size=int(rolling_av_time * time_to_ind) # ns
    v_line_x = None if equil_time is None else equil_time + rolling_av_time
    _general_plot(x_vals=times,
          y_vals=_np.array([lam_win._get_rolling_average(dh_dl, rolling_av_block_size) for dh_dl in dh_dls]),
          x_label="Simulation Time per Window per Run / ns",
          y_label=r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$",
          outfile=f"{lam_win.output_dir}/dhdl_chodera",
          # Shift the equilibration time by block size to account for the
          # delay in the rolling average calculation.
          vline_val= v_line_x)

    return equilibrated, equil_time