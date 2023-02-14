"""Functions for detecting equilibration based on an ensemble of simulations."""

import numpy as _np
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional
from pymbar import timeseries as _timeseries

from ._utils import plot as _plot


def check_equil_block_gradient(lam_win:"LamWindow") -> _Tuple[bool, _Optional[float]]:
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

    # Check if the mean gradient has been 0 at any point, making
    # sure to exclude the initial nans
    last_grad = mean_d_dh_dl[2*idx_block_size]
    for i, grad in enumerate(mean_d_dh_dl[2*idx_block_size:]):
        # Check if gradient has passed through 0
        if _np.sign(last_grad) != _np.sign(grad):
            equil_time = times[i]
            break
        last_grad = grad

    if equil_time:
        equilibrated = True

    # Write out data
    with open(f"{lam_win.output_dir}/lambda_{lam_win.lam:.3f}/equilibration_block_gradient.txt", "w") as ofile:
        ofile.write(f"Equilibrated: {equilibrated}\n")
        ofile.write(f"Equilibration time: {equil_time} ns\n")
        ofile.write(f"Block size: {lam_win.block_size} ns\n")

    # Save plots of dh/dl and d_dh/dl
    _plot(x_vals=times,
            y_vals=_np.array([lam_win._get_rolling_average(dh_dl, idx_block_size) for dh_dl in dh_dls]),
            x_label="Simulation Time per Window per Run / ns",
            y_label=r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$",
            outfile=f"{lam_win.output_dir}/lambda_{lam_win.lam:.3f}/dhdl_block_gradient",
            # Shift the equilibration time by 2 * block size to account for the
            # delay in the block average calculation.
            vline_val=equil_time + 1 * lam_win.block_size if equil_time else None)

    _plot(x_vals=times,
            y_vals=_np.array(d_dh_dls),
            x_label="Simulation Time per Window per Run / ns",
            y_label=r"$\frac{\partial}{\partial t}\frac{\partial H}{\partial \lambda}$ / kcal mol$^{-1}$ ns$^{-1}$",
            outfile=f"{lam_win.output_dir}/lambda_{lam_win.lam:.3f}/ddhdl_block_gradient",
            vline_val=equil_time + 2 * lam_win.block_size if equil_time else None,
            hline_val=0)

    return equilibrated, equil_time


def check_equil_chodera(lam_win:"LamWindow") -> _Tuple[bool, _Optional[float]]:
    """
    Check if the ensemble of simulations at the lambda window is
    equilibrated based Chodera's method of maximising the number
    of uncorrelated samples.

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

    # Write out data
    with open(f"{lam_win.output_dir}/lambda_{lam_win.lam:.3f}/equilibration_chodera.txt", "w") as ofile:
        ofile.write(f"Equilibrated: {equilibrated}\n")
        ofile.write(f"Equilibration time: {equil_time} ns\n")
        ofile.write(f"Number of uncorrelated samples: {Neff_max}\n")
        ofile.write(f"Staistical inefficiency: {g}\n")

    # Note that this method will always give an equilibration time
    if equil_time:
        equilibrated = True

    # Save plots of dh/dl and d_dh/dl
    # Use rolling average to smooth out the data
    rolling_av_time = 0.0005 # ns
    rolling_av_block_size=int(rolling_av_time * time_to_ind) # ns
    _plot(x_vals=times,
          y_vals=_np.array([lam_win._get_rolling_average(dh_dl, rolling_av_block_size) for dh_dl in dh_dls]),
          x_label="Simulation Time per Window per Run / ns",
          y_label=r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$",
          outfile=f"{lam_win.output_dir}/lambda_{lam_win.lam:.3f}/dhdl_chodera",
          # Shift the equilibration time by block size to account for the
          # delay in the rolling average calculation.
          vline_val=equil_time + rolling_av_time)

    return equilibrated, equil_time