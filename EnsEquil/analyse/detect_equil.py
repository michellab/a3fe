"""Functions for detecting equilibration based on an ensemble of simulations."""

import numpy as _np
from typing import (
    Dict as _Dict,
    List as _List,
    Tuple as _Tuple,
    Any as _Any,
    Optional as _Optional,
)
from pymbar import timeseries as _timeseries
import scipy.stats as _stats
from statsmodels.tsa.stattools import kpss as _kpss

from .plot import general_plot as _general_plot


def check_equil_block_gradient(lam_win: "LamWindow", run_nos: _Optional[_List[int]]) -> _Tuple[bool, _Optional[float]]:  # type: ignore
    """
    Check if the ensemble of simulations at the lambda window is
    equilibrated based on the ensemble gradient between averaged blocks.

    Parameters
    ----------
    lam_win : LamWindow
        Lambda window to check for equilibration.

    run_nos : List[int], Optional, default: None
        The run numbers to use for MBAR. If None, all runs will be used.

    Returns
    -------
    equilibrated : bool
        True if the simulation is equilibrated, False otherwise.
    equil_time : float
        Time taken to equilibrate per simulation, in ns.
    """
    run_nos = lam_win._get_valid_run_nos(run_nos)

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

    for run_no in run_nos:
        sim = lam_win.sims[run_no - 1]
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
                d_dh_dl[i] = (
                    rolling_av_dh_dl[i] - rolling_av_dh_dl[i - idx_block_size]
                ) / lam_win.block_size  # Gradient of dh/dl in kcal mol-1 ns-1
        d_dh_dls.append(d_dh_dl)

    # Calculate the mean gradient
    mean_d_dh_dl = _np.mean(d_dh_dls, axis=0)

    # Check if the mean gradient has been below the threshold at any point, making
    # sure to exclude the initial nans
    last_grad = mean_d_dh_dl[2 * idx_block_size]
    for i, grad in enumerate(mean_d_dh_dl[2 * idx_block_size :]):
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
        ofile.write(f"Run numbers: {run_nos}\n")

    # Change name of plots depending on whether a gradient threshold is set
    append_to_name = "_threshold" if gradient_threshold else ""

    # Save plots of dh/dl and d_dh/dl
    _general_plot(
        x_vals=times,
        y_vals=_np.array(
            [lam_win._get_rolling_average(dh_dl, idx_block_size) for dh_dl in dh_dls]
        ),
        x_label="Simulation Time per Window per Run / ns",
        y_label=r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$",
        outfile=f"{lam_win.output_dir}/dhdl_block_gradient" + append_to_name,
        # Shift the equilibration time by 2 * block size to account for the
        # delay in the block average calculation.
        vline_val=equil_time + 1 * lam_win.block_size
        if equil_time is not None
        else None,
        run_nos=run_nos,
    )

    _general_plot(
        x_vals=times,
        y_vals=_np.array(d_dh_dls),
        x_label="Simulation Time per Window per Run / ns",
        y_label=r"$\frac{\partial}{\partial t}\frac{\partial H}{\partial \lambda}$ / kcal mol$^{-1}$ ns$^{-1}$",
        outfile=f"{lam_win.output_dir}/ddhdl_block_gradient" + append_to_name,
        vline_val=equil_time + 2 * lam_win.block_size
        if equil_time is not None
        else None,
        hline_val=0,
        run_nos=run_nos,
    )

    return equilibrated, equil_time


def check_equil_multiwindow(
    lambda_windows: _List["LamWindow"],
    output_dir: str,
    run_nos: _Optional[_List[int]] = None,
) -> _Tuple[bool, _Optional[float]]:  # type: ignore
    """
    Check if a set of lambda windows are equilibrated based on the ensemble gradient
    of the free energy change. This is done by block averaging the free energy
    change over a variety of block sizes, starting from the largest possible. The
    stage is marked as equilibrated if the absolute gradient is significantly below
    the gradient threshold, giving the fractional equilibration time. The fractional
    equilibration time is then multiplied by the total simulation time for each
    simulation to obtain the per simulation equilibration times.

    Parameters
    ----------
    lambda_windows : List[LamWindow]
        List of lambda windows to check for equilibration.
    output_dir : str
        Directory to write output files to.
    run_nos : List[int], Optional, default: None
        The run numbers to use. If None, all runs will be used.

    Returns
    -------
    equilibrated : bool
        True if the set of lambda windows is equilibrated, False otherwise.
    fractional_equil_time : float
        Time taken to equilibrate, as a fraction of the total simulation time.
    """
    run_nos = lambda_windows[0]._get_valid_run_nos(run_nos)

    if any([not lam_win.lam_val_weight for lam_win in lambda_windows]):
        raise ValueError(
            "Lambda value weight not set for all windows. Please set the lambda value weight to "
            "use check_equil_multiwindow for equilibration detection."
        )

    # Combine all gradients to get the change in free energy with increasing simulation time.
    # Do this so that the total simulation time for each window is spread evenly over the total
    # simulation time for the whole calculation
    n_runs = len(run_nos)
    overall_dgs = _np.zeros(
        [n_runs, 100]
    )  # One point for each % of the total simulation time
    overall_times = _np.zeros([n_runs, 100])
    for lam_win in lambda_windows:
        for i, run in enumerate(run_nos):
            sim = lam_win.sims[run - 1]
            times, grads = sim.read_gradients()
            dgs = [grad * lam_win.lam_val_weight for grad in grads]
            # Convert times and dgs to arrays of length 100. Do this with block averaging
            # so that we don't lose any information
            times_resized = _np.linspace(0, times[-1], 100)
            dgs_resized = _np.zeros(100)
            # Average the gradients over 100 evenly-sized blocks. We have enough data so
            # that small issues with indexing should not cause problems
            indices = _np.array(list(round(x) for x in _np.linspace(0, len(dgs), 101)))
            for j in range(100):
                dgs_resized[j] = _np.mean(dgs[indices[j] : indices[j + 1]])
            overall_dgs[i] += dgs_resized
            overall_times[i] += times_resized
    # Check that we have the same total times for each run
    if not all(
        [_np.isclose(overall_times[i, -1], overall_times[0, -1]) for i in range(n_runs)]
    ):
        raise ValueError(
            "Total simulation times are not the same for all runs. Please ensure that "
            "the total simulation times are the same for all runs."
        )

    # Block average the gradients and check that the absolute mean gradient of the gradient
    # is significantly below some cut-off. This cut-off should be related to the threshold set
    # in the optimal efficiency loop
    equilibrated = False
    fractional_equil_time = None
    equil_time = None
    for percentage_discard in [0, 10, 30, 60]:
        # Discard the first percentage_discard % of the data
        overall_dgs_discarded = overall_dgs[:, percentage_discard:]
        overall_times_discarded = overall_times[:, percentage_discard:]
        # Average the first and last halves of the data and calculate the gradient
        middle_index = round(overall_dgs_discarded.shape[1] / 2)
        # Grad below in kcal mol^-1 ns^-1
        overall_grads_dg = _np.array(
            overall_dgs_discarded[:, middle_index:].mean(axis=1)
            - overall_dgs_discarded[:, :middle_index].mean(axis=1)
        ) / (overall_times_discarded[0][middle_index] - overall_times_discarded[0][0])
        # Calculate the mean gradient and 95 % confidence intervals
        overall_grad_dg = _np.mean(overall_grads_dg)
        conf_int = _stats.t.interval(
            0.95,
            len(overall_grads_dg) - 1,
            loc=_np.mean(overall_grads_dg),
            scale=_stats.sem(overall_grads_dg),
        )
        # Check that the gradient is not significantly different from zero
        if conf_int[0] < 0 and 0 < conf_int[1]:
            equilibrated = True
            fractional_equil_time = percentage_discard / 100
            equil_time = (
                _np.sum(
                    [
                        lam_win.get_total_simtime(run_nos=run_nos)
                        for lam_win in lambda_windows
                    ]
                )
                * fractional_equil_time
            )
            break

    # Directly set the _equilibrated and _equil_time attributes of the lambda windows
    if equilibrated:
        for lam_win in lambda_windows:
            lam_win._equilibrated = True
            # Equilibration time is per-simulation
            lam_win._equil_time = (
                fractional_equil_time
                * lam_win.get_total_simtime(run_nos=run_nos)
                / lam_win.ensemble_size
            )  # ns

    # Write out data
    with open(f"{output_dir}/check_equil_multiwindow.txt", "w") as ofile:
        ofile.write(f"Equilibrated: {equilibrated}\n")
        ofile.write(f"Overall gradient {overall_grad_dg} +/- {conf_int[1] - overall_grad_dg} kcal mol^-1 ns^-1\n")  # type: ignore
        ofile.write(f"Fractional equilibration time: {fractional_equil_time} \n")
        ofile.write(f"Equilibration time: {equil_time} ns\n")
        ofile.write(f"Run numbers: {run_nos}\n")

    # Create plots of the overall gradients
    _general_plot(
        x_vals=overall_times[0],
        y_vals=overall_dgs,
        x_label="Total Simulation Time / ns",
        y_label=r"$\Delta G$ / kcal mol$^{-1}$",
        outfile=f"{output_dir}/check_equil_multiwindow.png",
        run_nos=run_nos,
    )

    return equilibrated, fractional_equil_time


def check_equil_multiwindow_kpss(
    lambda_windows: _List["LamWindow"],
    output_dir: str,
    run_nos: _Optional[_List[int]] = None,
) -> _Tuple[bool, _Optional[float]]:  # type: ignore
    """
    Check if a set of lambda windows are equilibrated based on the KPSS test for stationarity.
    The overall timeseries of dG against simulation time is obtained by block averaging the free
    energy changes for each window over 100 blocks, then combining. The KPSS test is applied to
    this overall timeseries to check for equilibration. This is repeated discarding 0, 10, 30 and
    50 % of the data. The first time p > 0.05 is found, the simulation is considered equilibrated.
    The fractional equilibration time is then multiplied by the total simulation time for each
    simulation to obtain the per simulation equilibration times. We should maybe be correcting for
    multiple testing.

    Parameters
    ----------
    lambda_windows : List[LamWindow]
        List of lambda windows to check for equilibration.
    output_dir : str
        Directory to write output files to.
    run_nos : List[int], Optional, default: None
        The run numbers to use. If None, all runs will be used.

    Returns
    -------
    equilibrated : bool
        True if the set of lambda windows is equilibrated, False otherwise.
    fractional_equil_time : float
        Time taken to equilibrate, as a fraction of the total simulation time.
    """
    run_nos = lambda_windows[0]._get_valid_run_nos(run_nos)

    if any([not lam_win.lam_val_weight for lam_win in lambda_windows]):
        raise ValueError(
            "Lambda value weight not set for all windows. Please set the lambda value weight to "
            "use check_equil_multiwindow for equilibration detection."
        )

    # Combine all gradients to get the change in free energy with increasing simulation time.
    # Do this so that the total simulation time for each window is spread evenly over the total
    # simulation time for the whole calculation
    n_runs = len(run_nos)
    overall_dgs = _np.zeros(
        [n_runs, 100]
    )  # One point for each % of the total simulation time
    overall_times = _np.zeros([n_runs, 100])
    for lam_win in lambda_windows:
        for i, run_no in enumerate(run_nos):
            sim = lam_win.sims[run_no - 1]
            times, grads = sim.read_gradients()
            dgs = [grad * lam_win.lam_val_weight for grad in grads]
            # Convert times and dgs to arrays of length 100. Do this with block averaging
            # so that we don't lose any information
            times_resized = _np.linspace(0, times[-1], 100)
            dgs_resized = _np.zeros(100)
            # Average the gradients over 100 evenly-sized blocks. We have enough data so
            # that small issues with indexing should not cause problems
            indices = _np.array(list(round(x) for x in _np.linspace(0, len(dgs), 101)))
            for j in range(100):
                dgs_resized[j] = _np.mean(dgs[indices[j] : indices[j + 1]])
            overall_dgs[i] += dgs_resized
            overall_times[i] += times_resized
    # Check that we have the same total times for each run
    if not all(
        [_np.isclose(overall_times[i, -1], overall_times[0, -1]) for i in range(n_runs)]
    ):
        raise ValueError(
            "Total simulation times are not the same for all runs. Please ensure that "
            "the total simulation times are the same for all runs."
        )

    # Block average the gradients and check that the absolute mean gradient of the gradient
    # is significantly below some cut-off. This cut-off should be related to the threshold set
    # in the optimal efficiency loop
    equilibrated = False
    fractional_equil_time = None
    equil_time = None
    for percentage_discard in [0, 10, 30, 50]:
        # Discard the first percentage_discard % of the data
        overall_dgs_discarded = overall_dgs.mean(axis=0)[percentage_discard:]
        # KPSS test
        print(overall_dgs_discarded)
        _, p_value, *_ = _kpss(overall_dgs_discarded, regression="c", nlags="auto")
        # Decide if we have equilibrated based on the p value
        if p_value > 0.05:
            equilibrated = True
            fractional_equil_time = percentage_discard / 100
            equil_time = (
                _np.sum(
                    [
                        lam_win.get_tot_simtime(run_nos=run_nos)
                        for lam_win in lambda_windows
                    ]
                )
                * fractional_equil_time
            )
            break

    # Directly set the _equilibrated and _equil_time attributes of the lambda windows
    if equilibrated:
        for lam_win in lambda_windows:
            lam_win._equilibrated = True
            # Equilibration time is per-simulation
            lam_win._equil_time = (
                fractional_equil_time
                * lam_win.get_tot_simtime(run_nos=run_nos)
                / lam_win.ensemble_size
            )  # ns

    # Write out data
    with open(f"{output_dir}/check_equil_multiwindow_kpss.txt", "w") as ofile:
        ofile.write(f"Equilibrated: {equilibrated}\n")
        ofile.write(f"p value: {p_value}\n")
        ofile.write(f"Fractional equilibration time: {fractional_equil_time} \n")
        ofile.write(f"Equilibration time: {equil_time} ns\n")
        ofile.write(f"Run numbers: {run_nos}\n")

    # Create plots of the overall gradients
    _general_plot(
        x_vals=overall_times[0],
        y_vals=overall_dgs,
        x_label="Total Simulation Time / ns",
        y_label=r"$\Delta G$ / kcal mol$^{-1}$",
        outfile=f"{output_dir}/check_equil_multiwindow_kpss.png",
        vline_val=equil_time,
        run_nos=run_nos,
    )

    return equilibrated, fractional_equil_time


def dummy_check_equil_multiwindow(
    lam_win: "LamWindow", run_nos: _Optional[_List[int]] = None
) -> _Tuple[bool, _Optional[float]]:  # type: ignore
    """
    Becuse "check_equil_multiwindow" checks multiple windows at once and sets the _equilibrated
    and _equil_time attributes of the lambda windows, but EnsEquil was written based on per-window
    checks, we need a dummy function which just reads the attributes of the lambda window and
    assumes that they have already been set by "check_equil_multiwindow".

    Parameters
    ----------
    lam_win : LamWindow
        Lambda window to check for equilibration (ignored in practice)
    run_nos : List[int], Optional, default: None
        Ignored in practice, but included for consistency.

    Returns
    -------
    equilibrated : bool
        True if the simulation is equilibrated, False otherwise.
    equil_time : float
        Time taken to equilibrate per simulation, in ns.
    """
    return lam_win._equilibrated, lam_win._equil_time


def check_equil_shrinking_block_gradient(
    lam_win: "LamWindow", run_nos: _Optional[_List[int]] = None
) -> _Tuple[bool, _Optional[float]]:  # type: ignore
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
    run_nos : List[int], Optional, default: None
        The run numbers to use. If None, all runs will be used.

    Returns
    -------
    equilibrated : bool
        True if the simulation is equilibrated, False otherwise.
    equil_time : float
        Time taken to equilibrate per simulation, in ns.
    """
    run_nos = lam_win._get_valid_run_nos(run_nos)

    # Get the gradient threshold and complain if it does not exist
    if not lam_win.gradient_threshold:
        raise ValueError(
            "Gradient threshold not set. Please set the gradient threshold to "
            "use check_equil_block_gradient for equilibration detection."
        )

    if not lam_win.lam_val_weight:
        raise ValueError(
            "Lambda value weight not set. Please set the lambda value weight to "
            "use check_equil_shrinking_block_gradient for equilibration detection."
        )

    # Scale the gradient threshold according to how each window contributes to the
    # total uncertainty, as given by TI and the trapezoidal rule.
    gradient_threshold = lam_win.gradient_threshold / lam_win.lam_val_weight

    # Conversion between time and gradient indices.
    time_to_ind = 1 / (lam_win.sims[0].timestep * lam_win.sims[0].nrg_freq)

    # For each fractional block size, check for equilibration
    frac_block_sizes = [
        1 / 2,
        1 / 3,
        1 / 4,
        1 / 5,
    ]  # Fraction of the total simulation time
    equilibrated = False
    equil_time = None
    equil_idx_block_size = None  # Int
    equil_block_size = None  # Frac
    equil_d_dh_dls = None
    largest_block_dh_dls = []
    largest_idx_block_size = None
    for block_size_ind, frac_block_size in enumerate(frac_block_sizes):
        idx_block_size = int(
            (lam_win.get_tot_simtime(run_nos=run_nos) / lam_win.ensemble_size)
            * frac_block_size
            * time_to_ind
        )

        # Block average the dh/dl data
        d_dh_dls = []
        dh_dls = []
        times, _ = lam_win.sims[0].read_gradients()

        if block_size_ind == 1:
            largest_idx_block_size = idx_block_size

        for run_no in run_nos:
            sim = lam_win.sims[run_no - 1]
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
                    d_dh_dl[i] = (
                        rolling_av_dh_dl[i] - rolling_av_dh_dl[i - idx_block_size]
                    ) / lam_win.block_size  # Gradient of dh/dl in kcal mol-1 ns-1
            d_dh_dls.append(d_dh_dl)

        # Store the d_dh_dl with the largest block size
        if block_size_ind == 1:
            largest_block_dh_dls = d_dh_dls

        # Calculate the gradient of the block averaged dh/dl with standard deviation over the repeats
        mean_d_dh_dl = _np.mean(d_dh_dls, axis=0)
        ci_d_dh_dl = (
            _stats.t.interval(
                0.95, len(d_dh_dls) - 1, mean_d_dh_dl, scale=_stats.sem(d_dh_dls)
            )[1]
            - mean_d_dh_dl
        )

        # Check if the mean gradients has been simultaneously significantly below and above the threshold
        # at any point, making sure to exclude the initial nans
        for i, (mean_grad, ci) in enumerate(
            zip(mean_d_dh_dl[2 * idx_block_size :], ci_d_dh_dl[2 * idx_block_size :])
        ):
            if (
                mean_grad + ci < gradient_threshold
                and mean_grad - ci > -gradient_threshold
            ):
                equil_time = times[i]
                equilibrated = True
                equil_d_dh_dls = d_dh_dls
                equil_block_size = (
                    frac_block_size
                    * lam_win.get_tot_simtime(run_nos=run_nos)
                    / lam_win.ensemble_size
                )
                equil_idx_block_size = idx_block_size
                break

    # Write out data
    with open(
        f"{lam_win.output_dir}/equilibration_shrinking_block_gradient.txt", "w"
    ) as ofile:
        ofile.write(f"Equilibrated: {equilibrated}\n")
        ofile.write(f"Equilibration time: {equil_time} ns\n")
        ofile.write(f"Successful block size (ns): {equil_block_size}\n")
        ofile.write(f"Gradient threshold: {gradient_threshold}\n")
        ofile.write(f"Run numbers: {run_nos}\n")

    # Save plots of dh/dl and d_dh/dl

    idx_block_size = (
        equil_idx_block_size
        if equil_idx_block_size is not None
        else largest_idx_block_size
    )
    d_dh_dls = equil_d_dh_dls if equil_block_size is not None else largest_block_dh_dls
    _general_plot(
        x_vals=times,  # type: ignore
        # Use idx_block_size, the smallest block size tried, to plot the data
        y_vals=_np.array([lam_win._get_rolling_average(dh_dl, idx_block_size) for dh_dl in dh_dls]),  # type: ignore
        x_label="Simulation Time per Window per Run / ns",
        y_label=r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$",
        outfile=f"{lam_win.output_dir}/dhdl_block_gradient",
        # Shift the equilibration time by 2 * block size to account for the
        # delay in the block average calculation.
        vline_val=equil_time + 1 * equil_block_size
        if equil_block_size is not None
        else None,
        run_nos=run_nos,
    )  # type: ignore

    # This will use the latest dh/dls generated with the final block size above
    _general_plot(
        x_vals=times,  # type: ignore
        # The d_dh_dls are already block averaged with the final block size above
        y_vals=_np.array(d_dh_dls),  # type: ignore
        x_label="Simulation Time per Window per Run / ns",
        y_label=r"$\frac{\partial}{\partial t}\frac{\partial H}{\partial \lambda} \mathrm{~/~ kcal~} \mathrm{mol}^{-1} \mathrm{ns}^{-1} $",
        outfile=f"{lam_win.output_dir}/ddhdl_block_gradient",
        vline_val=equil_time + 2 * equil_block_size if equil_block_size is not None else None,  # type: ignore
        hline_val=0,
        run_nos=run_nos,
    )

    return equilibrated, equil_time


def check_equil_chodera(lam_win: "LamWindow", run_nos: _Optional[_List[int]] = None) -> _Tuple[bool, _Optional[float]]:  # type: ignore
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
    run_nos : List[int], Optional, default: None
        The run numbers to use. If None, all runs will be used.

    Returns
    -------
    equilibrated : bool
        True if the simulation is equilibrated, False otherwise.
    equil_time : float
        Time taken to equilibrate, in ns.
    """
    run_nos = lam_win._get_valid_run_nos(run_nos)

    # Conversion between time and gradient indices.
    time_to_ind = 1 / (lam_win.sims[0].timestep * lam_win.sims[0].nrg_freq)
    idx_block_size = int(lam_win.block_size * time_to_ind)

    # Read dh/dl data from all simulations
    dh_dls = []
    times, _ = lam_win.sims[0].read_gradients()  # Times should be the same for all sims
    equilibrated = False
    equil_time = None

    for run_no in run_nos:
        sim = lam_win.sims[run_no - 1]
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
        ofile.write(f"Run numbers: {run_nos}\n")

    # Save plots of dh/dl and d_dh/dl
    # Use rolling average to smooth out the data
    rolling_av_time = 0.05  # ns
    rolling_av_block_size = int(rolling_av_time * time_to_ind)  # ns
    v_line_x = None if equil_time is None else equil_time + rolling_av_time
    _general_plot(
        x_vals=times,
        y_vals=_np.array(
            [
                lam_win._get_rolling_average(dh_dl, rolling_av_block_size)
                for dh_dl in dh_dls
            ]
        ),
        x_label="Simulation Time per Window per Run / ns",
        y_label=r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$",
        outfile=f"{lam_win.output_dir}/dhdl_chodera",
        # Shift the equilibration time by block size to account for the
        # delay in the rolling average calculation.
        vline_val=v_line_x,
        run_nos=run_nos,
    )

    return equilibrated, equil_time
