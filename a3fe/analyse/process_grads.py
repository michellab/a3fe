"""Aknowledgement: "process_grads.py" is copied almost exactly from
"process_grads.py" here:
https://github.com/michellab/a3fe/blob/main/a3fe/analyse/process_grads.py
The license and original authorship are preserved below:

Copyright (c) 2023, Finlay Clark

LICENSE
xxx
"""

from __future__ import annotations

__all__ = ["GradientData"]

from multiprocessing import get_context as _get_context
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np
from scipy.constants import gas_constant as _R

from .autocorrelation import (
    get_statistical_inefficiency as _get_statistical_inefficiency,
)
from .mbar import collect_mbar_slurm as _collect_mbar_slurm
from .mbar import run_mbar as _run_mbar
from .mbar import submit_mbar_slurm as _submit_mbar_slurm


class GradientData:
    """A class to store and process gradient data."""

    def __init__(
        self,
        lam_winds: _List["LamWindow"],  # noqa: F821
        equilibrated: bool,
        run_nos: _Optional[_List[int]] = None,
    ) -> None:  # type: ignore
        """
        Calculate the gradients, means, and variances of the gradients for each lambda
        window of a list of LamWindows.

        Parameters
        ----------
        lam_winds : List[LamWindow]
            List of lambda windows.
        equilibrated : bool
            If True, only equilibrated data is used.
        run_nos : List[int], Optional, default: None
            The run numbers to use. If None, all runs will be used.
        """
        self.equilibrated = equilibrated

        # Get the overall timeseries
        self.overall_dgs, self.overall_times = get_time_series_multiwindow(
            lam_winds, equilibrated=equilibrated, run_nos=run_nos
        )

        # Store the relative cost of the windows
        self.relative_simulation_cost = lam_winds[0].relative_simulation_cost
        # Make sure that this is the same for all windows
        if not all(
            [
                win.relative_simulation_cost == self.relative_simulation_cost
                for win in lam_winds
            ]
        ):
            raise ValueError(
                "Relative simulation cost is not the same for all windows. Please ensure that "
                "the relative simulation cost is the same for all windows."
            )

        # Get mean and variance of gradients, including both intra-run and inter-run components
        # Note that sems/ vars generally refer to the sems/ vars of the gradients, and not the
        # free energy changes.
        lam_vals = []
        gradients_all_winds = []
        gradients_subsampled_all_winds = []
        means_all_winds = []
        sems_tot_all_winds = []
        sems_intra_all_winds = []
        sems_inter_all_winds = []
        vars_intra_all_winds = []
        stat_ineffs_all_winds = []

        for lam in lam_winds:
            # Record the lambda value and get sensible run numbers
            lam_vals.append(lam.lam)
            run_nos: _List[int] = lam._get_valid_run_nos(run_nos=run_nos)

            # Get all gradients and statistical inefficiencies
            gradients_wind = []
            means_intra = []
            stat_ineffs_wind = []
            gradients_subsampled_wind = []
            vars_intra = []
            squared_sems_intra = []
            # Get intra-run quantities
            for run_no in run_nos:
                sim = lam.sims[run_no - 1]
                # Get the gradients, mean, and statistical inefficiencies
                _, gradients = sim.read_gradients(equilibrated_only=equilibrated)
                stat_ineff = _get_statistical_inefficiency(gradients)
                mean = _np.mean(gradients)
                # Subsample the gradients to remove autocorrelation
                subsampled_grads = gradients[:: int(stat_ineff)]
                # Get the variance and squared SEM of the gradients
                var = _np.var(subsampled_grads)
                squared_sem = var / len(subsampled_grads)
                # Store the results
                gradients_wind.append(gradients)
                means_intra.append(mean)
                stat_ineffs_wind.append(stat_ineff)
                gradients_subsampled_wind.append(subsampled_grads)
                vars_intra.append(var)
                squared_sems_intra.append(squared_sem)

            # Get overall intra-run quantities
            var_intra = _np.mean(vars_intra)
            squared_sem_intra = _np.mean(squared_sems_intra) / len(run_nos)
            stat_ineff = _np.mean(stat_ineffs_wind)

            # Get inter-run quantities
            squared_sem_inter = _np.var(means_intra) / len(run_nos)
            mean_overall = _np.mean(means_intra)

            # Store the final results, converting to arrays for consistency.
            tot_sem = _np.sqrt(
                squared_sem_inter + squared_sem_intra
            )  # This isn't really a meaningful quantity
            sem_intra = _np.sqrt(squared_sem_intra)
            sem_inter = _np.sqrt(squared_sem_inter)
            gradients_all_winds.append(_np.array(gradients_wind))
            gradients_subsampled_all_winds.append(gradients_subsampled_wind)
            means_all_winds.append(mean_overall)
            sems_tot_all_winds.append(tot_sem)
            sems_intra_all_winds.append(sem_intra)
            sems_inter_all_winds.append(sem_inter)
            vars_intra_all_winds.append(var_intra)
            stat_ineffs_all_winds.append(stat_ineff)

        # Get the statistical inefficiencies in units of simulation time
        stat_ineffs_all_winds = (
            _np.array(stat_ineffs_all_winds) * lam_winds[0].sims[0].timestep
        )  # Timestep should be same for all sims

        # Get the SEMs of the free energy changes from the inter-run SEMs of the gradients
        lam_weights = _np.array([lam.lam_val_weight for lam in lam_winds])
        sems_inter_delta_g = _np.array(sems_inter_all_winds) * lam_weights

        # Get the times
        if equilibrated:
            start_times = _np.array([win._equil_time for win in lam_winds])
        else:
            start_times = _np.array([0 for win in lam_winds])
        end_times = _np.array(
            [win.sims[0].tot_simtime for win in lam_winds]
        )  # All sims at given lam run for same time
        times = [
            _np.linspace(start, end, len(gradients[0]) + 1)[1:]
            for start, end, gradients in zip(
                start_times, end_times, gradients_all_winds
            )
        ]

        # Get the total sampling time per window
        sampling_times = end_times - start_times

        # Save the calculated attributes
        self.n_lam = len(lam_vals)
        self.lam_vals = lam_vals
        self.gradients = gradients_all_winds
        self.subsampled_gradients = gradients_subsampled_all_winds
        self.times = times
        self.sampling_times = sampling_times
        self.total_times = end_times
        self.means = means_all_winds
        self.sems_overall = sems_tot_all_winds
        self.sems_intra = sems_intra_all_winds
        self.sems_inter = sems_inter_all_winds
        self.vars_intra = vars_intra_all_winds
        self.stat_ineffs = stat_ineffs_all_winds
        self.sems_inter_delta_g = sems_inter_delta_g
        self.runtime_constant = lam_winds[0].runtime_constant  # Assume all the same
        self.run_nos = run_nos
        self.lam_val_weights = lam_weights

    def get_time_normalised_sems(
        self, origin: str = "inter", smoothen: bool = True
    ) -> _np.ndarray:
        """
        Return the standardised standard error of the mean of the gradients, optionally
        smoothened by a block average over 3 points.

        Parameters
        ----------
        origin: str, optional, default="inter"
            Can be "inter", "intra", or "inter_delta_g". Whether to use the
            inter-run or intra-run standard error of the mean, or the
            inter-run standard error of the mean of the free energy changes,
            respectively.
        smoothen: bool, optional, default=True
            Whether to smoothen the standard error of the mean by a block average
            over 3 points.

        Returns
        -------
        sems: np.ndarray
            The standardised standard error of the mean of the gradients, in kcal mol^-1 ns^(1/2).
        """
        # Check options are valid
        if origin not in ["inter", "intra", "inter_delta_g"]:
            raise ValueError(
                "origin must be either 'inter' or 'intra' or 'inter_delta_g'"
            )

        origins = {
            "inter": self.sems_inter,
            "intra": self.sems_intra,
            "inter_delta_g": self.sems_inter_delta_g,
        }
        sems = origins[origin]

        # Standardise the SEMs according to the total simulation time. Because total times are
        # stored on a per-run basis, multiply them by the total number of runs.
        total_times_all_runs = _np.array(self.total_times) * len(self.run_nos)  # type: ignore
        sems *= _np.sqrt(total_times_all_runs)  # type: ignore

        if not smoothen:
            return sems  # type: ignore

        # Smoothen the standard error of the mean by a block average over 3 points
        smoothened_sems = []
        max_ind = len(sems) - 1  # type: ignore
        for i, sem in enumerate(sems):  # type: ignore
            # Calculate the block average for each point
            if i == 0:
                sem_smooth = (sem + sems[i + 1]) / 2
            elif i == max_ind:
                sem_smooth = (sem + sems[i - 1]) / 2
            else:
                sem_smooth = (sem + sems[i + 1] + sems[i - 1]) / 3
            smoothened_sems.append(sem_smooth)

        smoothened_sems = _np.array(smoothened_sems)
        self._smoothened_sems = smoothened_sems
        return smoothened_sems

    def get_integrated_error(
        self, er_type: str = "sem", origin: str = "inter", smoothen: bool = True
    ) -> _np.ndarray:
        """
        Calculate the integrated standard error of the mean or root variance of the gradients
        as a function of lambda, using the trapezoidal rule.

        Parameters
        ----------
        er_type: str, optional, default="sem"
            Whether to integrate the standard error of the mean ("sem") or root
            variance of the gradients ("root_var").
        origin: str, optional, default="inter"
            The origin of the SEM to integrate - this is ignore if er_type == "root_var".
            Can be either 'inter' or 'intra' for inter-run and intra-run SEMs respectively.
        smoothen: bool, optional, default=True
            Whether to use the smoothened SEMs or not. If False, the raw SEMs
            are used. If er_type == "root_var", this option is ignored.

        Returns
        -------
        integrated_errors: np.ndarray
            The integrated SEMs as a function of lambda, in kcal mol^-1 ns^(1/2).
        """
        # Check options are valid
        if er_type not in ["sem", "root_var"]:
            raise ValueError("er_type must be either 'sem' or 'root_var'")
        if origin not in ["inter", "intra"]:
            raise ValueError("origin must be either 'inter' or 'intra'")

        integrated_errors = []
        x_vals = self.lam_vals
        # Note that the trapezoidal rule results in some smoothing between neighbours
        # even without smoothening
        if er_type == "sem":
            y_vals = self.get_time_normalised_sems(origin=origin, smoothen=smoothen)
        elif er_type == "root_var":
            y_vals = _np.sqrt(self.vars_intra)
        n_vals = len(x_vals)

        for i in range(n_vals):
            # No need to worry about indexing off the end of the array with numpy
            # Note that _np.trapz(y_vals[:1], x_vals[:1]) gives 0, as required
            integrated_errors.append(_np.trapz(y_vals[: i + 1], x_vals[: i + 1]))  # type: ignore

        integrated_errors = _np.array(integrated_errors)
        self._integrated_sems = integrated_errors
        return integrated_errors

    def calculate_optimal_lam_vals(
        self,
        er_type: str = "sem",
        delta_er: _Optional[float] = None,
        n_lam_vals: _Optional[int] = None,
        sem_origin: str = "inter",
        smoothen_sems: bool = True,
        round_lams: bool = True,
    ) -> _np.ndarray:
        """
        Calculate the optimal lambda values for a given number of lambda values
        to sample, using the integrated standard error of the mean of the gradients
        or root variance as a function of lambda, using the trapezoidal rule.

        Parameters
        ----------
        er_type: str, optional, default="sem"
            Whether to integrate the standard error of the mean ("sem") or root
            variance of the gradients ("root_var").
        delta_er : float, optional
            If er_type == "root_var", the desired integrated root variance of the gradients
            between each lambda value, in kcal mol^(-1). If er_type == "sem", the
            desired integrated standard error of the mean of the gradients between each lambda
            value, in kcal mol^(-1) ns^(1/2). If not provided, the number of lambda
            windows must be provided with n_lam_vals.
        n_lam_vals : int, optional
            The number of lambda values to sample. If not provided, delta_er must be provided.
        sem_origin: str, optional, default="inter"
            The origin of the SEM to integrate. Can be either 'inter' or 'intra'
            for inter-run and intra-run SEMs respectively. If er_type == "root_var",
            this is ignored.
        smoothen_sems: bool, optional, default=True
            Whether to use the smoothened SEMs or not. If False, the raw SEMs
            are used. If True, the SEMs are smoothened by a block average over
            3 points. If er_type == "root_var", this is ignored.
        round_lams : bool, optional, default=True
            Whether to round the lambda values to 3 decimal places. This should always
            be True, other than during testing.

        Returns
        -------
        optimal_lam_vals : np.ndarray
            The optimal lambda values to sample.
        """
        if delta_er is None and n_lam_vals is None:
            raise ValueError("Either delta_er or n_lam_vals must be provided.")
        elif delta_er is not None and n_lam_vals is not None:
            raise ValueError("Only one of delta_er or n_lam_vals can be provided.")

        # Calculate the integrated gradient error (of the requested type)
        # as a function of lambda, using the trapezoidal rule.
        integrated_errors = self.get_integrated_error(
            er_type=er_type, origin=sem_origin, smoothen=smoothen_sems
        )

        total_error = integrated_errors[-1]

        # If the number of lambda values is not provided, calculate it from the
        # desired integrated standard error of the mean between lam vals
        if n_lam_vals is None:
            n_lam_vals: int = round(total_error / delta_er) + 1
        # Make sure that we have at least 2 windows
        n_lam_vals = max(n_lam_vals, 2)

        # Convert the number of lambda values to an array of SEM values
        requested_sem_vals = _np.linspace(0, total_error, n_lam_vals)

        # For each desired SEM value, map it to a lambda value
        optimal_lam_vals = []

        for requested_sem in requested_sem_vals:
            optimal_lam_val = _np.interp(
                requested_sem, integrated_errors, self.lam_vals
            )
            ## Note that rounding can result in slight decreases of the
            ## performance compared to the theoretical maximum
            if round_lams:
                optimal_lam_val = _np.round(optimal_lam_val, 3)
            optimal_lam_vals.append(optimal_lam_val)

        optimal_lam_vals = _np.array(optimal_lam_vals)
        self._optimal_lam_vals = optimal_lam_vals
        return optimal_lam_vals

    def get_predicted_overlap_mat(self, temperature: float = 298) -> _np.ndarray:
        """
        Calculate the predicted overlap matrix for the lambda windows
        based on the intra-run variances alone.

        Parameters
        ----------
        temperature: float, optional, default=298
            The temperature in Kelvin.

        Returns
        -------
        predicted_overlap_mat: np.ndarray
            The predicted overlap matrix for the lambda windows.
        """
        # Constants and empty matrix
        beta = (4.184 * 1000) / (_R * temperature)  # in kcal mol^-1
        predicted_overlap_mat = _np.zeros((self.n_lam, self.n_lam))

        # Start with upper off-diagonal
        for col_index in range(1, self.n_lam):
            delta_lam = self.lam_vals[col_index] - self.lam_vals[col_index - 1]
            average_var = (
                self.vars_intra[col_index] + self.vars_intra[col_index - 1]
            ) / 2
            overlap = 1 / (2 + 0.5 * beta**2 * delta_lam**2 * average_var)
            predicted_overlap_mat[col_index - 1, col_index] = overlap

        # Copy upper off-diagonal to lower off-diagonal
        for row_index in range(1, self.n_lam):
            predicted_overlap_mat[row_index, row_index - 1] = predicted_overlap_mat[
                row_index - 1, row_index
            ]

        # Compute the diagonal elements as 1 - current row sum
        for row_index in range(self.n_lam):
            predicted_overlap_mat[row_index, row_index] = (
                1 - predicted_overlap_mat[row_index, :].sum()
            )

        return predicted_overlap_mat

    def get_predicted_improvement_factor(
        self,
        er_type: str = "sem",
        sem_origin: str = "inter",
        smoothen_sems: bool = False,
        initial_lam_vals: _Optional[_List[float]] = None,
    ) -> float:
        """
        Calculate the theoretical improvement factor in the standard deviation (error
        type = "root_var") or standard error of the mean (error type = "sem") of the
        free energy change between the first and last lambda windows, if the windows
        were to be spaced optimally. The improvement factor is defined as the ratio
        of the standard deviation or standard error of the mean of the free energy
        change with the optimal spacing to that with the initial spacing (with equal
        number of lambda windows).

        See: Lundborg, Magnus, Jack Lidmar, and Berk Hess. "On the Path to Optimal Alchemistry."
        The Protein Journal (2023): 1-13.

        Parameters
        ----------
        er_type: str, optional, default="sem"
            Whether to compute the metric based on the standard error of the mean ("sem")
            or standard deviation ("root_var")(which does not account for autocorrelation).
        sem_origin: str, optional, default="inter"
            The origin of the SEM to integrate. Can be either 'inter' or 'intra'
            for inter-run and intra-run SEMs respectively. If er_type == "root_var",
            this is ignored.
        smoothen_sems: bool, optional, default=True
            Whether to use the smoothened SEMs or not. If False, the raw SEMs
            are used. If True, the SEMs are smoothened by a block average over
            3 points. If er_type == "root_var", this is ignored.
        initial_lam_vals: List[float], optional, default=None
            The reference lambda values to compare the theoretically-ideal lambda values
            to (same number of lambda values but with optimal spacing). If None, the current
            lambda values are used.

        Returns
        -------
        predicted_improvement_factor : float
            The theoretical improvement in the standard deviation or
            standard error of the mean of the free energy change if optimal
            spacing were used.
        """
        if initial_lam_vals is None:
            initial_lam_vals = self.lam_vals

        # Get the SEM or root variance of the free energy change
        if er_type == "sem":
            errors = self.get_time_normalised_sems(
                origin=sem_origin, smoothen=smoothen_sems
            )
        elif er_type == "root_var":
            errors = _np.sqrt(self.vars_intra)
        else:
            raise ValueError("er_type must be either 'sem' or 'root_var'")

        # Calculate the optimal variance, e.g. thermodynamic length squared
        v_opt = (_np.trapz(errors, self.lam_vals) ** 2) / len(initial_lam_vals)

        # Assign boundaries for each lambda value
        lam_boundaries = {}
        for i, lam_val in enumerate(initial_lam_vals):
            if i == 0:
                lam_boundaries[lam_val] = [0, lam_val + 0.5 * initial_lam_vals[i + 1]]
            elif i == len(initial_lam_vals) - 1:
                lam_boundaries[lam_val] = [
                    lam_val - 0.5 * (lam_val - initial_lam_vals[i - 1]),
                    1,
                ]
            else:
                lam_boundaries[lam_val] = [
                    lam_val - 0.5 * (lam_val - initial_lam_vals[i - 1]),
                    lam_val + 0.5 * (initial_lam_vals[i + 1] - lam_val),
                ]

        # Now, get the variances by assigning the integrated errors from the
        # measured variances to the initial lambda values
        integrated_errors = self.get_integrated_error(
            er_type=er_type, origin=sem_origin, smoothen=smoothen_sems
        )
        weighted_initial_errors = []
        for lam_val in initial_lam_vals:
            boundaries = lam_boundaries[lam_val]
            initial_error = _np.interp(boundaries[0], self.lam_vals, integrated_errors)
            final_error = _np.interp(boundaries[1], self.lam_vals, integrated_errors)
            weighted_initial_errors.append(final_error - initial_error)

        weighted_initial_errors = _np.array(weighted_initial_errors)
        v_initial = _np.sum(weighted_initial_errors**2)

        # Return improvement factor, which tells us how much the error would be reduced
        return _np.sqrt(v_opt / v_initial)


def get_time_series_multiwindow(
    lambda_windows: _List["LamWindow"],  # noqa: F821
    equilibrated: bool = False,
    run_nos: _Optional[_List[int]] = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Get a combined timeseries of free energy changes against simulation time,
    based on the gradients from each lambda window. The contributions from each
    lambda window are combined so that at a given fraction of total simulation
    time, each lambda window contributes the same fraction of its data. Because
    the simulations at different lambda windows are assumed to have run for
    different amounts of simulation times, the data is combined by block averaging
    each lambda window's data into 100 blocks, then combining these blocks over
    runs.

    Parameters
    ----------
    lambda_windows : List[LamWindow]
        The lambda windows to combine.
    equilibrated : bool, optional, default=False
        Whether or not to discard data before the equilibration time. This
        can only be True if _equilibrated and _equil_time are set for
        all lambda windows.
    run_nos : List[int], optional, default=None
        The run numbers to use for each lambda window. If None, all runs are
        used.
    start_frac : float, optional, default=0.
        The fraction of the total simulation time to start the combined timeseries
        from.
    end_frac : float, optional, default=1
        The fraction of the total simulation time to end the combined timeseries
        at.

    Returns
    -------
    overall_dgs : np.ndarray
        The free energy changes at each increment of total simulation time.
    overall_times : np.ndarray
        The total simulation times (per run).
    """
    # Check that weights are defined for all windows
    if any([not lam_win.lam_val_weight for lam_win in lambda_windows]):
        raise ValueError(
            "Lambda value weight not set for all windows. Please set the lambda value weight to "
            "use get_time_series_multiwindow."
        )

    # Check that equilibration stats have been set for all lambda windows if equilibrated is True
    if equilibrated and any([not lam_win._equilibrated for lam_win in lambda_windows]):
        raise ValueError(
            "_equilibrated must be set for all windows to use get_time_series_multiwindow"
        )

    # Make sure that the required equilibrated simfiles exist if equilibrated is True
    if equilibrated:
        for lam_win in lambda_windows:
            lam_win._write_equilibrated_simfiles()

    # Check the run numbers
    run_nos: _List[int] = lambda_windows[0]._get_valid_run_nos(run_nos)

    # Combine all gradients to get the change in free energy with increasing simulation time.
    # Do this so that the total simulation time for each window is spread evenly over the total
    # simulation time for the whole calculation
    n_runs = len(run_nos)
    overall_dgs = _np.zeros(
        [
            n_runs,
            100,
        ]
    )  # One point for each % of the total simulation time
    overall_times = _np.zeros([n_runs, 100])
    for lam_win in lambda_windows:
        for i, run_no in enumerate(run_nos):  # type: ignore
            sim = lam_win.sims[run_no - 1]
            times, grads = sim.read_gradients()

            dgs = [grad * lam_win.lam_val_weight for grad in grads]
            # Truncate here if necessary
            start_idx = 0 if start_frac is None else round(start_frac * len(dgs))
            end_idx = len(dgs) if end_frac is None else round(end_frac * len(dgs))
            # Make sure we have enough data for our block averaging
            if end_idx - start_idx < 100:
                raise ValueError(
                    "Not enough data to combine windows. Please use a larger fraction of the "
                    "total simulation time or run the simulations for longer."
                )
            times, dgs = times[start_idx:end_idx], dgs[start_idx:end_idx]
            # Convert times and dgs to arrays of length 100. Do this with block averaging
            # so that we don't lose any information
            times_resized = _np.linspace(times[0], times[-1], 100)
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
            [
                _np.isclose(overall_times[i, -1], overall_times[0, -1])
                for i in range(n_runs)
            ]
        ):
            raise ValueError(
                "Total simulation times are not the same for all runs. Please ensure that "
                "the total simulation times are the same for all runs."
            )

        # Check that we didn't get any NaNs
        if _np.isnan(overall_dgs).any():
            raise ValueError(
                "NaNs found in the free energy change. Please check that the simulation "
                "has run correctly."
            )

    return overall_dgs, overall_times


def get_time_series_multiwindow_mbar(
    lambda_windows: _List["LamWindow"],  # noqa: F821
    output_dir: str,
    equilibrated: bool = False,
    run_nos: _Optional[_List[int]] = None,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Get a combined timeseries of free energy changes against simulation time,
    based on the gradients from each lambda window. The contributions from each
    lambda window are combined so that at a given fraction of total simulation
    time, each lambda window contributes the same fraction of its data. Because
    the simulations at different lambda windows are assumed to have run for
    different amounts of simulation times, the data is combined by block averaging
    each lambda window's data into 100 blocks, then combining these blocks over
    runs. This version of the function is based on MBAR, in contrast to the original
    version which is based on TI (faster but noisier).

    Parameters
    ----------
    lambda_windows : List[LamWindow]
        The lambda windows to combine.
    output_dir : str
        The stage output directory.
    equilibrated : bool, optional, default=False
        Whether or not to discard data before the equilibration time. This
        can only be True if _equilibrated and _equil_time are set for
        all lambda windows.
    run_nos : List[int], optional, default=None
        The run numbers to use for each lambda window. If None, all runs are
        used.
    start_frac : float, optional, default=0.
        The fraction of the total simulation time to start the combined timeseries
        from.
    end_frac : float, optional, default=1
        The fraction of the total simulation time to end the combined timeseries
        at.

    Returns
    -------
    overall_dgs : np.ndarray
        The free energy changes at each increment of total simulation time.
    overall_times : np.ndarray
        The total simulation times (per run).
    """
    # Check that equilibration stats have been set for all lambda windows if equilibrated is True
    if equilibrated and not all([lam.equilibrated for lam in lambda_windows]):
        raise ValueError(
            "The equilibration times and statistics have not been set for all lambda "
            "windows in the stage. Please set these before running this function."
        )

    # Make sure that the required equilibrated simfiles exist if equilibrated is True
    if equilibrated:
        for lam_win in lambda_windows:
            lam_win._write_equilibrated_simfiles()

    # Check the run numbers
    run_nos: _List[int] = lambda_windows[0]._get_valid_run_nos(run_nos)

    # Combine all gradients to get the change in free energy with increasing simulation time.
    # Do this so that the total simulation time for each window is spread evenly over the total
    # simulation time for the whole calculation
    n_runs = len(run_nos)
    n_points = 100
    overall_dgs = _np.zeros(
        [
            n_runs,
            n_points,
        ]
    )  # One point for each % of the total simulation time
    overall_times = _np.zeros([n_runs, n_points])
    start_and_end_fracs = [
        (i, i + (end_frac - start_frac) / n_points)
        for i in _np.linspace(start_frac, end_frac, n_points + 1)
    ][:-1]  # Throw away the last point as > 1
    # Round the values to avoid floating point errors
    start_and_end_fracs = [
        (round(x[0], 5), round(x[1], 5)) for x in start_and_end_fracs
    ]
    print('start_and_end_fracs------', start_and_end_fracs)
    # Check whether we should use slurm or not.
    use_slurms = [
        getattr(lam_win, "slurm_equil_detection", True) for lam_win in lambda_windows
    ]
    # Raise an error if they're not all the same.
    if not all(use_slurm == use_slurms[0] for use_slurm in use_slurms):
        raise ValueError(
            "use_slurm is not the same for all lambda windows. Please ensure that "
            "use_slurm is the same for all lambda windows."
        )

    if not use_slurms[0]:
        # Run MBAR in parallel
        with _get_context("spawn").Pool() as pool:
            results = pool.starmap(
                _compute_dg,
                [
                    (run_no, start_frac, end_frac, output_dir, equilibrated)
                    for run_no in run_nos
                    for start_frac, end_frac in start_and_end_fracs
                ],
            )

        # Reshape the results
        for i, run_no in enumerate(run_nos):
            for j, (start_frac, end_frac) in enumerate(start_and_end_fracs):
                overall_dgs[i, j] = results[i * len(start_and_end_fracs) + j]

    else:  # Use slurm
        frac_jobs = []
        results = []
        for start_frac, end_frac in start_and_end_fracs:
            frac_jobs.append(
                _submit_mbar_slurm(
                    output_dir=output_dir,
                    virtual_queue=lambda_windows[0].virtual_queue,
                    run_nos=run_nos,
                    run_somd_dir=lambda_windows[0].input_dir,
                    percentage_end=end_frac * 100,
                    percentage_start=start_frac * 100,
                    subsampling=False,
                    equilibrated=equilibrated,
                )
            )

        for frac_job in frac_jobs:
            jobs, mbar_outfiles, tmp_simfiles = frac_job
            results.append(
                _collect_mbar_slurm(
                    output_dir=output_dir,
                    run_nos=run_nos,
                    jobs=jobs,
                    mbar_out_files=mbar_outfiles,
                    virtual_queue=lambda_windows[0].virtual_queue,
                    tmp_simfiles=tmp_simfiles,
                    delete_outfiles=True,
                )
            )

        # Reshape the results to be consistent with the non-slurm case
        for i, run_no in enumerate(run_nos):
            for j, (start_frac, end_frac) in enumerate(start_and_end_fracs):
                overall_dgs[i, j] = results[j][0][i]

    # Get times per run
    for i, run_no in enumerate(run_nos):
        total_time = sum(
            [lam_win.get_tot_simtime([run_no]) for lam_win in lambda_windows]
        )
        equil_time = (
            sum([lam_win.equil_time for lam_win in lambda_windows])
            if equilibrated
            else 0
        )
        times = [
            (total_time - equil_time) * fracs[0] + equil_time
            for fracs in start_and_end_fracs
        ]
        overall_times[i] = times

    # Check that we have the same total times for each run
    if not all(
        [_np.isclose(overall_times[i, -1], overall_times[0, -1]) for i in range(n_runs)]
    ):
        raise ValueError(
            "Total simulation times are not the same for all runs. Please ensure that "
            "the total simulation times are the same for all runs."
        )

    # Check that we didn't get any NaNs
    if _np.isnan(overall_dgs).any():
        raise ValueError(
            "NaNs found in the free energy change. Please check that the simulation "
            "has run correctly."
        )

    return overall_dgs, overall_times


def _compute_dg(
    run_no: int, start_frac: float, end_frac: float, output_dir: str, equilibrated: bool
) -> float:
    """
    Helper function to compute the free energy change for a single run. Arguments are as
    defined in get_time_series_multiwindow_mbar. Defined at the module level so that it
    can be used with multiprocessing.
    """
    free_energies, *_ = _run_mbar(
        output_dir=output_dir,
        run_nos=[run_no],
        equilibrated=equilibrated,
        percentage_end=end_frac * 100,
        percentage_start=start_frac * 100,
        subsampling=False,
        delete_outfiles=False,
    )
    return free_energies[0]
