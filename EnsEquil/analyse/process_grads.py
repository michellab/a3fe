"""Functionality to process the gradient data."""

import numpy as _np
from typing import List as _List, Tuple as _Tuple, Optional as _Optional, Dict as _Dict, Union as _Union

from .autocorrelation import get_statistical_inefficiency as _get_statistical_inefficiency

class GradientData():
    """A class to store and process gradient data."""

    def __init__(self, lam_winds: _List["LamWindow"], equilibrated: bool, )-> None: # type: ignore
        """ 
        Calculate the gradients, means, and variances of the gradients for each lambda
        window of a list of LamWindows.

        Parameters
        ----------
        lam_winds : List[LamWindow]
            List of lambda windows.
        equilibrated : bool
            If True, only equilibrated data is used.
        """
        self.equilibrated = equilibrated

        # Get mean and variance of gradients, including both intra-run and inter-run components
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
            # Record the lambda value
            lam_vals.append(lam.lam)

            # Get all gradients and statistical inefficiencies
            gradients_wind = []
            means_intra = []
            stat_ineffs_wind = []
            gradients_subsampled_wind = []
            vars_intra = []
            squared_sems_intra = []
            # Get intra-run quantities
            for sim in lam.sims:
                # Get the gradients, mean, and statistical inefficiencies
                _, gradients = sim.read_gradients(equilibrated_only=equilibrated)
                stat_ineff = _get_statistical_inefficiency(gradients)
                mean = _np.mean(gradients)
                # Subsample the gradients to remove autocorrelation
                subsampled_grads = gradients[::int(stat_ineff)]
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
            squared_sem_intra = _np.mean(squared_sems_intra) / len(lam.sims) 
            stat_ineff = _np.mean(stat_ineffs_wind)

            # Get inter-run quantities
            squared_sem_inter = _np.var(means_intra) / len(lam.sims)
            mean_overall = _np.mean(means_intra)

            # Store the final results, converting to arrays for consistency.
            tot_sem = _np.sqrt(squared_sem_inter + squared_sem_intra)
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
        stat_ineffs_all_winds = _np.array(stat_ineffs_all_winds) * lam_winds[0].sims[0].timestep # Timestep should be same for all sims

        # Get the times
        if equilibrated:
            start_times = _np.array([win._equil_time for win in lam_winds])
        else:
            start_times = _np.array([0 for win in lam_winds])
        end_times = _np.array([win.sims[0].tot_simtime for win in lam_winds]) # All sims at given lam run for same time
        times = _np.array([_np.linspace(start, end, len(gradients[0]) + 1)[1:] for start, end, gradients in zip(start_times, end_times, gradients_all_winds)])

        # Save the calculated attributes
        self.lam_vals = lam_vals
        self.gradients = gradients_all_winds
        self.subsampled_gradients = gradients_subsampled_all_winds
        self.times = times
        self.means = means_all_winds
        self.sems_overall = sems_tot_all_winds
        self.sems_intra = sems_intra_all_winds
        self.sems_inter = sems_inter_all_winds
        self.vars_intra = vars_intra_all_winds
        self.stat_ineffs = stat_ineffs_all_winds

    @property
    def smoothened_sems(self)-> _np.ndarray:
        """Calculate the standard error of the mean of the gradients, using a block
        average over 3 points to smooth the data."""
        smoothened_sems = []
        max_ind = len(self.sems_overall) - 1
        for i, sem in enumerate(self.sems_overall):
            # Calculate the block average for each point
            if i == 0:
                sem_smooth = (sem + self.sems_overall[i+1]) /2
            elif i == max_ind:
                sem_smooth = (sem + self.sems_overall[i-1]) /2
            else:
                sem_smooth = (sem + self.sems_overall[i+1] + self.sems_overall[i-1]) / 3 
            smoothened_sems.append(sem_smooth)
            
        smoothened_sems = _np.array(smoothened_sems)
        self._smoothened_sems = smoothened_sems
        return smoothened_sems

    @property
    def integrated_sems(self)-> _np.ndarray:
        """Calculate the integrated standard error of the mean of the gradients
        as a function of lambda, using the trapezoidal rule."""

        integrated_sems = []
        x_vals = self.lam_vals
        # No need to use smoothened SEMs as the trapezoidal rule results in some smoothing
        # between neighbours
        y_vals = self.sems_overall
        n_vals = len(x_vals)

        for i in range(n_vals):
            # No need to worry about indexing off the end of the array with numpy
            # Note that _np.trapz(y_vals[:1], x_vals[:1]) gives 0, as required
            integrated_sems.append(_np.trapz(y_vals[:i+1], x_vals[:i+1]))
        
        integrated_sems = _np.array(integrated_sems)
        self._integrated_sems = integrated_sems
        return integrated_sems
    
    def calculate_optimal_lam_vals(self, delta_sem: _Optional[float] = None, 
                                   n_lam_vals: _Optional[int] = None)-> _np.ndarray:
        """
        Calculate the optimal lambda values for a given number of lambda values
        to sample, using the integrated standard error of the mean of the gradients
        as a function of lambda, using the trapezoidal rule.

        Parameters
        ----------
        delta_sem : float, optional
            The desired integrated standard error of the mean of the gradients
            between each lambda value, in kcal mol-1. If not provided, the number of lambda
            windows must be provided with n_lam_vals.
        n_lam_vals : int, optional
            The number of lambda values to sample. If not provided, the desired
            integrated standard error of the mean of the gradients between each
            lambda value must be provided with delta_sem.

        Returns
        -------
        optimal_lam_vals : np.ndarray
            The optimal lambda values to sample.
        """
        if delta_sem is None and n_lam_vals is None:
            raise ValueError("Either delta_sem or n_lam_vals must be provided.")
        elif delta_sem is not None and n_lam_vals is not None:
            raise ValueError("Only one of delta_sem or n_lam_vals can be provided.")

        # Calculate the integrated standard error of the mean of the gradients
        # as a function of lambda, using the trapezoidal rule.
        integrated_sems = self.integrated_sems
        total_sem = integrated_sems[-1]

        # If the number of lambda values is not provided, calculate it from the
        # desired integrated standard error of the mean between lam vals
        if n_lam_vals is None:
            n_lam_vals = int(total_sem / delta_sem) + 1

        # Convert the number of lambda values to an array of SEM values
        requested_sem_vals = _np.linspace(0, total_sem, n_lam_vals)

        # For each desired SEM value, map it to a lambda value
        optimal_lam_vals = []
        for requested_sem in requested_sem_vals:
            optimal_lam_val = _np.interp(requested_sem, integrated_sems, self.lam_vals)
            optimal_lam_val = _np.round(optimal_lam_val, 3)
            optimal_lam_vals.append(optimal_lam_val)

        optimal_lam_vals = _np.array(optimal_lam_vals)
        self._optimal_lam_vals = optimal_lam_vals
        return optimal_lam_vals
