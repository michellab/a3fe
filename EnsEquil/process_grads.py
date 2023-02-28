"""Functionality to process the gradient data."""

import numpy as _np
from typing import List as _List, Tuple as _Tuple, Optional as _Optional, Dict as _Dict, Union as _Union

from .calc_corr import get_statistical_inefficiency as _get_statistical_inefficiency

class GradientData():
    """A class to store and process gradient data."""

    def __init__(self, lam_winds: _List["LamWindow"], equilibrated: bool, )-> None:
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
            stat_ineffs_wind = []
            for sim in lam.sims:
                _, gradients = sim.read_gradients(equilibrated_only=equilibrated)
                stat_ineff = _get_statistical_inefficiency(gradients)
                gradients_wind.append(gradients)
                stat_ineffs_wind.append(stat_ineff)

            # Get intra-run quantities
            vars_intra = _np.var(gradients_wind, axis=1)
            means_intra = _np.mean(gradients_wind, axis=1)
            # Convert variances to squared standard errors using the number
            # of uncorrelated samples
            n_uncorr_samples = _np.array([len(gradients) / stat_ineff for gradients, stat_ineff in zip(gradients_wind, stat_ineffs_wind)])
            squared_sems_intra = vars_intra / n_uncorr_samples
            squared_sem_intra = _np.mean(squared_sems_intra) / len(lam.sims) 

            # Get inter-run quantities
            mean_overall = _np.mean(means_intra)
            var_inter = _np.var(means_intra)
            # Assume that each run is uncorrelated to generate the standard error
            squared_sem_inter = var_inter / len(lam.sims)

            # Store the final results, converting to arrays for consistency.
            tot_sem = _np.sqrt(squared_sem_inter + squared_sem_intra)
            sem_intra = _np.sqrt(squared_sem_intra)
            sem_inter = _np.sqrt(squared_sem_inter)
            gradients_all_winds.append(_np.array(gradients_wind))
            means_all_winds.append(mean_overall)
            sems_tot_all_winds.append(tot_sem)
            sems_intra_all_winds.append(sem_intra)
            sems_inter_all_winds.append(sem_inter)
            vars_intra_all_winds.append(vars_intra.mean())
            stat_ineffs_all_winds.append(_np.mean(stat_ineffs_wind))

        # Save the calculated attributes
        self.lam_vals = lam_vals
        self.gradients = gradients_all_winds
        self.means = means_all_winds
        self.sems_overall = sems_tot_all_winds
        self.sems_intra = sems_intra_all_winds
        self.sems_inter = sems_inter_all_winds
        self.vars_intra = vars_intra_all_winds
        self.stat_ineffs = stat_ineffs_all_winds
