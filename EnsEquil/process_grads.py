"""Functionality to process the gradient data."""

import numpy as _np
from typing import List as _List, Tuple as _Tuple, Optional as _Optional

def get_gradient_data(lams: _List["LamWindow"], 
                      equilibrated: bool, 
                      inter_var: bool = True,
                      intra_var: bool = True, 
                    ) -> _Tuple[_List[_np.ndarray], _List[_np.ndarray],_List[ _np.ndarray]]:
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
    gradients_all_winds : _List[_np.ndarray]
        Array of the gradients for each lambda window.
    means_all_winds : _List[_np.ndarray]
        Array of the means of the gradients for each lambda window.
    variances_all_winds : _List[_np.ndarray]
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

    # Return lists of numpy arrays
    return gradients_all_winds, means_all_winds, variances_all_winds