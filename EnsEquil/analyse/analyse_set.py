"""Functionality to analyse a set of calculations and compare the result with experiment"""

__all__ = [
    "compute_stats",
    "get_bootstrapped_results",
    "compute_statistic",
]

from ensurepip import bootstrap
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple

import numpy as _np
import pandas as _pd
from scipy import stats as _stats
from sklearn import metrics as _metrics


def compute_stats(all_results: _pd.DataFrame) -> _Dict[str, _List[float]]:
    """
    Compute statistics for the passed results, generating 95 % C.I.s
    by bootstrapping.

    Parameters
    ----------
    all_results : pd.DataFrame
        The dataframe containing all results.

    Returns
    -------
    Dict[str, List[float]]
        A dictionary of the computed statistics, and their upper and lower
        confidence bounds.
    """

    # This will hold metric: [value, upper_err, lower_er]
    results = {"r": [], "r2": [], "mue": [], "rmse": [], "rho": [], "tau": []}

    # Get the bootstrapped results
    n_bootstrap = 1000
    bootstrapped_exp_dg, bootstrapped_calc_dg = get_bootstrapped_results(
        all_results=all_results, n_bootstrap=n_bootstrap
    )

    # For each metric, calculate i) over the actual data ii) overall bootstrapped data and extract stats
    for metric in results:
        results[metric].append(
            compute_statistic(all_results["exp_dg"], all_results["calc_dg"], metric)
        )
        bootstrapped_metric = _np.zeros([n_bootstrap])
        for i in range(n_bootstrap):
            bootstrapped_metric[i] = compute_statistic(
                bootstrapped_exp_dg[i], bootstrapped_calc_dg[i], metric
            )
        percentiles = _np.percentile(bootstrapped_metric, [5, 95])  # 95 % C.I.s
        results[metric].append(percentiles[0])
        results[metric].append(percentiles[1])

    return results


def get_bootstrapped_results(
    all_results: _pd.DataFrame, n_bootstrap: int = 1000
) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Return n_bootstrap bootstrapped versions of the original experimental
    and calculated free energies.

    Parameters
    ----------
    all_results : pd.DataFrame
        The dataframe containing all results.
    n_bootstrap : int, optional, default = 1000
        Number of boostrap iterations to perform

    Returns
    -------
    boostrapped_exp_dg: np.ndarray
        The bootstrapped experimental free energy changes
    bootstrapped_calc_dg: np_ndarray
        The bootstrapped calculated free energy changes
    """
    exp_dg = all_results["exp_dg"]
    calc_dg = all_results["calc_dg"]
    exp_sem = all_results["exp_er"] / 1.96
    calc_sem = all_results["calc_er"] / 1.96

    # Check that the data passed are of the same length
    if len(exp_dg) != len(calc_dg):
        raise ValueError(
            "The lengths of the calculated and experimental free energy values must match"
        )
    n_samples = len(exp_dg)

    bootstrapped_exp_dg = _np.zeros([n_bootstrap, n_samples])
    bootstrapped_calc_dg = _np.zeros([n_bootstrap, n_samples])
    for i in range(n_bootstrap):
        # Ensure we use same indices for the experimental and calculated results to avoid mixing
        # results
        indices = _np.random.choice(_np.arange(n_samples), size=n_samples, replace=True)
        bootstrapped_exp_dg[i] = _np.array(
            [_np.random.normal(loc=exp_dg[i], scale=exp_sem[i]) for i in indices]
        )
        bootstrapped_calc_dg[i] = _np.array(
            [_np.random.normal(loc=calc_dg[i], scale=calc_sem[i]) for i in indices]
        )

    return bootstrapped_exp_dg, bootstrapped_calc_dg


def compute_statistic(exp_dg: _pd.Series, calc_dg: _pd.Series, statistic: str) -> float:
    """
    Compute the desired statistic for one set of experimental and
    calculated values.

    Parameters
    ----------
    exp_dg : pd.Series
        The experimental free energies
    calc_dg : pd.Series
        The calculated free energies
    statistic : str
        The desired statistic to be calculated, from "r", "mue", "rmse"
        "rho", or "tau".

    Returns
    -------
    float
        The desired statistic.
    """
    # Check that requested metric is implemented
    allowed_stats = ["r", "r2", "mue", "rmse", "rho", "tau"]
    if statistic not in allowed_stats:
        raise ValueError(
            f"Statistic must be one of {allowed_stats} but was {statistic}"
        )

    if statistic == "r":
        return _stats.pearsonr(exp_dg, calc_dg)[0]
    if statistic == "r2":
        m, c, r, p, sem = _stats.linregress(exp_dg, calc_dg)
        return r**2
    if statistic == "mue":
        return _metrics.mean_absolute_error(exp_dg, calc_dg)
    if statistic == "rmse":
        return _metrics.mean_squared_error(exp_dg, calc_dg)
    if statistic == "rho":
        return _stats.spearmanr(exp_dg, calc_dg)[0]
    if statistic == "tau":
        return _stats.kendalltau(exp_dg, calc_dg)[0]
