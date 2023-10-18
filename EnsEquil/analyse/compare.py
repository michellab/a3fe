"""Functionality for comparing two or more simulation runners."""


import numpy as _np
from ..run._utils import SimulationRunnerIterator as _SimulationRunnerIterator

from scipy.stats import levene as _levene

from typing import List as _List, Tuple as _Tuple, Union as _Union

__all__ = [
    "get_comparitive_convergence_data",
    "compare_variances_brown_forsythe",
]


def compare_variances_brown_forsythe(sim_runners: _SimulationRunnerIterator) -> float:
    """
    Compare the variances of the free energies of two simulation runners using the Brown-Forsythe test.

    Parameters
    ----------
    sim_runners : SimulationRunnerIterator
        An iterator of two simulation runners to compare.

    Returns
    -------
    float
        The p-value of the Brown-Forsythe test.
    """
    # Check that we have been given two simulation runners
    if len(sim_runners) != 2:
        raise ValueError(
            "Must provide exactly two simulation runners to compare variances."
        )
    # Get the stored free energies for each simulation runner
    free_energies = []
    for sim_runner in sim_runners:
        # Analyse the calculation if this has not been done already.
        if not hasattr(sim_runner, "_delta_g"):
            sim_runner.analyse()
            sim_runner._dump()
        free_energies.append(sim_runner._delta_g)

    # Get the Brown-Forsythe test statistic and p-value
    stat, p = _levene(*free_energies, center="median", proportiontocut=0.00)

    return p


def get_comparitive_convergence_data(
    sim_runners: _SimulationRunnerIterator,
    equilibrated: bool = False,
) -> _List[_Tuple[_np.ndarray, _np.ndarray]]:
    """
    Get the convergence of multiple simulation runners against each other. The maximum time used for the
    comparison is taken from the shortest simulation runner.

    Parameters
    ----------
    sim_runners : SimulationRunnerIterator
        An iterator of simulation runners to compare.
    equilibrated : bool, default: False
        Whether to use the equilibrated simulation time or the total simulation time. If False,
        all simulation data will be used, otherwise only the equilibrated data will be used.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray], ...]
        A tuple of tuples of the convergence of each simulation runner. For each simulation runner, the first array is the
        total simulation time plotted, and the second array is the corresponding average free energy change for each value
        of total equilibrated simtime for each of the ensemble size repeats.
    """
    # Get the minimum time used for the comparison
    min_time = float("inf")
    for sim_runner in sim_runners:
        # Make sure that the equilibration time is zero in all cases to allow for a fair comparison
        min_time = min(min_time, sim_runner.tot_simtime)

    # Get the convergence of each simulation runner
    results = []
    for sim_runner in sim_runners:
        # Adjust the fraction analysed so that the total time is the same for all simulation runners
        fraction = min_time / sim_runner.tot_simtime
        fracs, free_energies = sim_runner.analyse_convergence(
            fraction=fraction, equilibrated=equilibrated
        )
        times = fracs * sim_runner.tot_simtime
        results.append((times, free_energies))

    return results
