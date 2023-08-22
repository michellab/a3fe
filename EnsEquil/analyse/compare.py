"""Functionality for comparing two or more simulation runners."""


import numpy as _np
from ..run._utils import SimulationRunnerIterator as _SimulationRunnerIterator

from typing import List as _List, Tuple as _Tuple, Union as _Union


def get_comparitive_convergence_data(
    sim_runners: _SimulationRunnerIterator,
) -> _List[_Tuple[_np.ndarray, _np.ndarray]]:
    """
    Get the convergence of multiple simulation runners against each other. The maximum time used for the
    comparison is taken from the shortest simulation runner.

    Parameters
    ----------
    sim_runners : SimulationRunnerIterator
        An iterator of simulation runners to compare.

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
            fraction=fraction, equilibrated=False
        )
        times = fracs * sim_runner.tot_simtime
        results.append((times, free_energies))

    return results
