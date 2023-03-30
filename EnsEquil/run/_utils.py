""""Utilities for SimulationRunners."""

import BioSimSpace.Sandpit.Exscientia as _BSS
from ._simulation_runner import SimulationRunner as _SimulationRunner

def check_has_wat_and_box(system: _BSS._SireWrappers._system.System) -> None: # type: ignore
    """Check that the system has water and a box."""
    if system.getBox() == (None, None):
        raise ValueError("System does not have a box.")
    if system.nWaterMolecules() == 0:
        raise ValueError("System does not have water.")

def get_simtime(sim_runner: _SimulationRunner) -> float:
    """
    Get the simulation time of a sub simulation runner, in ns. This function
    is used with multiprocessing to speed up the calculation.
    
    Parameters
    ----------
    sim_runner : SimulationRunner
        The simulation runner to get the simulation time of.
    """
    return sim_runner.tot_simtime # ns