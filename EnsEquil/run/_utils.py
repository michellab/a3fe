""""Utilities for SimulationRunners."""

import BioSimSpace.Sandpit.Exscientia as _BSS
from logging import Logger as _Logger
from time import sleep as _sleep
from typing import Callable as _Callable, Tuple as _Tuple

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

#### Adapted from https://stackoverflow.com/questions/50246304/using-python-decorators-to-retry-request ####
def retry(times:int, 
          exceptions:_Tuple[Exception], 
          wait_time:int,
          logger:_Logger) -> _Callable:
    """
    Retry a function a given number of times if the specified exceptions are raised.

    Parameters
    ----------
    times : int
        The number of retries to attempt before raising the error.
    exceptions : Tuple[Exception]
        A list of exceptions for which the function will be retried. The 
        function will not be retried if an exception is raised which is not
        supplied.
    wait_time : int
        How long to wait between retries, in seconds.

    Returns
    -------
    decorator: Callable
        The retry decorator
    """    
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.error(
                        f'Exception thrown when attempting to run {func}, attempt '
                        f'{attempt} of {times}'
                    )
                    logger.error(f"Exception thrown: {e}")
                    attempt += 1
                    # Wait for specified time before trying again
                    _sleep(wait_time)

            return func(*args, **kwargs)
        return newfn
    return decorator
