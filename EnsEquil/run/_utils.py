""""Utilities for SimulationRunners."""

import BioSimSpace.Sandpit.Exscientia as _BSS
import contextlib as _contextlib
from logging import Logger as _Logger
import os as _os
from time import sleep as _sleep
from typing import (
    Callable as _Callable,
    Tuple as _Tuple,
    Optional as _Optional,
    List as _List,
)

from ._simulation_runner import SimulationRunner as _SimulationRunner


def check_has_wat_and_box(system: _BSS._SireWrappers._system.System) -> None:  # type: ignore
    """Check that the system has water and a box."""
    if system.getBox() == (None, None):
        raise ValueError("System does not have a box.")
    if system.nWaterMolecules() == 0:
        raise ValueError("System does not have water.")


def get_simtime(
    sim_runner: _SimulationRunner, run_nos: _Optional[_List[int]] = None
) -> float:
    """
    Get the simulation time of a sub simulation runner, in ns. This function
    is used with multiprocessing to speed up the calculation.

    Parameters
    ----------
    sim_runner : SimulationRunner
        The simulation runner to get the simulation time of.
    run_nos : List[int], Optional, default: None
        The run numbers to use for MBAR. If None, all runs will be used.
    """
    run_nos = sim_runner._get_valid_run_nos(run_nos=run_nos)
    return sim_runner.get_tot_simtime(run_nos=run_nos)  # ns


#### Adapted from https://stackoverflow.com/questions/50246304/using-python-decorators-to-retry-request ####
def retry(
    times: int, exceptions: _Tuple[Exception], wait_time: int, logger: _Logger
) -> _Callable:
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
                        f"Exception thrown when attempting to run {func}, attempt "
                        f"{attempt} of {times}"
                    )
                    logger.error(f"Exception thrown: {e}")
                    attempt += 1
                    # Wait for specified time before trying again
                    _sleep(wait_time)

            return func(*args, **kwargs)

        return newfn

    return decorator


#### Adapted from https://stackoverflow.com/questions/75048986/way-to-temporarily-change-the-directory-in-python-to-execute-code-without-affect ####
@_contextlib.contextmanager
def TmpWorkingDir(path):
    """Temporarily changes to path working directory."""
    old_cwd = _os.getcwd()
    print(f"Changing directory to {path}")
    _os.chdir(_os.path.abspath(path))
    try:
        yield
    finally:
        print(f"Changing directory to {old_cwd}")
        _os.chdir(old_cwd)
