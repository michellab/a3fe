""" "Utilities for SimulationRunners."""

from __future__ import annotations

import contextlib as _contextlib
import os as _os
from logging import Logger as _Logger
from time import sleep as _sleep
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Generic as _Generic
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Type as _Type
from typing import TypeVar as _TypeVar

import BioSimSpace as _BSS

_T = _TypeVar("_T", bound="SimulationRunner")  # noqa: F821


def check_has_wat_and_box(system: _BSS._SireWrappers._system.System) -> None:  # type: ignore
    """Check that the system has water and a box."""
    if system.getBox() == (None, None):
        raise ValueError("System does not have a box.")
    if system.nWaterMolecules() == 0:
        raise ValueError("System does not have water.")

def get_single_mol(system: _BSS._SireWrappers._system.System, mol_name: str) -> BSS._SireWrappers._molecule.Molecule:  # type: ignore
    """Get a single molecule from a BSS system."""
    mols = system.search(f"resname {mol_name}").molecules()
    if len(mols) != 1:
        raise ValueError(f"Expected 1 molecule with name {mol_name}, got {len(mols)}")
    return mols[0]

def get_simtime(
    sim_runner: "SimulationRunner",  # noqa: F821
    run_nos: _Optional[_List[int]] = None,
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


class SimulationRunnerIterator(_Generic[_T]):
    """
    Iterator for SimulationRunners. This is required to avoid too many
    open files, because each simulation runner opens its own loggers.
    Hence, simulation runners are set up before being yielded, and then
    deleted after being yielded.
    """

    def __init__(
        self,
        base_dirs: _List[str],
        subclass: _Type[_T],
        **kwargs: _Any,
    ) -> None:
        """
        Parameters
        ----------
        base_dirs : List[str]
            A list of the base directories for the simulation runners.
        subclass : Type[_T]
            The subclass of SimulationRunner to use.
        **kwargs : Any
            Any keyword arguments to pass to the subclass when initialising.
        """
        self.base_dirs = base_dirs
        self.subclass = subclass
        self.current_sim_runner = None
        self.kwargs = kwargs
        self.i = 0

    def __iter__(self):
        self.i = 0  # Reset the iterator so we can reuse it
        return self

    def __next__(self) -> _T:
        if self.i >= len(self.base_dirs):
            # Tear down the current simulation runner
            if self.current_sim_runner is not None:
                self.current_sim_runner._close_logging_handlers()
                del self.current_sim_runner
                self.current_sim_runner = None
            raise StopIteration

        # Tear down the current simulation runner
        if self.current_sim_runner is not None:
            self.current_sim_runner._close_logging_handlers()
            del self.current_sim_runner
        # Set up the next simulation runner
        self.current_sim_runner = self.subclass(
            **self.kwargs, base_dir=self.base_dirs[self.i]
        )
        self.i += 1
        return self.current_sim_runner

    def __len__(self) -> int:
        return len(self.base_dirs)
