"""Functionality for running simulations at a single lambda value."""

import glob as _glob
import logging as _logging
import numpy as _np
import os as _os
import subprocess as _subprocess
from typing import (
    Dict as _Dict,
    List as _List,
    Tuple as _Tuple,
    Any as _Any,
    Optional as _Optional,
    Union as _Union,
)

from ..analyse.detect_equil import (
    check_equil_block_gradient as _check_equil_block_gradient,
    check_equil_chodera as _check_equil_chodera,
    dummy_check_equil_multiwindow as _dummy_check_equil_multiwindow,
)
from .simulation import Simulation as _Simulation
from ._simulation_runner import SimulationRunner as _SimulationRunner
from ._virtual_queue import VirtualQueue as _VirtualQueue


class LamWindow(_SimulationRunner):
    """A class to hold and manipulate a set of SOMD simulations at a given lambda value."""

    equil_detection_methods = {
        "multiwindow": _dummy_check_equil_multiwindow,
        "block_gradient": _check_equil_block_gradient,
        "chodera": _check_equil_chodera,
    }

    def __init__(
        self,
        lam: float,
        virtual_queue: _VirtualQueue,
        lam_val_weight: _Optional[float] = None,
        block_size: float = 1,
        equil_detection: str = "block_gradient",
        gradient_threshold: _Optional[float] = None,
        runtime_constant: _Optional[float] = None,
        ensemble_size: int = 5,
        base_dir: _Optional[str] = None,
        input_dir: _Optional[str] = None,
        output_dir: _Optional[str] = None,
        stream_log_level: int = _logging.INFO,
        update_paths: bool = True,
    ) -> None:
        """
        Initialise a LamWindow object.

        Parameters
        ----------
        lam : float
            Lambda value for the simulation.
        virtual_queue : VirtualQueue
            VirtualQueue object to use for submitting jobs.
        lam_val_weight : float, Optional, default: None
            Weight to use for this lambda value in the free energy calculation.
            This must be suplied if using the check_equil_shrinking_block_gradient
            method for equilibration detection.
        block_size : float, Optional, default: 1
            Size of the blocks to use for equilibration detection,
            in ns.
        equil_detection : str, Optional, default: "block_gradient"
            Method to use for equilibration detection. Options are:
            - "multiwindow": Use the multiwindow method to detect equilibration.
            - "block_gradient": Use the gradient of the block averages to detect equilibration.
            - "chodera": Use Chodera's method to detect equilibration.
        gradient_threshold : float, Optional, default: None
            The threshold for the absolute value of the gradient, in kcal mol-1 ns-1,
            below which the simulation is considered equilibrated. If None, no theshold is
            set and the simulation is equilibrated when the gradient passes through 0. A
            sensible value appears to be 0.5 kcal mol-1 ns-1.
        runtime_constant : float, Optional, default: None
            The runtime constant to use for the calculation. This must be supplied if running
            adaptively. Each window is run until the SEM**2 / runtime >= runtime_constant.
        ensemble_size : int, Optional, default: 5
            Number of simulations to run at this lambda value.
        base_dir : str, Optional, default: None
            Path to the base directory. If None,
            this is set to the current working directory.
        input_dir : str, Optional, default: None
            Path to directory containing input files for the simulations. If None, this
            will be set to "current_working_directory/input".
        output_dir : str, Optional, default: None
            Path to directory to store output files from the simulations. If None, this
            will be set to "current_working_directory/output".
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            Ensemble object and its child objects.
        update_paths: bool, Optional, default: True
            If true, if the simulation runner is loaded by unpickling, then
            update_paths() is called.

        Returns
        -------
        None
        """
        # Set the lamdbda value first, as this is required for __str__,
        # and therefore the super().__init__ call
        self.lam = lam

        super().__init__(
            base_dir=base_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            stream_log_level=stream_log_level,
            ensemble_size=ensemble_size,
            update_paths=update_paths,
        )

        if not self.loaded_from_pickle:
            self.lam_val_weight = lam_val_weight
            self.virtual_queue = virtual_queue
            self.block_size = block_size
            if equil_detection not in self.equil_detection_methods:
                raise ValueError(
                    f"Equilibration detection method {equil_detection} not recognised."
                )
            # Need to pass self object to equilibration detection function
            self.check_equil = self.equil_detection_methods[equil_detection]
            # Ensure that we do not try to use a gradient threshold with a method that does not use it
            if equil_detection != "block_gradient" and gradient_threshold is not None:
                raise ValueError(
                    "Gradient threshold can only be set for block gradient method."
                )
            self.gradient_threshold = gradient_threshold
            self.runtime_constant = runtime_constant
            self._equilibrated: bool = False
            self._running: bool = False

            # Create the required simulations for this lambda value
            for run_no in range(1, ensemble_size + 1):
                # Copy the input files over to the simulation base directory,
                # which is also the simulation input directory and output directory
                run_name = "run_" + str(run_no).zfill(2)
                sim_base_dir = _os.path.join(self.base_dir, run_name)
                _subprocess.call(["mkdir", "-p", sim_base_dir])
                for file in _glob.glob(_os.path.join(self.input_dir, "*")):
                    # If the file is a coordinates, topology, or restraint file,
                    # make a symbolic link to it, otherwise copy it
                    if file.endswith((".rst7", ".prm7", ".txt")):
                        _subprocess.call(
                            [
                                "ln",
                                "-s",
                                _os.path.relpath(file, sim_base_dir),
                                sim_base_dir,
                            ]
                        )
                    else:
                        _subprocess.call(["cp", file, sim_base_dir])
                self.sims.append(
                    _Simulation(
                        lam=lam,
                        run_no=run_no,
                        virtual_queue=virtual_queue,
                        base_dir=sim_base_dir,
                        input_dir=sim_base_dir,
                        output_dir=sim_base_dir,
                        stream_log_level=stream_log_level,
                    )
                )

            # Save the state and update log
            self._update_log()
            self._dump()

    def __str__(self) -> str:
        return f"LamWindow (lam={self.lam:.3f})"

    @property
    def sims(self) -> _List[_Simulation]:
        return self._sub_sim_runners

    @sims.setter
    def legs(self, value) -> None:
        self._logger.info("Modifying/ creating simulations")
        self._sub_sim_runners = value

    def run(self, duration: float = 2.5) -> None:
        """
        Run all simulations at the lambda value.

        Parameters
        ----------
        duration : float, Optional, default: 2.5
            Duration of simulation, in ns.

        Returns
        -------
        None
        """
        # Run the simulations
        self._logger.info(f"Running simulations for {duration:.3f} ns")
        for sim in self.sims:
            sim.run(duration)

        self._running = True

    def kill(self) -> None:
        """Kill all simulations at the lambda value."""
        if self.running:
            self._logger.info("Killing all simulations")
            for sim in self.sims:
                sim.kill()
            self._running = False

    @property
    def equilibrated(self) -> bool:
        """
        Check if the ensemble of simulations at the lambda window is
        equilibrated, and update the equilibration time and status if
        so.

        Returns
        -------
        self._equilibrated : bool
            True if the simulation is equilibrated, False otherwise.
        """
        self._equilibrated, self._equil_time = self.check_equil(self)
        return self._equilibrated

    @property
    def equil_time(self) -> float:
        # Avoid calling expensive self.check_equil() function if we don't have to
        if not self._equilibrated:
            self._equilibrated, self._equil_time = self.check_equil(self)
        if self._equil_time is None:
            raise RuntimeError(
                "Equilibration is not complete so equilibration time cannot be determined"
            )
        return self._equil_time  # ns

    @property
    def failed_simulations(self) -> _List[_SimulationRunner]:
        """The failed simulations"""
        return [sim for sim in self.sims if sim.failed]

    def _get_rolling_average(
        self, data: _np.ndarray, idx_block_size: int
    ) -> _np.ndarray:
        """
        Calculate the rolling average of a 1D array.

        Parameters
        ----------
        data : np.ndarray
            1D array of data to be block averaged.
        idx_block_size : int
            Index size of the blocks to be used in the block average.

        Returns
        -------
        block_av : np.ndarray
            1D array of block averages of the same length as data.
            Initial values (before there is sufficient data to calculate
            a block average) are set to nan.
        """
        if idx_block_size > len(data):
            raise ValueError(
                "Block size cannot be larger than the length of the data array."
            )

        rolling_av = _np.full(len(data), _np.nan)

        for i in range(len(data)):
            if i < idx_block_size:
                continue
            else:
                block_av = _np.mean(data[i - idx_block_size : i])
                rolling_av[i] = block_av

        return rolling_av

    def analyse(self) -> None:
        raise NotImplementedError(
            "Analysis cannot be performed for a single lambda window."
        )

    def analyse_convergence(self) -> None:
        raise (
            NotImplementedError(
                "Convergence analysis is not performed for a single lambda window, only "
                " at the level of a stage or above."
            )
        )

    def setup(self) -> None:
        raise NotImplementedError("LamWindows are set up when they are created")

    @property
    def is_complete(self) -> bool:
        raise NotImplementedError()
