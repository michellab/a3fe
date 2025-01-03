"""Functionality for running simulations at a single lambda value."""

__all__ = ["LamWindow"]

import glob as _glob
import logging as _logging
import os as _os
import subprocess as _subprocess
from copy import deepcopy as _deepcopy
from typing import List as _List
from typing import Optional as _Optional
from typing import Union as _Union

import numpy as _np

from ..analyse.detect_equil import check_equil_chodera as _check_equil_chodera
from ..analyse.detect_equil import (
    dummy_check_equil_multiwindow as _dummy_check_equil_multiwindow,
)
from ._simulation_runner import SimulationRunner as _SimulationRunner
from ._virtual_queue import VirtualQueue as _VirtualQueue
from .simulation import Simulation as _Simulation


class LamWindow(_SimulationRunner):
    """A class to hold and manipulate a set of SOMD simulations at a given lambda value."""

    equil_detection_methods = {
        "multiwindow": _dummy_check_equil_multiwindow,
        "chodera": _check_equil_chodera,
    }

    runtime_attributes = _deepcopy(_SimulationRunner.runtime_attributes)
    runtime_attributes["_equilibrated"] = False
    runtime_attributes["_equil_time"] = None

    def __init__(
        self,
        lam: float,
        virtual_queue: _VirtualQueue,
        lam_val_weight: _Optional[float] = None,
        block_size: float = 1,
        equil_detection: str = "multiwindow",
        slurm_equil_detection: bool = True,
        gradient_threshold: _Optional[float] = None,
        runtime_constant: _Optional[float] = 0.005,
        relative_simulation_cost: float = 1,
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
            Weight to use for this lambda value in the free energy calculation
            (e.g. from Trapezoidal rule).
        block_size : float, Optional, default: 1
            Size of the blocks to use for equilibration detection,
            in ns. Only used for the block gradient equilibration detection method.
        equil_detection : str, Optional, default: "multiwindow"
            Method to use for equilibration detection. Options are:
            - "multiwindow": Use the multiwindow paired t-test method to detect equilibration.
            - "block_gradient": Use the gradient of the block averages to detect equilibration.
            - "chodera": Use Chodera's method to detect equilibration.
        slurm_equil_detection : bool, Optional, default: True
            Whether to use SLURM to run the equilibration detection MBAR calculations.
        gradient_threshold : float, Optional, default: None
            The threshold for the absolute value of the gradient, in kcal mol-1 ns-1,
            below which the simulation is considered equilibrated. If None, no theshold is
            set and the simulation is equilibrated when the gradient passes through 0. A
            sensible value appears to be 0.5 kcal mol-1 ns-1. Only required when the equilibration
            detection method is "block_gradient".
        runtime_constant: float, Optional, default: 0.005
            The runtime_constant (kcal**2 mol**-2 ns*-1) only affects behaviour if running adaptively, and must
            be supplied if running adaptively. This is used to calculate how long to run each simulation for based on
            the current uncertainty of the per-window free energy estimate, as discussed in the docstring of the run() method.
        relative_simlation_cost : float, Optional, default: 1
            The relative cost of the simulation for a given runtime. This is used to calculate the
            predicted optimal runtime during adaptive simulations. The recommended use
            is to set this to 1 for the bound leg and to (speed of bound leg / speed of free leg)
            for the free leg.
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
            dump=False,
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
            self.slurm_equil_detection = slurm_equil_detection
            # Ensure that we do not try to use a gradient threshold with a method that does not use it
            if equil_detection != "block_gradient" and gradient_threshold is not None:
                raise ValueError(
                    "Gradient threshold can only be set for block gradient method."
                )
            self.gradient_threshold = gradient_threshold
            self.runtime_constant = runtime_constant
            self.relative_simulation_cost = relative_simulation_cost
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

    def run(self, run_nos: _Optional[_List[int]] = None, runtime: float = 2.5) -> None:
        """
        Run all simulations at the lambda value.

        Parameters
        ----------
        run_nos : List[int], Optional, default: None
            List of run numbers to run. If None, all simulations are run.
        runtime : float, Optional, default: 2.5
            Runtime of simulation, in ns.

        Returns
        -------
        None
        """
        run_nos = self._get_valid_run_nos(run_nos)

        # Run the simulations
        self._logger.info(f"Running simulations {run_nos} for {runtime:.3f} ns")
        for run_no in run_nos:  # type: ignore
            self.sims[run_no - 1].run(runtime)

        self._running = True

    def kill(self) -> None:
        """Kill all simulations at the lambda value."""
        if self.running:
            self._logger.info("Killing all simulations")
            for sim in self.sims:
                sim.kill()
            self._running = False

    def get_tot_simtime(self, run_nos: _Optional[_List[int]] = None) -> float:
        """
        Get the total simulation time for all specified runs, in ns.

        Parameters
        ----------
        run_nos : List[int], Optional, default: None
            The run numbers to use for MBAR. If None, all runs will be used.

        Returns
        -------
        tot_simtime : float
            Total simulation time, in ns.
        """
        run_nos = self._get_valid_run_nos(run_nos)
        return sum([self.sims[run_no - 1].get_tot_simtime() for run_no in run_nos])

    def get_tot_gpu_time(self, run_nos: _Optional[_List[int]] = None) -> float:
        """
        Get the total GPU time for all specified runs, in GPU hours.

        Parameters
        ----------
        run_nos : List[int], Optional, default: None
            The run numbers to use for MBAR. If None, all runs will be used.

        Returns
        -------
        tot_gpu_time : float
            Total GPU time, in GPU hours.
        """
        run_nos = self._get_valid_run_nos(run_nos)
        return sum([self.sims[run_no - 1].get_tot_gpu_time() for run_no in run_nos])

    def is_equilibrated(self, run_nos: _Optional[_List[int]] = None) -> bool:
        """
        Check if the ensemble of simulations at the lambda window is
        equilibrated, based on the run numbers specified and the
        equilibration detection method. Store the equilibration status
        and time in private variables if so.

        Parameters
        ----------
        run_nos : List[int], Optional, default: None
            The run numbers to equilibration detection. If None, all runs will be used.

        Returns
        -------
        equilibrated : bool
            True if the simulation is equilibrated, False otherwise.
        """
        self._equilibrated, self._equil_time = self.check_equil(self, run_nos=run_nos)
        return self._equilibrated

    @property
    def equilibrated(self) -> bool:
        """Whether equilibration has been achieved."""
        return self._equilibrated

    @property
    def equil_time(self) -> _Union[float, None]:
        """The equilibration time in ns, per run."""
        return self._equil_time  # ns

    @property
    def failed_simulations(self) -> _List[_SimulationRunner]:
        """The failed simulations"""
        return [sim for sim in self.sims if sim.failed]

    def set_equilibration_time(self, equil_time: float) -> None:
        """
        Set the equilibration time for the simulation runner and any sub-simulation runners.

        Parameters
        ----------
        equil_time : float
            The equilibration time to set, in ns per run per lambda window.
        """
        self._logger.info(f"Setting equilibration time to {equil_time:.3f} ns per run")
        self._equilibrated = True
        self._equil_time = equil_time

    def _write_equilibrated_simfiles(self) -> None:
        """
        Remove unequilibrated data from simulation files for all simulations
        at the lambda window, and write a new simulation file containing only
        the equilibrated data. This is the natural place for this function
        because the equilibration time is stored at the lambda window level.
        """
        # Check that we have the required equilibration data
        if self.equil_time is None:
            raise ValueError(
                "Equilibration time not set. "
                "Please run is_equilibrated() before calling this function."
            )

        # Get the index of the first equilibrated data point
        # Minus 1 because first energy is only written after the first nrg_freq steps
        equil_index = (
            int(self._equil_time / (self.sims[0].timestep * self.sims[0].nrg_freq)) - 1  # type: ignore
        )

        # Write the equilibrated data for each simulation
        for sim in self.sims:
            in_simfile = sim.output_dir + "/simfile.dat"
            out_simfile = sim.output_dir + "/simfile_equilibrated.dat"

            with open(in_simfile, "r") as ifile:
                lines = ifile.readlines()

            # Figure out how many lines come before the data
            non_data_lines = 0
            for line in lines:
                if line.startswith("#"):
                    non_data_lines += 1
                else:
                    break

            # Overwrite the original file with one containing only the equilibrated data
            with open(out_simfile, "w") as ofile:
                # First, write the header
                for line in lines[:non_data_lines]:
                    ofile.write(line)
                # Now write the data, skipping the non-equilibrated portion
                for line in lines[equil_index + non_data_lines :]:
                    ofile.write(line)

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
