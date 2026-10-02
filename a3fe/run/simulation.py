"""Functionality to run a single simulation."""

__all__ = ["Simulation"]

import glob as _glob
import logging as _logging
import os as _os
import pathlib as _pathlib
import subprocess as _subprocess
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np

from ..configuration import EngineType as _EngineType
from ..configuration import JobStatus as _JobStatus
from ..configuration import SlurmConfig as _SlurmConfig
from ..configuration import _EngineConfig
from ._simulation_runner import SimulationRunner as _SimulationRunner
from ._virtual_queue import Job as _Job
from ._virtual_queue import VirtualQueue as _VirtualQueue


class Simulation(_SimulationRunner):
    """Class to store information about a single simulation."""

    # Files to be cleaned by self.clean()
    run_files = _SimulationRunner.run_files + ["*.out"]

    def __init__(
        self,
        lam: float,
        run_no: int,
        virtual_queue: _VirtualQueue,
        base_dir: _Optional[str] = None,
        input_dir: _Optional[str] = None,
        output_dir: _Optional[str] = None,
        stream_log_level: int = _logging.INFO,
        slurm_config: _Optional[_SlurmConfig] = None,
        analysis_slurm_config: _Optional[_SlurmConfig] = None,
        engine_config: _Optional[_EngineConfig] = None,
        engine_type: _EngineType = _EngineType.SOMD,
        update_paths: bool = True,
    ) -> None:
        """
        Initialise a Simulation object.

        Parameters
        ----------
        lam : float
            Lambda value for the simulation.
        run_no : int
            Index of repeat for the simulation.
        virtual_queue : VirtualQueue
            Virtual queue object to use for the simulation.
        base_dir : str, Optional, default: None
            Path to the base directory. If None,
            this is set to the current working directory.
        input_dir : str, Optional, default: None
            Path to directory containing input files for the simulation. If None, this
            will be set to "current_working_directory/input".
        output_dir : str, Optional, default: None
            Path to directory to store output files from the simulation. If None, this
            will be set to "current_working_directory/output".
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            simulation object and its child objects.
        slurm_config: SlurmConfig, default: None
            Configuration for the SLURM job scheduler. If None, the
            default partition is used.
        analysis_slurm_config: SlurmConfig, default: None
            Configuration for the SLURM job scheduler for the analysis.
            This is helpful e.g. if you want to submit analysis to the CPU
            partition, but the main simulation to the GPU partition. If None,
        engine_config: EngineConfig, default: None
            Configuration for the engine. If None, the default configuration is used.
        engine_type: EngineType, default: EngineType.SOMD
            The type of engine to use for the production simulations.
        update_paths: bool, Optional, default: True
            If True, if the simulation runner is loaded by unpickling, then
            update_paths() is called.

        Returns
        -------
        None
        """
        # Set the lambda value and run number first, as these are
        # required for __str__, and therefore the super().__init__ call
        self.lam = lam
        self.run_no = run_no

        super().__init__(
            base_dir=base_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            stream_log_level=stream_log_level,
            slurm_config=slurm_config,
            analysis_slurm_config=analysis_slurm_config,
            engine_config=engine_config,
            engine_type=engine_type,
            update_paths=update_paths,
            dump=False,
        )

        if self.engine_config.lambda_values is None:  # type: ignore
            raise ValueError("No lambda values specified in engine config")

        if self.lam not in self.engine_config.lambda_values:
            raise ValueError(
                f"Lambda value {self.lam} not in list of lambda values: {self.engine_config.lambda_values}"  # type: ignore
            )

        if not self.loaded_from_pickle:
            self.virtual_queue = virtual_queue
            # Check that the input directory contains the required files
            self._validate_input()
            self.job: _Optional[_Job] = None
            self._running: bool = False
            # Select the correct coordinate and, if supplied, restraint files
            self._select_input_files()

            # Save state and update log
            self._dump()
            self._update_log()

    def __str__(self) -> str:
        return f"Simulation (lam={self.lam}, run_no={self.run_no})"

    @property
    def running(self) -> bool:
        """
        Check if the simulation is still running,
        and update the running attribute accordingly.
        Also resubmit job if it has failed.

        Returns
        -------
        self._running : bool
            True if the simulation is still running, False otherwise.
        """
        if self.job is None:
            self._running = False
            return self._running

        # Get job ids of currently running jobs - but note that the queue is updated at the
        # Stage level
        if self.job in self.virtual_queue.queue:
            self._running = True
            self._logger.info("Still running")

        else:  # Must have finished
            self._logger.info("Not running")
            self._running = False
            # Check that job finished successfully
            if self.job.status == _JobStatus.FINISHED:
                self._logger.info(f"{self.job} finished successfully")
            elif self.job.status == _JobStatus.FAILED:
                old_job = self.job
                self._logger.info(f"{old_job} failed - resubmitting")
                # Move log files and checkpoint files so that the job does not restart
                _subprocess.run(["mkdir", "-p", f"{self.output_dir}/failure"])

                restart_file_pattern = _os.path.join(
                    self.output_dir, self.engine_backend.restart_file_pattern
                )
                for restart_file in _glob.glob(restart_file_pattern, recursive=True):
                    _subprocess.run(["mv", restart_file, f"{self.output_dir}/failure"])

                _subprocess.run(
                    ["mv", old_job.slurm_outfile, f"{self.output_dir}/failure"]
                )

                # Now resubmit
                cmd_list = old_job.command_list
                self.job = self.virtual_queue.submit(
                    command_list=cmd_list, slurm_file_base=self.slurm_file_base
                )
                self._logger.info(f"{old_job} failed and was resubmitted as {self.job}")
                self._running = True

        return self._running

    def _validate_input(self) -> None:
        """Check that the required input files are present."""

        # Check that the input directory exists
        if not _os.path.isdir(self.input_dir):
            raise FileNotFoundError("Input directory does not exist.")

        # Check that the required input files are present
        for file in self.engine_backend.required_input_files:
            if not _os.path.isfile(_os.path.join(self.input_dir, file)):
                raise FileNotFoundError("Required input file " + file + " not found.")

    def _select_input_files(self) -> None:
        """Select the coordinate and restraint files for this run."""
        file_prefix, file_extension = (
            self.engine_backend.coordinate_prefix_and_extension
        )

        coordinate_files = _glob.glob(f"{self.input_dir}/*.{file_extension}")
        if len(coordinate_files) == 0:
            raise FileNotFoundError(
                f"No {file_extension} files found in input directory"
            )
        elif len(coordinate_files) > 1:
            source_file = (
                f"{self.input_dir}/{file_prefix}_{self.run_no}.{file_extension}"
            )
            target_file = f"{self.input_dir}/{file_prefix}.{file_extension}"
            self._logger.debug(f"Multiple {file_extension} files found - renaming")
            _subprocess.run(["mv", source_file, target_file])
            unwanted_files = _glob.glob(
                f"{self.input_dir}/{file_prefix}_?.{file_extension}"
            )
            for file in unwanted_files:
                _subprocess.run(["rm", file])
        else:
            self._logger.info(f"Only one {file_extension} file found - not renaming")

        # Deal with restraints. Get the name of the restraint file for this run
        old_restr_file = _os.path.join(self.input_dir, f"restraint_{self.run_no}.txt")

        # If we already have a restraints.txt file, continue,
        if _os.path.isfile(_os.path.join(self.input_dir, "restraint.txt")):
            self._logger.info("restraint.txt file found")
        elif _os.path.isfile(old_restr_file):
            self._logger.info("restraint.txt file found - renaming")
            target_restr_file = _os.path.join(self.input_dir, "restraint.txt")
            self._logger.info(f"Renaming {old_restr_file} to {target_restr_file}")
            _subprocess.run(["mv", old_restr_file, target_restr_file])
            unwanted_rest_files = _glob.glob(
                _os.path.join(self.input_dir, "restraint_?.txt")
            )
            for file in unwanted_rest_files:
                _subprocess.run(["rm", file])
        else:
            self._logger.debug("No restraint file found")

    @property
    def slurm_file_base(self) -> str:
        """Get the base name of the SLURM output file."""
        slurm_file_base = self.slurm_config.get_slurm_output_file_base(
            run_dir=self.input_dir
        )
        self._logger.debug(f"Found slurm output file basename: {slurm_file_base}")
        return slurm_file_base

    def run(self, runtime: float = 2.5) -> None:
        """
        Run a simulation.

        Parameters
        ----------
        runtime : float, Optional, default: 2.5
            Runtime of simulation, in ns.

        Returns
        -------
        None
        """
        self.engine_backend.write_run_config(
            config=self.engine_config,
            output_dir=self.output_dir,
            lam=self.lam,
            runtime=runtime,
        )

        # Get the commands to run the simulation
        cmd = self.engine_config.get_run_cmd(self.lam)
        cmd_list = self.slurm_config.get_submission_cmds(
            cmd=cmd, run_dir=self.output_dir
        )

        self.job = self.virtual_queue.submit(
            command_list=cmd_list, slurm_file_base=self.slurm_file_base
        )
        self._logger.info(f"Submitted with job {self.job}")

    def get_tot_simtime(self) -> float:
        """Get the total simulation time in ns."""
        return self.engine_backend.get_tot_simtime(self.output_dir, self.engine_config)

    def get_tot_gpu_time(self) -> float:
        """Get the total simulation time in GPU hours."""
        return self.engine_backend.get_tot_gpu_time(self.slurm_output_files)

    @property
    def tot_simtime(self) -> float:
        """Get the total simulation time in ns"""
        return self.get_tot_simtime()

    @property
    def tot_gpu_time(self) -> float:
        """Get the total simulation time in GPU hours"""
        # Get output files
        return self.get_tot_gpu_time()

    @property
    def failed(self) -> bool:
        """Whether the simulation has failed."""
        if self.running or self.job is None:
            return False
        return not self.engine_backend.run_completed(
            self.output_dir, self.slurm_output_files
        )

    @property
    def slurm_output_files(self) -> _List[str]:
        """Get a list of all slurm output files for this simulation."""
        return _glob.glob(f"{self.slurm_file_base}*")

    def kill(self) -> None:
        """Kill the job."""
        if not self.job:
            raise ValueError("Stage has no job object. Cannot kill job.")
        if self.job in self.virtual_queue.queue:
            self._logger.info(f"Killing job {self.job}")
            self.virtual_queue.kill(self.job)

    def lighten(self) -> None:
        """Lighten the simulation by deleting restart and trajectory files."""
        for directory in [self.base_dir, self.output_dir]:
            for pattern in self.engine_backend.lighten_file_patterns:
                for file in _pathlib.Path(directory).glob(pattern):
                    self._logger.info(f"Deleting {file}")
                    _subprocess.run(["rm", str(file)])

    def read_gradients(
        self, equilibrated_only: bool = False, endstate: bool = False
    ) -> _Tuple[_np.ndarray, _np.ndarray]:
        """Read simulation times in ns and gradients in kcal/mol."""
        return self.engine_backend.read_gradients(
            output_dir=self.output_dir,
            config=self.engine_config,
            equilibrated_only=equilibrated_only,
            endstate=endstate,
        )

    def update_paths(self, old_sub_path: str, new_sub_path: str) -> None:
        """
        Replace the old sub-path with the new sub-path in the base, input, and output directory

        Parameters
        ----------
        old_sub_path : str
            The old sub-path to replace.
        new_sub_path : str
            The new sub-path to replace the old sub-path with.
        """
        super().update_paths(old_sub_path, new_sub_path)

    def analyse(self) -> None:
        raise NotImplementedError(
            "Analysis cannot be performed for a single simulation"
        )

    @property
    def equil_time(self) -> None:
        raise NotImplementedError(
            "Equilibration time is not determined for a single simulation, only "
            "for an ensemble of simulations within a lambda window."
        )

    @property
    def equilibrated(self) -> None:
        raise NotImplementedError(
            "Equilibration is not detected at the level of single simulations, only "
            "for an ensemble of simulations within a lambda window."
        )

    def set_equilibration_time(self, equil_time: float) -> None:
        raise NotImplementedError(
            "Equilibration time is not set for a single simulation, only "
            "for an ensemble of simulations within a lambda window."
        )

    def analyse_convergence(self) -> None:
        raise (
            NotImplementedError(
                "Convergence analysis is not performed for a single simulation, only "
                " at the level of a stage or above."
            )
        )

    def setup(self) -> None:
        raise NotImplementedError("Simulations are set up when they are created")
