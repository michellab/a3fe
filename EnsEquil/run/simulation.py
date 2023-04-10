"""Functionality to run a single SOMD simulation."""

from decimal import Decimal as _Decimal
import glob as _glob
import os as _os
import pathlib as _pathlib
import logging as _logging
import numpy as _np
import subprocess as _subprocess
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

from ..read._process_somd_files import read_simfile_option as _read_simfile_option, write_simfile_option as _write_simfile_option
from ._simulation_runner import SimulationRunner as _SimulationRunner
from ._virtual_queue import Job as _Job, VirtualQueue as _VirtualQueue, JobStatus as _JobStatus

class Simulation(_SimulationRunner):
    """Class to store information about a single SOMD simulation."""

    required_input_files=["run_somd.sh",
                          "somd.cfg",
                          "somd.prm7",
                          "somd.rst7",
                          "somd.pert"]

    # Files to be cleaned by self.clean()
    run_files = _SimulationRunner.run_files + ["*.dcd", 
                                               "*.out", 
                                               "moves.dat", 
                                               "simfile.dat",
                                               "simfile_equilibrated.dat",
                                               "*.s3", 
                                               "*.s3.previous",
                                               "latest.pdb", 
                                               "gradients.dat",]

    def __init__(self, 
                 lam: float, 
                 run_no: int,
                 virtual_queue: _VirtualQueue,
                 base_dir: _Optional[str] = None,
                 input_dir: _Optional[str] = None,
                 output_dir: _Optional[str] = None,
                 stream_log_level: int=_logging.INFO) -> None:
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

        Returns
        -------
        None
        """
        # Set the lambda value and run number first, as these are 
        # required for __str__, and therefore the super().__init__ call
        self.lam=lam
        self.run_no=run_no

        super().__init__(base_dir=base_dir,
                         input_dir=input_dir,
                         output_dir=output_dir,
                         stream_log_level=stream_log_level)

        if not self.loaded_from_pickle:
            self.virtual_queue=virtual_queue
            # Check that the input directory contains the required files
            self._validate_input()
            self.job: _Optional[_Job]=None
            self._running: bool=False
            self.simfile_path = _os.path.join(self.base_dir, "somd.cfg")
            # Select the correct rst7 and, if supplied, restraints
            self._select_input_files()
            # Now set lambda value and some paths in the simfile,
            # as well as the restraints if necessary
            self._update_simfile()
            # Now read useful parameters from the simulation file options
            self._add_attributes_from_simfile()
            # Get slurm file base
            self._get_slurm_file_base()

            # Save state and update log
            self._dump()
            self._update_log()

    def __str__(self) -> str:
        return f"Simulation (lam={self.lam}, run_no={self.run_no})"

    @ property
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
            self._running=False
            return self._running

        # Get job ids of currently running jobs - but note that the queue is updated at the
        # Stage level
        if self.job in self.virtual_queue.queue:
            self._running=True
            self._logger.info(f"Still running")

        else: # Must have finished
            self._logger.info(f"Not running")
            self._running=False
            # Check that job finished successfully
            if self.job.status == _JobStatus.FINISHED:
                self._logger.info(f"{self.job} finished successfully")
            elif self.job.status == "FAILED":
                old_job = self.job
                self._logger.info(f"{old_job} failed - resubmitting")
                # Move log files and s3 files so that the job does not restart
                _subprocess.run(["mkdir", f"{self.output_dir}/failure"])
                for s3_file in _glob.glob(f"{self.output_dir}/*.s3"):
                    _subprocess.run(["mv", f"{self.output_dir}/{s3_file}", f"{self.output_dir}/failure"])
                _subprocess.run(["mv", old_job.slurm_outfile, f"{self.output_dir}/failure"])
                # Now resubmit
                cmd = old_job.command
                self.job=self.virtual_queue.submit(command = cmd, slurm_file_base = self.slurm_file_base)
                self._logger.info(f"{old_job} failed and was resubmitted as {self.job}")
                self._running=True

        return self._running

    def _validate_input(self) -> None:
        """ Check that the required input files are present. """

        # Check that the input directory exists
        if not _os.path.isdir(self.input_dir):
            raise FileNotFoundError("Input directory does not exist.")

        # Check that the required input files are present
        for file in Simulation.required_input_files:
            if not _os.path.isfile(_os.path.join(self.input_dir, file)):
                raise FileNotFoundError("Required input file " + file + " not found.")

    def _add_attributes_from_simfile(self) -> None:
        """
        Read the SOMD simulation option file and
        add useful attributes to the Simulation object.
        All times in ns.
        """

        timestep=None  # ns
        nmoves=None  # number of moves per cycle
        nrg_freq=None  # number of timesteps between energy calculations
        timestep = float(_read_simfile_option(self.simfile_path, "timestep").split()[0]) # Need to remove femtoseconds from the end
        nmoves = float(_read_simfile_option(self.simfile_path, "nmoves"))
        nrg_freq = float(_read_simfile_option(self.simfile_path, "energy frequency"))

        self.timestep=timestep / 1_000_000  # fs to ns
        self.nrg_freq=nrg_freq 
        self.time_per_cycle=timestep * nmoves / 1_000_000  # fs to ns

    def _select_input_files(self) -> None:
        """ Select the correct rst7 and, if supplied, restraints,
        according to the run number. """
        # Check if we have multiple rst7 files, or only one
        rst7_files = _glob.glob(_os.path.join(self.input_dir, "*.rst7"))
        if len(rst7_files) == 0:
            raise FileNotFoundError("No rst7 files found in input directory")
        elif len(rst7_files) > 1:
            # Rename the rst7 file for this run to somd.rst7 and delete any other
            # rst7 files
            self._logger.info("Multiple rst7 files found - renaming")
            _subprocess.run(["mv", _os.path.join(self.input_dir, f"somd_{self.run_no}.rst7"), _os.path.join(self.input_dir, "somd.rst7")])
            unwanted_rst7_files = _glob.glob(_os.path.join(self.input_dir, "somd_?.rst7"))
            for file in unwanted_rst7_files:
                _subprocess.run(["rm", file])
        else:
            self._logger.info("Only one rst7 file found - not renaming")

        # Check how may restraints files we have
        restraint_files = _glob.glob(_os.path.join(self.input_dir, "restraint*.txt"))
        # If we have many restraints files, rename the correct one to
        # somd.restraints and delete any others
        if len(restraint_files) > 1:
            self._logger.info("Multiple restraint files found - renaming")
            _subprocess.run(["mv", _os.path.join(self.input_dir, f"restraint_{self.run_no}.txt"), _os.path.join(self.input_dir, "restraint.txt")])
            unwanted_rest_files = _glob.glob(_os.path.join(self.input_dir, "restraint_?.txt"))
            for file in unwanted_rest_files:
                _subprocess.run(["rm", file])


    def _update_simfile(self) -> None:
        """Set the lambda value in the simulation file, as well as some
        paths to input files."""

        # Check that the lambda value has been set
        if not hasattr(self, "lam"):
            raise AttributeError("Lambda value not set for simulation")

        # Check that the set lambda value is in the list of lamvals in the simfile
        lamvals = [float(lam_val) for lam_val in _read_simfile_option(self.simfile_path, "lambda array").split(",")]
        if self.lam not in lamvals:
            raise ValueError(f"Lambda value {self.lam} not in list of lambda values in simfile")
        
        # Set the lambda value in the simfile
        _write_simfile_option(self.simfile_path, "lambda_val", str(self.lam))
        
        # Set the paths to the input files
        input_paths = {"morphfile": "somd.pert",
                       "topfile": "somd.prm7",
                       "crdfile": "somd.rst7"}

        for option, name in input_paths.items():
            _write_simfile_option(self.simfile_path, option, _os.path.join(self.input_dir, name))

        # Add the restraints file if it exists
        if _os.path.isfile(_os.path.join(self.input_dir, "restraint.txt")):
            with open(_os.path.join(self.input_dir, "restraint.txt"), "r") as f:
                restraint = f.readlines()[0].split("=")[1]
            _write_simfile_option(self.simfile_path, "boresch restraints dictionary", restraint)

    def _get_slurm_file_base(self) -> None:
        """Find out what the slurm output file will be called."""
        # Find the slurm output file
        with open(f"{self.input_dir}/run_somd.sh", "r") as f:
            for line in f:
                split_line = line.split()
                if len(split_line) > 0 and split_line[0] == "#SBATCH":
                    if split_line[1] == "--output" or split_line[1] == "-o":
                        slurm_pattern = split_line[2]
                        if "%" in slurm_pattern:
                            file_base = slurm_pattern.split("%")[0]
                            self.slurm_file_base = _os.path.join(self.input_dir, file_base)
                            self._logger.info(f"Found slurm output file basename: {self.slurm_file_base}")
                            break
                        else:
                            self.slurm_file_base = _os.path.join(self.input_dir, slurm_pattern)
                            self._logger.info(f"Found slurm output file basename: {self.slurm_file_base}")
                            break
        

    def run(self, duration: float=2.5) -> None:
        """
        Run a SOMD simulation.

        Parameters
        ----------
        duration : float, Optional, default: 2.5
            Duration of simulation, in ns.

        Returns
        -------
        None
        """
        # Need to make sure that duration is a multiple of the time per cycle
        # otherwise actual time could be quite different from requested duration
        remainder=_Decimal(str(duration)) % _Decimal(str(self.time_per_cycle))
        if round(float(remainder), 4) != 0:
            raise ValueError(("Duration must be a multiple of the time per cycle. "
                              f"Duration is {duration} ns, and time per cycle is {self.time_per_cycle} ns."))
        # Need to modify the config file to set the correction n_cycles
        n_cycles=int(duration / self.time_per_cycle)
        self._set_n_cycles(n_cycles)

        # Run SOMD - note that command excludes sbatch as this is added by the virtual queue
        cmd=f"--chdir {self.output_dir} {self.input_dir}/run_somd.sh {self.lam}"
        self.job=self.virtual_queue.submit(command = cmd, slurm_file_base=self.slurm_file_base)
        self._logger.info(f"Submitted with job {self.job}")

    @property
    def tot_simtime(self) -> float:
        """Get the total simulation time in ns"""
        # Check that the required file exists
        data_simfile = f"{self.output_dir}/simfile.dat"
        if not _pathlib.Path(data_simfile).is_file():
            # Simuation has not been run, hence total simulation time is 0
            return 0
        elif  _os.stat(data_simfile).st_size == 0:
            # Simfile is empty, hence total simulation time is 0
            return 0
        else:
            # Read last line of simfile with subprocess to make as fast as possible
            step = int(_subprocess.check_output(['tail', '-1', f"{self.output_dir}/simfile.dat"]).decode("utf-8").strip().split()[0])
            return step * self.timestep # ns

    def kill(self) -> None:
        """Kill the job."""
        if not self.job:
            raise ValueError("Stage has no job object. Cannot kill job.")
        if self.job in self.virtual_queue.queue:
            self._logger.info(f"Killing job {self.job}")
            self.virtual_queue.kill(self.job)

    def _set_n_cycles(self, n_cycles: int) -> None:
        """
        Set the number of cycles in the SOMD config file.

        Parameters
        ----------
        n_cycles : int
            Number of cycles to set in the config file.

        Returns
        -------
        None
        """
        # Find the line with n_cycles and replace
        with open(_os.path.join(self.input_dir, "somd.cfg"), "r") as ifile:
            lines=ifile.readlines()
            for i, line in enumerate(lines):
                if line.startswith("ncycles ="):
                    lines[i]="ncycles = " + str(n_cycles) + "\n"
                    break

        # Now write the new file
        with open(_os.path.join(self.input_dir, "somd.cfg"), "w+") as ofile:
            for line in lines:
                ofile.write(line)

    def read_gradients(self, equilibrated_only:bool = False, endstate: bool = False) -> _Tuple[_np.ndarray, _np.ndarray]:
        """
        Read the gradients from the output file. These can be either the infiniesimal gradients
        at the given value of lambda, or the differences in energy between the end state 
        Hamiltonians.

        Parameters
        ----------
        equilibrated_only : bool, Optional, default: False
            Whether to read the gradients from the equilibrated region of the simulation (True)
            or the whole simulation (False).
        endstate : bool, Optional, default: False
            Whether to return the difference in energy between the end state Hamiltonians (True)
            or the infiniesimal gradients at the given value of lambda (False).

        Returns
        -------
        times : np.ndarray
            Array of times, in ns.
        grads : np.ndarray
            Array of gradients, in kcal/mol.
        """
        # Read the output file
        if equilibrated_only:
            with open(_os.path.join(self.output_dir, "simfile_equilibrated.dat"), "r") as ifile:
                lines=ifile.readlines()
        else:
            with open(_os.path.join(self.output_dir, "simfile.dat"), "r") as ifile:
                lines=ifile.readlines()

        steps=[]
        grads=[]

        for line in lines:
            vals=line.split()
            if not line.startswith("#"):
                step=int(vals[0].strip())
                if not endstate: #  Return the infinitesimal gradients
                    grad=float(vals[2].strip())
                else: # Return the difference in energy between the end state Hamiltonians
                    energy_start = float(vals[5].strip())
                    energy_end = float(vals[-1].strip())
                    grad=energy_end - energy_start
                steps.append(step)
                grads.append(grad)

        times=[x * self.timestep for x in steps]  # Timestep already in ns

        times_arr=_np.array(times)
        grads_arr=_np.array(grads)

        return times_arr, grads_arr

    def update_paths(self, old_sub_path: str, new_sub_path: str) -> None:
        """ 
        Replace the old sub-path with the new sub-path in the base, input, and output directory
        paths. Also update the slurm file base and the paths in the simfile.

        Parameters
        ----------
        old_sub_path : str
            The old sub-path to replace.
        new_sub_path : str
            The new sub-path to replace the old sub-path with.
        """
        super().update_paths(old_sub_path, new_sub_path)

        # Also need to update the slurm file base and the paths in the simfile
        self.simfile_path = _os.path.join(self.base_dir, "somd.cfg")
        if self.slurm_file_base:
            self.slurm_file_base = self.slurm_file_base.replace(old_sub_path, new_sub_path)

        # Now we can easily change the paths in the simfile
        input_paths = {"morphfile": "somd.pert",
                       "topfile": "somd.prm7",
                       "crdfile": "somd.rst7"}
        for option, name in input_paths.items():
            _write_simfile_option(self.simfile_path, option, _os.path.join(self.input_dir, name))

    def set_simfile_option(self, option: str, value: str) -> None:
        """Set the value of an option in the simulation configuration file."""
        _write_simfile_option(self.simfile_path, option, value)

    def analyse(self) -> None:
        raise NotImplementedError("Analysis cannot be performed for a single simulation")

    @property
    def equil_time(self) -> None:
        raise NotImplementedError("Equilibration time is not determined for a single simulation, only "
                                  "for an ensemble of simulations within a lambda window.")
    @property
    def equilibrated(self) -> None:
        raise NotImplementedError("Equilibration is not detected at the level of single simulations, only "
                                  "for an ensemble of simulations within a lambda window.")

    def analyse_convergence(self) -> None:
        raise(NotImplementedError("Convergence analysis is not performed for a single simulation, only "
                                  " at the level of a stage or above."))
