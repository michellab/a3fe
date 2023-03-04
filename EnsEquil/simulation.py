"""Functionality to run a single SOMD simulation."""

from decimal import Decimal as _Decimal
import os as _os
import logging as _logging
import numpy as _np
import subprocess as _subprocess
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

from ._utils import Job as _Job, VirtualQueue as _VirtualQueue

class Simulation():
    """Class to store information about a single SOMD simulation."""

    required_input_files=["run_somd.sh",
                          "somd.cfg",
                          "somd.prm7",
                          "somd.rst7",
                          "morph.pert"]

    def __init__(self, lam: float, run_no: int,
                 virtual_queue: _VirtualQueue,
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
        input_dir : str, Optional, default: None
            Path to directory containing input files for the simulation. If None, this
            will be set to "current_working_directory/input".
        output_dir : str, Optional, default: None
            Path to directory to store output files from the simulation. If None, this
            will be set to "current_working_directory/output".
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            Ensemble object and its child objects.

        Returns
        -------
        None
        """
        self.lam=lam
        self.virtual_queue=virtual_queue
        self.run_no=run_no
        if input_dir is None:
            input_dir = _os.path.join(_os.getcwd(), "input")
        self.input_dir = input_dir
        if output_dir is None:
            output_dir = _os.path.join(_os.getcwd(), "output")
        self.output_dir = output_dir
        # Check that the input directory contains the required files
        self._validate_input()
        self.output_dir=_os.path.abspath(output_dir)
        self.job: _Optional[_Job]=None
        self._running: bool=False
        self.output_subdir: str=output_dir + "/lambda_" + f"{lam:.3f}" + "/run_" + str(run_no).zfill(2)
        self.tot_simtime: float=0  # ns
        # Now read useful parameters from the simulation file options
        self._add_attributes_from_simfile()

        # Create the output subdirectory
        _subprocess.call(["mkdir", "-p", self.output_subdir])
        # Create a soft link to the input dir to simplify running simulations
        _subprocess.call(["cp", "-r", self.input_dir, self.output_subdir + "/input"])

        # Set up logging
        self._logger=_logging.getLogger(str(self))
        # For the file handler, we want to log everything
        self._logger.setLevel(_logging.DEBUG)
        file_handler=_logging.FileHandler(f"{self.output_subdir}/simulation.log")
        file_handler.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self._logger.addHandler(file_handler)
        # For the stream handler, we want the user-specified level
        stream_handler=_logging.StreamHandler()
        stream_handler.setFormatter(_logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        stream_handler.setLevel(stream_log_level)
        self._logger.addHandler(stream_handler)

    def __str__(self) -> str:
        return f"Simulation (lam={self.lam}, run_no={self.run_no})"

    @ property
    def running(self) -> bool:
        """
        Check if the simulation is still running,
        and update the running attribute accordingly.

        Returns
        -------
        self._running : bool
            True if the simulation is still running, False otherwise.
        """
        # Get job ids of currently running jobs - but note that the queue is updated at the
        # Ensemble level
        if self.job in self.virtual_queue.queue:
            self._running=True
            self._logger.info(f"Still running")
        else:
            self._running=False
            self._logger.info(f"Finished")

        return self._running

    @ running.setter
    def running(self, value: bool) -> None:
        self._running=value

    def _validate_input(self) -> None:
        """ Check that the required input files are present. """

        # Check that the input directory exists
        if not _os.path.isdir(self.input_dir):
            raise FileNotFoundError("Input directory does not exist.")

        # Check that the required input files are present
        for file in Simulation.required_input_files:
            if not _os.path.isfile(self.input_dir + "/" + file):
                raise FileNotFoundError("Required input file " + file + " not found.")

    def _add_attributes_from_simfile(self) -> None:
        """
        Read the SOMD simulation option file and
        add useful attributes to the Simulation object.

        Returns
        -------
        time_per_cycle : int
            Time per cycle, in ns.
        """

        timestep=None  # ns
        nmoves=None  # number of moves per cycle
        nrg_freq=None  # number of timesteps between energy calculations
        with open(self.input_dir + "/somd.cfg", "r") as ifile:
            lines=ifile.readlines()
            for line in lines:
                if line.startswith("timestep ="):
                    timestep=float(line.split("=")[1].split()[0])
                if line.startswith("nmoves ="):
                    nmoves=float(line.split("=")[1])
                if line.startswith("energy frequency ="):
                    nrg_freq=float(line.split("=")[1])

        if timestep is None or nmoves is None or nrg_freq is None:
            raise ValueError("Could not find timestep or nmoves in somd.cfg.")

        self.timestep=timestep / 1_000_000  # fs to ns
        self.nrg_freq=nrg_freq
        self.time_per_cycle=timestep * nmoves / 1_000_000  # fs to ns

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
        cmd=f"--chdir {self.output_subdir} {self.output_subdir}/input/run_somd.sh {self.lam}"
        self.job=self.virtual_queue.submit(cmd)
        self.running=True
        self.tot_simtime += duration
        self._logger.info(f"Submitted with job {self.job}")

    def kill(self) -> None:
        """Kill the job."""
        if not self.job:
            raise ValueError("No job found. Cannot kill job.")
        self._logger.info(f"Killing job {self.job}")
        self.virtual_queue.kill(self.job)
        self.running=False

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
        with open(self.output_subdir + "/input/somd.cfg", "r") as ifile:
            lines=ifile.readlines()
            for i, line in enumerate(lines):
                if line.startswith("ncycles ="):
                    lines[i]="ncycles = " + str(n_cycles) + "\n"
                    break

        # Now write the new file
        with open(self.output_subdir + "/input/somd.cfg", "w+") as ofile:
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
            with open(self.output_subdir + "/simfile_equilibrated.dat", "r") as ifile:
                lines=ifile.readlines()
        else:
            with open(self.output_subdir + "/simfile.dat", "r") as ifile:
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

    def _update_log(self) -> None:
        """ Write the status of the simulation to a log file. """

        self._logger.debug("##############################################")
        for var in vars(self):
            self._logger.debug(f"{var}: {getattr(self, var)} ")
        self._logger.debug("##############################################")
