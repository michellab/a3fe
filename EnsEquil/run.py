"""Functions for running free energy calculations with SOMD with automated
 equilibration detection based on an ensemble of simulations."""

import subprocess as _subprocess
from decimal import Decimal as _Decimal
import os as _os
import threading as _threading
import logging as _logging
import matplotlib.pyplot as _plt
import numpy as _np
import pickle as _pkl
import scipy.stats as _stats
from time import sleep as _sleep
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

from .equil_detection import check_equil_block_gradient as _check_equil_block_gradient
from .equil_detection import check_equil_chodera as _check_equil_chodera
from ._utils import Job as _Job, VirtualQueue as _VirtualQueue, read_mbar_outfile as _read_mbar_outfile
from .plot import plot_lam_gradient as _plot_lam_gradient, plot_lam_gradient_hists as _plot_lam_gradient_hists


class Ensemble():
    """
    Class to hold and manipulate an ensemble of SOMD simulations.
    """

    def __init__(self, 
                 block_size: float = 1,
                 equil_detection: str = "block_gradient",
                 gradient_threshold: _Optional[float] = None,
                 ensemble_size: int = 5,
                 input_dir: str = "./input",
                 output_dir: str = "./output",
                 stream_log_level: int = _logging.INFO) -> None:
        """
        Initialise an ensemble of SOMD simulations. If ensemble.pkl exists in the
        output directory, the ensemble will be loaded from this file and any arguments
        supplied will be overwritten.

        Parameters
        ----------
        block_size : float, Optional, default: 1
            Size of blocks to use for equilibration detection, in ns.
        equil_detection : str, Optional, default: "block_gradient"
            Method to use for equilibration detection. Options are:
            - "block_gradient": Use the gradient of the block averages to detect equilibration.
            - "chodera": Use Chodera's method to detect equilibration.
        gradient_threshold : float, Optional, default: None
            The threshold for the absolute value of the gradient, in kcal mol-1 ns-1,
            below which the simulation is considered equilibrated. If None, no theshold is
            set and the simulation is equilibrated when the gradient passes through 0. A 
            sensible value appears to be 0.5 kcal mol-1 ns-1.
        ensemble_size : int, Optional, default: 5
            Number of simulations to run in the ensemble.
        input_dir : str, Optional, default: "./input"
            Path to directory containing input files for the simulations.
        output_dir : str, Optional, default: "./output"
            Path to directory to store output files from the simulations.
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            Ensemble object and its child objects.

        Returns
        -------
        None
        """
        # Check if we are starting from a previous simulation
        if _os.path.isfile(f"{output_dir}/ensemble.pkl"):
            print("Loading previous ensemble. Any arguments will be overwritten...")
            with open(f"{output_dir}/ensemble.pkl", "rb") as file:
                self.__dict__ = _pkl.load(file)
            # TODO: Check if the simulations are still running and continue if so. Ensure that the
            # total simulation times are correct by checking the sim files.

        else:  # No pkl file to resume from
            print("Creating new ensemble...")

            self.block_size = block_size
            self.ensemble_size = ensemble_size
            self.input_dir = input_dir
            self.output_dir = output_dir
            self._running: bool = False
            self.run_thread: _Optional[_threading.Thread] = None
            # Set boolean to allow us to kill the thread
            self.kill_thread: bool = False
            self.lam_vals: _List[float] = self._get_lam_vals()
            self.lam_windows: _List[LamWindow] = []
            self.running_wins: _List[LamWindow] = []
            # Would be created by lam windows anyway, but is needed for queue log
            _os.mkdir(output_dir)
            self.virtual_queue = _VirtualQueue(log_dir=output_dir)

            # Creating lambda window objects sets up required input directories
            for lam in self.lam_vals:
                self.lam_windows.append(LamWindow(lam, self.virtual_queue,
                                                  self.block_size,
                                                  equil_detection,
                                                  gradient_threshold,
                                                  self.ensemble_size, self.input_dir,
                                                  output_dir, stream_log_level,
                                                  ))
            # Set up logging
            self._logger = _logging.getLogger(str(self))
            # For the file handler, we want to log everything
            self._logger.setLevel(_logging.DEBUG)
            file_handler = _logging.FileHandler(f"{output_dir}/ensemble.log")
            file_handler.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(file_handler)
            # For the stream handler, we want to log at the user-specified level
            stream_handler = _logging.StreamHandler()
            stream_handler.setFormatter(_logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
            stream_handler.setLevel(stream_log_level)
            self._logger.addHandler(stream_handler)

            # Save state
            self._dump()

    def __str__(self) -> str:
        return f"Ensemble (repeats = {self.ensemble_size}, no windows = {len(self.lam_vals)})"

    def run(self) -> None:
        """Run the ensemble of simulations with adaptive equilibration detection,
        and perform analysis once finished."""
        # Run in the background with threading so that user can continuously check
        # the status of the Ensemble object
        self.run_thread = _threading.Thread(target=self._run_without_threading, name="Ensemble")
        self.run_thread.start()
        self.running = True

    def kill(self) -> None:
        """Kill all running simulations."""
        # Stop the run loop
        self.kill_thread = True

        self._logger.info("Killing all lambda windows")
        for win in self.lam_windows:
            win.kill()
        self.running = False

    @property
    def running(self) -> bool:
        """Return True if the ensemble is currently running."""
        if self.run_thread is not None:
            self._running = self.run_thread.is_alive()
        else:
            self._running = False
        return self._running

    @running.setter
    def running(self, value: bool) -> None:
        """Set the running attribute."""
        self._running = value

    def _run_without_threading(self) -> None:
        """ Run the ensemble of simulations with adaptive equilibration detection,
        and perform analysis once finished. This function is called by run() with threading,
        so that the function can be run in the background and the user can continuously
        check the status of the Ensemble object."""

        # Reset self.kill_thread so we can restart after killing
        self.kill_thread = False

        # Run initial SOMD simulations
        self._logger.info("Starting ensemble of simulations...")
        for win in self.lam_windows:
            # Add buffer of 1 block_size to give chance for the equilibration to be detected.
            win.run(3 * self.block_size)
            win._update_log()
            self._dump()

        # Periodically check the simulations and analyse/ resubmit as necessary
        self.running_wins = self.lam_windows
        self._dump()
        while self.running_wins:
            _sleep(20 * 1)  # Check every 20 seconds
            # Check if we've requested to kill the thread
            if self.kill_thread:
                self._logger.info(f"Kill thread requested: exiting run loop")
                return
            # Update the queue before checking the simulations
            self.virtual_queue.update()
            for win in self.running_wins:
                # Check if the window has now finished - calling win.running updates the win._running attribute
                if not win.running:
                    self._logger.info(f"{win} has finished at {win.tot_simtime:.3f} ns")
                    # Check if the simulation has equilibrated and if not, resubmit
                    if win.equilibrated:
                        self._logger.info(f"{win} has equilibrated at {win.equil_time:.3f} ns")
                    else:
                        self._logger.info(f"{win} has not equilibrated. Resubmitting for {self.block_size:.3f} ns")
                        win.run(self.block_size)

                    # Write status after checking for running and equilibration, as this updates the
                    # _running and _equilibrated attributes
                    win._update_log()
                    self._dump()

            self.running_wins = [win for win in self.running_wins if win.running]

        # All simulations are now finished, so perform final analysis
        self.analyse()

    def _get_lam_vals(self) -> _List[float]:
        """
        Return list of lambda values for the simulations,
        based on the configuration file.

        Returns
        -------
        lam_vals : List[float]
            List of lambda values for the simulations.
        """

        # Read number of lambda windows from input file
        lam_vals_str = []
        with open(self.input_dir + "/sim.cfg", "r") as ifile:
            lines = ifile.readlines()
            for line in lines:
                if line.startswith("lambda array ="):
                    lam_vals_str = line.split("=")[1].split(",")
                    break
        lam_vals = [float(lam) for lam in lam_vals_str]

        return lam_vals

    def analyse(self, get_frnrg:bool = True) -> None:
        """ Analyse the results of the ensemble of simulations. Requires that
        all lambda windows have equilibrated.
          
        Parameters
        ----------
        get_frnrg : bool, optional, default=True
            If True, the free energy will be calculated with MBAR, otherwise
            this will be skipped.
        
        Returns
        -------
        None
        """

        # Check that all simulations have equilibrated
        for win in self.lam_windows:
            # Avoid checking win.equilibrated as this causes expensive equilibration detection to be run
            if not win._equilibrated:
                raise RuntimeError("Not all lambda windows have equilibrated. Analysis cannot be performed.")
            if win.equil_time is None:
                raise RuntimeError("Despite equilibration being detected, no equilibration time was found.")

        if get_frnrg:
            # Remove unequilibrated data from the equilibrated output directory
            for win in self.lam_windows:
                equil_time = win.equil_time
                # Minus 1 because first energy is only written after the first nrg_freq steps
                equil_index = int(equil_time / (win.sims[0].timestep * win.sims[0].nrg_freq)) - 1
                for sim in win.sims:
                    in_simfile = sim.output_subdir + "/simfile.dat"
                    out_simfile = sim.output_subdir + "/simfile_equilibrated.dat"
                    self._write_equilibrated_data(in_simfile, out_simfile, equil_index)

            # Analyse data with MBAR and compute uncertainties
            output_dir = self.output_dir
            ensemble_size = self.ensemble_size
            # This is nasty - assumes naming always the same
            mbar_out_files = []
            for run in range(1, ensemble_size + 1):
                outfile = f"{self.output_dir}/freenrg-MBAR-run_{str(run).zfill(2)}.dat"
                mbar_out_files.append(outfile)
                with open(outfile, "w") as ofile:
                    _subprocess.run(["/home/finlayclark/sire.app/bin/analyse_freenrg",
                                    "mbar", "-i", f"{output_dir}/lambda*/run_{str(run).zfill(2)}/simfile_equilibrated.dat",
                                    "-p", "100", "--overlap", "--temperature",
                                    "298.0"], stdout=ofile)

            # Compute overall uncertainty
            free_energies = _np.array([_read_mbar_outfile(ofile)[0] for ofile in mbar_out_files])  # Ignore MBAR error
            mean_free_energy = _np.mean(free_energies)
            # Gaussian 95 % C.I.
            conf_int = _stats.t.interval(0.95,
                                        len(free_energies)-1,
                                        mean_free_energy,
                                        scale=_stats.sem(free_energies))[1] - mean_free_energy  # 95 % C.I.

            # Write overall MBAR stats to file
            with open(f"{self.output_dir}/overall_stats.dat", "a") as ofile:
                if get_frnrg:
                    ofile.write("###################################### Free Energies ########################################\n")
                    ofile.write(f"Mean free energy: {mean_free_energy: .3f} + /- {conf_int:.3f} kcal/mol\n")
                    for i in range(5):
                        ofile.write(f"Free energy from run {i+1}: {free_energies[i]: .3f} kcal/mol\n")
                    ofile.write("Errors are 95 % C.I.s based on the assumption of a Gaussian distribution of free energies\n")

        # Plot free variance of gradients for all data
        # Plot mean
        _plot_lam_gradient(lams=self.lam_windows, outdir=self.output_dir, plot_mean=True, plot_variance=False, equilibrated=False, inter_var=True, intra_var=True)
        # Plot variance - intra
        _plot_lam_gradient(lams=self.lam_windows, outdir=self.output_dir, plot_mean=False, plot_variance=True, equilibrated=False, inter_var=False, intra_var=True)
        # Plot variance - inter
        _plot_lam_gradient(lams=self.lam_windows, outdir=self.output_dir, plot_mean=False, plot_variance=True, equilibrated=False, inter_var=True, intra_var=False)
        # Plot variance - total
        _plot_lam_gradient(lams=self.lam_windows, outdir=self.output_dir, plot_mean=False, plot_variance=True, equilibrated=False, inter_var=True, intra_var=True)
        # Plot histograms
        _plot_lam_gradient_hists(lams=self.lam_windows, outdir=self.output_dir, equilibrated=True)

        # Make plots of equilibration time
        fig, ax=_plt.subplots(figsize=(8, 6))
        # Plot the total time simulated per simulation, so we can see how efficient
        # the protocol is
        ax.bar([win.lam for win in self.lam_windows],
                [win.sims[0].tot_simtime for win in self.lam_windows],  # All sims at given lam run for same time
                width=0.02, edgecolor='black', label="Total time simulated per simulation")
        # Now plot the equilibration time
        ax.bar([win.lam for win in self.lam_windows],
               [win.equil_time for win in self.lam_windows],
               width=0.02, edgecolor='black', label="Equilibration time per simulation")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Time (ns)")
        fig.legend()
        fig.savefig(f"{self.output_dir}/equil_times", dpi=300,
                    bbox_inches='tight', facecolor='white', transparent=False)

        # Write out stats
        with open(f"{self.output_dir}/overall_stats.dat", "a") as ofile:
            for win in self.lam_windows:
                ofile.write(f"Equilibration time for lambda = {win.lam}: {win.equil_time:.3f} ns per simulation\n")
                ofile.write(f"Total time simulated for lambda = {win.lam}: {win.sims[0].tot_simtime:.3f} ns per simulation\n")


        # TODO: Make convergence plots (which should be flat)

    def _write_equilibrated_data(self, in_simfile: str,
                                 out_simfile: str,
                                 equil_index: int) -> None:
        """
        Remove unequilibrated data from a simulation file and write a new
        file containing only the equilibrated data.

        Parameters
        ----------
        in_simfile : str
            Path to simulation file.
        out_simfile : str
            Path to new simulation file, containing only the equilibrated data
            from the original file.
        equil_index : int
            Index of the first equilibrated frame, given by
            equil_time / (timestep * nrg_freq)

        Returns
        -------
        None
        """
        with open(in_simfile, "r") as ifile:
            lines=ifile.readlines()

        # Figure out how many lines come before the data
        non_data_lines=0
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
            for line in lines[equil_index + non_data_lines:]:
                ofile.write(line)

    def _update_log(self) -> None:
        """ Update the status log file with the current status of the ensemble. """
        self._logger.debug("##############################################")
        for var in vars(self):
            self._logger.debug(f"{var}: {getattr(self, var)}")
        self._logger.debug("##############################################")

    def _dump(self) -> None:
        """ Dump the current state of the ensemble to a pickle file. Specifically,
         pickle self.__dict__ with self.run_thread = None, as _thread_lock objects
         can't be pickled.   """
        temp_dict={key: val for key, val in self.__dict__.items() if key != "run_thread"}
        temp_dict["run_thread"]=None
        with open(f"{self.output_dir}/ensemble.pkl", "wb") as ofile:
            _pkl.dump(temp_dict, ofile)


class LamWindow():
    """A class to hold and manipulate a set of SOMD simulations at a given lambda value."""

    equil_detection_methods={"block_gradient": _check_equil_block_gradient,
                               "chodera": _check_equil_chodera}

    def __init__(self, lam: float,
                 virtual_queue: _VirtualQueue,
                 block_size: float=1,
                 equil_detection: str="block_gradient",
                 gradient_threshold: _Optional[float]=None,
                 ensemble_size: int=5, input_dir: str="./input",
                 output_dir: str="./output",
                 stream_log_level: int=_logging.INFO) -> None:
        """
        Initialise a LamWindow object.

        Parameters
        ----------
        lam : float
            Lambda value for the simulation.
        virtual_queue : VirtualQueue
            VirtualQueue object to use for submitting jobs.
        block_size : float, Optional, default: 1
            Size of the blocks to use for equilibration detection,
            in ns.
        equil_detection : str, Optional, default: "block_gradient"
            Method to use for equilibration detection. Options are:
            - "block_gradient": Use the gradient of the block averages to detect equilibration.
            - "chodera": Use Chodera's method to detect equilibration.
        gradient_threshold : float, Optional, default: None
            The threshold for the absolute value of the gradient, in kcal mol-1 ns-1,
            below which the simulation is considered equilibrated. If None, no theshold is
            set and the simulation is equilibrated when the gradient passes through 0. A 
            sensible value appears to be 0.5 kcal mol-1 ns-1.
        ensemble_size : int, Optional, default: 5
            Number of simulations to run at this lambda value.
        input_dir : str, Optional, default: "./input"
            Path to the input directory.
        output_dir : str
            Path to the output directory.
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            Ensemble object and its child objects.

        Returns
        -------
        None
        """
        self.lam=lam
        self.virtual_queue=virtual_queue
        self.block_size=block_size
        self.ensemble_size=ensemble_size
        if equil_detection not in self.equil_detection_methods:
            raise ValueError(f"Equilibration detection method {equil_detection} not recognised.")
        # Need to pass self object to equilibration detection function
        self.check_equil=self.equil_detection_methods[equil_detection]
        # Ensure that we do not try to use a gradient threshold with a method that does not use it
        if equil_detection != "block_gradient" and gradient_threshold is not None:
            raise ValueError("Gradient threshold can only be set for block gradient method.")
        self.gradient_threshold=gradient_threshold
        self.gradient_threshold=gradient_threshold
        self.input_dir=input_dir
        self.output_dir=output_dir
        self._equilibrated: bool=False
        self.equil_time: _Optional[float]=None
        self._running: bool=False
        self.tot_simtime: float=0  # ns

        # Create the required simulations for this lambda value
        self.sims=[]
        for i in range(1, ensemble_size + 1):
            sim=Simulation(lam, i, self.virtual_queue, input_dir, output_dir, stream_log_level)
            self.sims.append(sim)

        # Set up logging
        self._logger=_logging.getLogger(str(self))
        # For the file handler, we want to log everything
        self._logger.setLevel(_logging.DEBUG)
        file_handler=_logging.FileHandler(f"{self.output_dir}/lambda_{self.lam:.3f}/window.log")
        file_handler.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self._logger.addHandler(file_handler)
        # For the stream handler, we only want to log at the user-specified level
        stream_handler=_logging.StreamHandler()
        stream_handler.setFormatter(_logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        stream_handler.setLevel(stream_log_level)
        self._logger.addHandler(stream_handler)

    def __str__(self) -> str:
        return f"LamWindow (lam={self.lam:.3f})"

    def run(self, duration: float=2.5) -> None:
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
            self.tot_simtime += duration

        self._running=True

    def kill(self) -> None:
        """ Kill all simulations at the lambda value. """
        self._logger.info("Killing all simulations")
        for sim in self.sims:
            sim.kill()
        self.running=False

    @ property
    def running(self) -> bool:
        """
        Check if all the simulations at the lambda window are still running
        and update the running attribute accordingly.

        Returns
        -------
        self._running : bool
            True if the simulation is still running, False otherwise.
        """
        all_finished=True
        for sim in self.sims:
            if sim.running:
                all_finished=False
                break
        self._running=not all_finished

        return self._running

    @ running.setter
    def running(self, value: bool) -> None:
        self._running=value

    @ property
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
        self._equilibrated, self.equil_time=self.check_equil(self)
        return self._equilibrated

    @ equilibrated.setter
    def equilibrated(self, value: bool) -> None:
        self._equilibrated=value

    def _get_rolling_average(self, data: _np.ndarray, idx_block_size: int) -> _np.ndarray:
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
        rolling_av=_np.full(len(data), _np.nan)

        for i in range(len(data)):
            if i < idx_block_size:
                continue
            else:
                block_av=_np.mean(data[i - idx_block_size: i])
                rolling_av[i]=block_av

        return rolling_av

    def _update_log(self) -> None:
        """Write the status of the lambda window and all simulations to their log files."""
        self._logger.debug("##############################################")
        for var in vars(self):
            self._logger.debug(f"{var}: {getattr(self, var)}")
        self._logger.debug("##############################################")

        for sim in self.sims:
            sim._update_log()


class Simulation():
    """Class to store information about a single SOMD simulation."""

    def __init__(self, lam: float, run_no: int,
                 virtual_queue: _VirtualQueue,
                 input_dir: str="./input",
                 output_dir: str="./output",
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
        output_dir : str
            Path to the output directory.
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
        self.input_dir=_os.path.abspath(input_dir)
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
        required_files=["run_somd.sh", "sim.cfg", "system.top", "system.crd", "morph.pert"]
        for file in required_files:
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
        with open(self.input_dir + "/sim.cfg", "r") as ifile:
            lines=ifile.readlines()
            for line in lines:
                if line.startswith("timestep ="):
                    timestep=float(line.split("=")[1].split()[0])
                if line.startswith("nmoves ="):
                    nmoves=float(line.split("=")[1])
                if line.startswith("energy frequency ="):
                    nrg_freq=float(line.split("=")[1])

        if timestep is None or nmoves is None or nrg_freq is None:
            raise ValueError("Could not find timestep or nmoves in sim.cfg.")

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
        with open(self.output_subdir + "/input/sim.cfg", "r") as ifile:
            lines=ifile.readlines()
            for i, line in enumerate(lines):
                if line.startswith("ncycles ="):
                    lines[i]="ncycles = " + str(n_cycles) + "\n"
                    break

        # Now write the new file
        with open(self.output_subdir + "/input/sim.cfg", "w+") as ofile:
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
