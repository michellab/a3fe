"""Functions for running free energy calculations with SOMD with automated
 equilibration detection based on an ensemble of simulations."""

from enum import Enum as _Enum
import os as _os
import threading as _threading
import logging as _logging
from multiprocessing import Pool as _Pool
import numpy as _np
import pathlib as _pathlib
import pickle as _pkl
import subprocess as _subprocess
import scipy.stats as _stats
from time import sleep as _sleep
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional, Union as _Union

from ._utils import Job as _Job, VirtualQueue as _VirtualQueue, read_mbar_result as _read_mbar_result, _get_simtime
from .lambda_window import LamWindow as _LamWindow
from .plot import (
    plot_gradient_stats as _plot_gradient_stats, 
    plot_gradient_hists as _plot_gradient_hists, 
    plot_equilibration_time as _plot_equilibration_time,
    plot_overlap_mats as _plot_overlap_mats,
)
from .process_grads import GradientData as _GradientData
from ._simfile import write_simfile_option as _write_simfile_option
from ._simulation_runner import SimulationRunner as _SimulationRunner

class StageType(_Enum):
    """Enumeration of the types of stage."""
    RESTRAIN = 1
    DISCHARGE = 2
    VANISH = 3

    @property
    def bss_perturbation_type(self) -> str:
        """Return the corresponding BSS perturbation type."""
        if self == StageType.RESTRAIN:
            return "restraint"
        elif self == StageType.DISCHARGE:
            return "discharge_soft"
        elif self == StageType.VANISH:
            return "vanish_soft"
        else:
            raise ValueError("Unknown stage type.")
        

class Stage(_SimulationRunner):
    """
    Class to hold and manipulate an ensemble of SOMD simulations for a
    single stage of a calculation.
    """

    def __init__(self, 
                 stage_type: StageType,
                 block_size: float = 1,
                 equil_detection: str = "block_gradient",
                 gradient_threshold: _Optional[float] = None,
                 ensemble_size: int = 5,
                 lambda_values: _Optional[_List[float]] = None,
                 base_dir: _Optional[str] = None,
                 input_dir: _Optional[str] = None,
                 output_dir: _Optional[str] = None,
                 stream_log_level: int = _logging.INFO) -> None:
        """
        Initialise an ensemble of SOMD simulations, constituting the Stage. If Stage.pkl exists in the
        output directory, the Stage will be loaded from this file and any arguments
        supplied will be overwritten.

        Parameters
        ----------
        stage_type : StageType
            The type of stage.
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
        lambda_values : List[float], Optional, default: None
            List of lambda values to use for the simulations. If None, the lambda values
            will be read from the simfile.
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

        Returns
        -------
        None
        """
        # Set the stage type first, as this is required for __str__,
        # and threrefore the super().__init__ call
        self.stage_type = stage_type

        super().__init__(base_dir=base_dir,
                         input_dir=input_dir,
                         output_dir=output_dir,
                         stream_log_level=stream_log_level,
                         ensemble_size=ensemble_size)

        if not self.loaded_from_pickle:
            if lambda_values is not None:
                self.lam_vals = lambda_values
            else:
                self.lam_vals = self._get_lam_vals()
            self.block_size = block_size
            self.equil_detection = equil_detection
            self.gradient_threshold = gradient_threshold
            self._running: bool = False
            self.run_thread: _Optional[_threading.Thread] = None
            # Set boolean to allow us to kill the thread
            self.kill_thread: bool = False
            self.lam_windows: _List[_LamWindow] = []
            self.running_wins: _List[_LamWindow] = []
            self.virtual_queue = _VirtualQueue(log_dir=self.base_dir)
            # Creating lambda window objects sets up required input directories
            for lam_val in self.lam_vals:
                lam_base_dir = _os.path.join(self.output_dir, f"lambda_{lam_val:.3f}")
                self.lam_windows.append(_LamWindow(lam=lam_val, 
                                                   virtual_queue=self.virtual_queue,
                                                    block_size=self.block_size,
                                                    equil_detection=self.equil_detection,
                                                    gradient_threshold=self.gradient_threshold,
                                                    ensemble_size=self.ensemble_size,
                                                    base_dir=lam_base_dir,
                                                    input_dir=self.input_dir,
                                                    stream_log_level=self.stream_log_level
                                                  )
                                       )

            # Point self._sub_sim_runners at the lambda windows
            self._sub_sim_runners = self.lam_windows

            # Save the state and update log
            self._update_log()
            self._dump()

    def __str__(self) -> str:
        return f"Stage (type = {self.stage_type.name.lower()})"

    def run(self, adaptive:bool=True, runtime:_Optional[float]=None) -> None:
        """ Run the ensemble of simulations constituting the stage (optionally with adaptive 
        equilibration detection), and, if using adaptive equilibration detection, perform 
        analysis once finished. 

        Parameters
        ----------
        adaptive : bool, Optional, default: True
            If True, the stage will run until the simulations are equilibrated and perform analysis afterwards.
            If False, the stage will run for the specified runtime and analysis will not be performed.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this number of nanoseconds. 

        Returns
        -------
        None
        """
        if not adaptive and runtime is None:
            raise ValueError("If adaptive equilibration detection is disabled, a runtime must be supplied.")
        if adaptive and runtime is not None:
            raise ValueError("If adaptive equilibration detection is enabled, a runtime cannot be supplied.")

        # Run in the background with threading so that user can continuously check
        # the status of the Stage object
        self.run_thread = _threading.Thread(target=self._run_without_threading, args=(adaptive, runtime), name=str(self))
        self.run_thread.start()

    def kill(self) -> None:
        """Kill all running simulations."""
        # Stop the run loop
        if self.running:
            self._logger.info("Killing all lambda windows")
            self.kill_thread = True
            for win in self.lam_windows:
                win.kill()

    def _run_without_threading(self, adaptive:bool=True, runtime:_Optional[float]=None) -> None:
        """ Run the ensemble of simulations constituting the stage (optionally with adaptive 
        equilibration detection), and, if using adaptive equilibration detection, perform 
        analysis once finished.  This function is called by run() with threading, so that
        the function can be run in the background and the user can continuously check 
        the status of the Stage object.
        
        Parameters
        ----------
        adaptive : bool, Optional, default: True
            If True, the stage will run until the simulations are equilibrated and perform analysis afterwards.
            If False, the stage will run for the specified runtime and analysis will not be performed.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this number of nanoseconds. 

        Returns
        -------
        None
        """
        # Use stage context manager to ensure that the stage is killed if an exception is raised
        with StageContextManager([self]):
            try:
                # Reset self.kill_thread so we can restart after killing
                self.kill_thread = False

                if not adaptive and runtime is None:
                    raise ValueError("If adaptive equilibration detection is disabled, a runtime must be supplied.")
                if adaptive and runtime is not None:
                    raise ValueError("If adaptive equilibration detection is enabled, a runtime cannot be supplied.")

                if not adaptive:
                    self._logger.info(f"Starting {self}. Adaptive equilibration = {adaptive}...")
                elif adaptive:
                    self._logger.info(f"Starting {self}. Adaptive equilibration = {adaptive}...")
                    if runtime is None:
                        runtime = 3 * self.block_size

                # Run initial SOMD simulations
                for win in self.lam_windows:
                    # Add buffer of 1 block_size to give chance for the equilibration to be detected.
                    win.run(runtime) # type: ignore
                    win._update_log()
                    self._dump()

                # Periodically check the simulations and analyse/ resubmit as necessary
                # Copy to ensure that we don't modify self.lam_windows when updating self.running_wins
                self.running_wins = self.lam_windows.copy() 
                self._dump()
                while self.running_wins:
                    _sleep(60 * 1)  # Check every 20 seconds
                    # Check if we've requested to kill the thread
                    if self.kill_thread:
                        self._logger.info(f"Kill thread requested: exiting run loop")
                        return
                    # Update the queue before checking the simulations
                    self.virtual_queue.update()
                    for win in self.running_wins:
                        # Check if the window has now finished - calling win.running updates the win._running attribute
                        if not win.running:
                            # If we are in adaptive mode, check if the simulation has equilibrated and if not, resubmit
                            if adaptive:
                                if win.equilibrated:
                                    self._logger.info(f"{win} has equilibrated at {win.equil_time:.3f} ns")
                                else:
                                    self._logger.info(f"{win} has not equilibrated. Resubmitting for {self.block_size:.3f} ns")
                                    win.run(self.block_size)
                            else: # Not in adaptive mode
                                self._logger.info(f"{win} has finished at {win.tot_simtime:.3f} ns")

                            # Write status after checking for running and equilibration, as this updates the
                            # _running and _equilibrated attributes
                            win._update_log()
                            self._dump()

                    self.running_wins = [win for win in self.running_wins.copy() if win.running]

            except Exception as e:
                self._logger.exception("")
                raise e

        # All simulations are now finished, so perform final analysis
        self._logger.info(f"All simulations in {self} have finished.")

    def get_optimal_lam_vals(self, delta_sem: float = 0.1) -> _np.ndarray:
        """
        Get the optimal lambda values for the stage, based on the 
        integrated SEM, and create plots.

        Parameters
        ----------
        delta_sem : float, default: 0.1
            The desired integrated standard error of the mean of the gradients
            between each lambda value, in kcal mol-1.        
        Returns
        -------
        optimal_lam_vals : np.ndarray
            List of optimal lambda values for the stage.
        """
        self._logger.info("Calculating optimal lambda values...")
        unequilibrated_gradient_data = _GradientData(lam_winds=self.lam_windows, equilibrated=False)
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="mean")
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="intra_run_variance")
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="sem")
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="stat_ineff")
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="integrated_sem")
        _plot_gradient_hists(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir)
        return unequilibrated_gradient_data.calculate_optimal_lam_vals(delta_sem=delta_sem)


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
        with open(self.input_dir + "/somd.cfg", "r") as ifile:
            lines = ifile.readlines()
            for line in lines:
                if line.startswith("lambda array ="):
                    lam_vals_str = line.split("=")[1].split(",")
                    break
        lam_vals = [float(lam) for lam in lam_vals_str]

        return lam_vals

    def analyse(self, get_frnrg:bool = True) -> _Union[_Tuple[float, float], _Tuple[None, None]]:
        r""" Analyse the results of the ensemble of simulations. Requires that
        all lambda windows have equilibrated.
          
        Parameters
        ----------
        get_frnrg : bool, optional, default=True
            If True, the free energy will be calculated with MBAR, otherwise
            this will be skipped.
        
        Returns
        -------
        free_energies : np.ndarray or None
            The free energy changes for the stage for each of the ensemble
            size runs, in kcal mol-1.  If get_frnrg is False, this is None.
        errors : np.ndarray or None
            The MBAR error estimates for the free energy changes for the stage
            for each of the ensemble size runs, in kcal mol-1.  If get_frnrg is
            False, this is None.
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
                    in_simfile = sim.output_dir + "/simfile.dat"
                    out_simfile = sim.output_dir + "/simfile_equilibrated.dat"
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
                    _subprocess.run(["analyse_freenrg",
                                    "mbar", "-i", f"{output_dir}/lambda*/run_{str(run).zfill(2)}/simfile_equilibrated.dat",
                                    "-p", "100", "--overlap", "--temperature", "298.0"], stdout=ofile)

            # Compute overall uncertainty
            free_energies = _np.array([_read_mbar_result(ofile)[0] for ofile in mbar_out_files]) 
            errors = _np.array([_read_mbar_result(ofile)[1] for ofile in mbar_out_files])
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

            # Plot overlap matrices
            _plot_overlap_mats([ofile for ofile in mbar_out_files], self.output_dir)


        # Analyse the gradient data and make plots
        unequilibrated_gradient_data = _GradientData(lam_winds=self.lam_windows, equilibrated=False)
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="mean")
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="intra_run_variance")
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="sem")
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="stat_ineff")
        _plot_gradient_stats(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir, plot_type="integrated_sem")
        _plot_gradient_hists(gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir)

        # Make plots of equilibration time
        _plot_equilibration_time(lam_windows=self.lam_windows, output_dir=self.output_dir)

        # Write out stats
        with open(f"{self.output_dir}/overall_stats.dat", "a") as ofile:
            for win in self.lam_windows:
                ofile.write(f"Equilibration time for lambda = {win.lam}: {win.equil_time:.3f} ns per simulation\n")
                ofile.write(f"Total time simulated for lambda = {win.lam}: {win.sims[0].tot_simtime:.3f} ns per simulation\n")


        # TODO: Make convergence plots (which should be flat)
        # TODO: Plot PMFs

        if get_frnrg:
            self._logger.info(f"Overall free energy changes: {free_energies} kcal mol-1") # type: ignore
            self._logger.info(f"Overall errors: {errors} kcal mol-1") # type: ignore
            return free_energies, errors # type: ignore
        else:
            return None, None

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

    @property
    def tot_simtime(self) -> float:
        f"""The total simulation time in ns for the stage."""
        # Use multiprocessing at the level of stages to speed this us - this is a good place as stages
        # have lots of windows, so we benefit the most from parallelisation here.
        with _Pool() as pool:
            return sum(pool.map(_get_simtime, self._sub_sim_runners))

    def _mv_output(self, save_name: str) -> None:
        """
        Move the output directory to a new location, without
        changing self.output_dir.

        Parameters
        ----------
        save_name : str
            The new name of the old output directory.
        """
        self._logger.info(f"Moving contents of output directory to {save_name}")
        base_dir = _pathlib.Path(self.output_dir).parent.resolve()
        _os.rename(self.output_dir, _os.path.join(base_dir, save_name))

    def update(self, save_name: str = "output_saved") -> None:
        """
        Delete the current set of lamda windows and simulations, and
        create a new set of simulations based on the current state of
        the stage. This is useful if you want to change the number of
        simulations per lambda window, or the number of lambda windows.

        Parameters
        ----------
        save_name : str, default "output_saved"
            The name of the directory to save the old output directory to.
        """
        if self.running:
            raise RuntimeError("Can't update while ensemble is running")
        if _os.path.isdir(self.output_dir):
            self._mv_output(save_name)
        # Update the list of lambda windows in the simfile
        _write_simfile_option(simfile=f"{self.input_dir}/somd.cfg",
                              option="lambda array", 
                              value=", ".join([str(lam) for lam in self.lam_vals]))
        self._logger.info("Deleting old lambda windows...")
        del(self.lam_windows)
        self._logger.info("Creating lambda windows...")
        self.lam_windows = []
        for lam_val in self.lam_vals:
            lam_base_dir = _os.path.join(self.output_dir, f"lambda_{lam_val:.3f}")
            self.lam_windows.append(_LamWindow(lam=lam_val, 
                                                virtual_queue=self.virtual_queue,
                                                block_size=self.block_size,
                                                equil_detection=self.equil_detection,
                                                gradient_threshold=self.gradient_threshold,
                                                ensemble_size=self.ensemble_size,
                                                base_dir=lam_base_dir,
                                                input_dir=self.input_dir,
                                                stream_log_level=self.stream_log_level
                                                )
                                    )

        # Point self._sub_sim_runners to self.lam_windows
        self._sub_sim_runners = self.lam_windows

class StageContextManager():
    """Stage context manager to ensure that all stages are killed when
    the context is exited."""

    def __init__(self, stages: _List[Stage]):
        """ 
        Parameters
        ----------
        stages : List[Stage]
            The stages to kill when the context is exited.
        """
        self.stages = stages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for stage in self.stages:
            stage.kill()
        