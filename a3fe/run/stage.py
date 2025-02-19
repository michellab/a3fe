"""Functions for running free energy calculations with SOMD with automated
equilibration detection based on an ensemble of simulations."""

__all__ = ["Stage"]

import logging as _logging
import os as _os
import pathlib as _pathlib
import threading as _threading
from copy import deepcopy as _deepcopy
from math import ceil as _ceil
from multiprocessing import get_context as _get_context
from time import sleep as _sleep
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

from ..configuration.enums import EngineType as _EngineType
import numpy as _np
import pandas as _pd
import scipy.stats as _stats

from ..analyse.detect_equil import (
    check_equil_multiwindow_gelman_rubin as _check_equil_multiwindow_gelman_rubin,
)
from ..analyse.detect_equil import (
    check_equil_multiwindow_paired_t as _check_equil_multiwindow_paired_t,
)
from ..analyse.detect_equil import (
    dummy_check_equil_multiwindow as _dummy_check_equil_multiwindow,
)
from ..analyse.mbar import collect_mbar_slurm as _collect_mbar_slurm
from ..analyse.mbar import run_mbar as _run_mbar
from ..analyse.mbar import submit_mbar_slurm as _submit_mbar_slurm
from ..analyse.plot import plot_convergence as _plot_convergence
from ..analyse.plot import plot_equilibration_time as _plot_equilibration_time
from ..analyse.plot import plot_gradient_hists as _plot_gradient_hists
from ..analyse.plot import plot_gradient_stats as _plot_gradient_stats
from ..analyse.plot import plot_gradient_timeseries as _plot_gradient_timeseries
from ..analyse.plot import (
    plot_mbar_gradient_convergence as _plot_mbar_gradient_convergence,
)
from ..analyse.plot import plot_mbar_pmf as _plot_mbar_pmf
from ..analyse.plot import plot_overlap_mats as _plot_overlap_mats
from ..analyse.plot import plot_rmsds as _plot_rmsds
from ..analyse.plot import plot_sq_sem_convergence as _plot_sq_sem_convergence
from ..analyse.process_grads import GradientData as _GradientData
from ._simulation_runner import SimulationRunner as _SimulationRunner
from ._virtual_queue import VirtualQueue as _VirtualQueue
from ..configuration.enums import StageType as _StageType
from .lambda_window import LamWindow as _LamWindow
from ..configuration.slurm_config import SlurmConfig as _SlurmConfig
from ..configuration.engine_config import SomdConfig as _SomdConfig


class Stage(_SimulationRunner):
    """
    Class to hold and manipulate an ensemble of SOMD simulations for a
    single stage of a calculation.
    """

    # Files to be cleaned by self.clean()
    run_files = _SimulationRunner.run_files + [
        "check_equil_multiwindow*.txt",
        "freenrg-MBAR*.dat",
        "*.out",
    ]

    runtime_attributes = _deepcopy(_SimulationRunner.runtime_attributes)
    runtime_attributes["_maximally_efficient"] = False

    # A thread-safe counter for e.g. naming unique files.
    _counter = _get_context("spawn").Value("i", 0)

    def __init__(
        self,
        stage_type: _StageType,
        equil_detection: str = "multiwindow",
        runtime_constant: _Optional[float] = 0.005,
        relative_simulation_cost: float = 1,
        ensemble_size: int = 5,
        lambda_values: _Optional[_List[float]] = None,
        base_dir: _Optional[str] = None,
        input_dir: _Optional[str] = None,
        output_dir: _Optional[str] = None,
        stream_log_level: int = _logging.INFO,
        slurm_config: _Optional[_SlurmConfig] = None,
        analysis_slurm_config: _Optional[_SlurmConfig] = None,
        engine_config: _Optional[_SomdConfig] = None,
        engine_type: _EngineType = _EngineType.SOMD,
        update_paths: bool = True,
    ) -> None:
        """
        Initialise an ensemble of SOMD simulations, constituting the Stage. If Stage.pkl exists in the
        output directory, the Stage will be loaded from this file and any arguments
        supplied will be overwritten.

        Parameters
        ----------
        stage_type : StageType
            The type of stage.
        equil_detection : str, Optional, default: "multiwindow"
            Method to use for equilibration detection. Options are:
            - "multiwindow": Use the multiwindow paired t-test method to detect equilibration.
            - "chodera": Use Chodera's method to detect equilibration.
        runtime_constant : float, Optional, default: 0.005
            The runtime_constant (kcal**2 mol**-2 ns*-1) only affects behaviour if running adaptively, and must
            be supplied if running adaptively. This is used to calculate how long to run each simulation for based on
            the current uncertainty of the per-window free energy estimate, as discussed in the docstring of the run() method.
        relative_simlation_cost : float, Optional, default: 1
            The relative cost of the simulation for a given runtime. This is used to calculate the
            predicted optimal runtime during adaptive simulations. The recommended use
            is to set this to 1 for the bound leg and to (speed of bound leg / speed of free leg)
            for the free leg.
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
        slurm_config: SlurmConfig, default: None
            Configuration for the SLURM job scheduler. If None, the
            default partition is used.
        analysis_slurm_config: SlurmConfig, default: None
            Configuration for the SLURM job scheduler for the analysis.
            This is helpful e.g. if you want to submit analysis to the CPU
            partition, but the main simulation to the GPU partition. If None,
        engine_config: SomdConfig, default: None
            Configuration for the SOMD engine. If None, the default configuration is used.
        engine_type: EngineType, default: EngineType.SOMD
            The type of engine to use for the production simulations.
        update_paths: bool, Optional, default: True
            If True, if the simulation runner is loaded by unpickling, then
            update_paths() is called.

        Returns
        -------
        None
        """
        # Set the stage type first, as this is required for __str__,
        # and threrefore the super().__init__ call
        self.stage_type = stage_type

        super().__init__(
            base_dir=base_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            stream_log_level=stream_log_level,
            slurm_config=slurm_config,
            analysis_slurm_config=analysis_slurm_config,
            engine_config=engine_config,
            engine_type=engine_type,
            ensemble_size=ensemble_size,
            update_paths=update_paths,
            dump=False,
        )

        if not self.loaded_from_pickle:
            if lambda_values is not None:
                self.lam_vals = lambda_values
            else:
                self.lam_vals = self._get_lam_vals()

            if self.engine_config is not None:
                self.engine_config.lambda_values = self.lam_vals

            self.equil_detection = equil_detection
            self.runtime_constant = runtime_constant
            self.relative_simulation_cost = relative_simulation_cost
            self._maximally_efficient = False  # Set to True if the stage has been run so as to reach max efficieny
            self._running: bool = False
            self.run_thread: _Optional[_threading.Thread] = None
            # Set boolean to allow us to kill the thread
            self.kill_thread: bool = False
            self.running_wins: _List[_LamWindow] = []
            self.virtual_queue = _VirtualQueue(
                log_dir=self.base_dir, stream_log_level=self.stream_log_level
            )
            # Creating lambda window objects sets up required input directories
            lam_val_weights = self.lam_val_weights
            for i, lam_val in enumerate(self.lam_vals):
                lam_base_dir = _os.path.join(self.output_dir, f"lambda_{lam_val:.3f}")
                self.lam_windows.append(
                    _LamWindow(
                        lam=lam_val,
                        lam_val_weight=lam_val_weights[i],
                        virtual_queue=self.virtual_queue,
                        equil_detection=self.equil_detection,
                        runtime_constant=self.runtime_constant,
                        relative_simulation_cost=self.relative_simulation_cost,
                        ensemble_size=self.ensemble_size,
                        base_dir=lam_base_dir,
                        input_dir=self.input_dir,
                        stream_log_level=self.stream_log_level,
                        slurm_config=self.slurm_config,
                        analysis_slurm_config=self.analysis_slurm_config,
                        engine_config=self.engine_config.copy(),
                        engine_type=self.engine_type,
                    )
                )

            # Point self._sub_sim_runners at the lambda windows
            self._sub_sim_runners = self.lam_windows

            # Save the state and update log
            self._update_log()
            self._dump()

    def __str__(self) -> str:
        return f"Stage (type = {self.stage_type.name.lower()})"

    @property
    def lam_windows(self) -> _List[_LamWindow]:
        return self._sub_sim_runners

    @lam_windows.setter
    def legs(self, value) -> None:
        self._logger.info("Modifying/ creating lambda windows")
        self._sub_sim_runners = value

    @property
    def lam_val_weights(self) -> _List[float]:
        """Return the weights for each lambda window. These are calculated
        according to how each windows contributes to the overall free energy
        estimate, as given by TI and the trapezoidal rule."""
        lam_val_weights = []
        for i, lam_val in enumerate(self.lam_vals):
            if i == 0:
                lam_val_weights.append(0.5 * (self.lam_vals[i + 1] - lam_val))
            elif i == len(self.lam_vals) - 1:
                lam_val_weights.append(0.5 * (lam_val - self.lam_vals[i - 1]))
            else:
                lam_val_weights.append(
                    0.5 * (self.lam_vals[i + 1] - self.lam_vals[i - 1])
                )
        return lam_val_weights

    def run(
        self,
        run_nos: _Optional[_List[int]] = None,
        adaptive: bool = True,
        runtime: _Optional[float] = None,
        runtime_constant: _Optional[float] = None,
    ) -> None:
        """Run the ensemble of simulations constituting the stage (optionally with adaptive
        equilibration detection), and, if using adaptive equilibration detection, perform
        analysis once finished. If running adaptively, cycles of short runs then optimal
        runtime estimation are performed, where the optimal runtime is estimated according to

        .. math::

            t_{\\mathrm{Optimal, k}} = \\sqrt{\\frac{t_{\\mathrm{Current}, k}}{C}}\\sigma_{\\mathrm{Current}}(\\Delta \\widehat{F}_k)

        where:
        - :math:`t_{\\mathrm{Optimal, k}}` is the calculated optimal runtime for lambda window :math:`k`
        - :math:`t_{\\mathrm{Current}, k}` is the current runtime for lambda window :math:`k`
        - :math:`C` is the runtime constant
        - :math:`\\sigma_{\\mathrm{Current}}(\\Delta \\widehat{F}_k)` is the current uncertainty in the free energy change contribution for lambda window :math:`k`. This is estimated from inter-run deviations.
        - :math:`\\Delta \\widehat{F}_k` is the free energy change contribution for lambda window :math:`k`

        Parameters
        ----------
        adaptive : bool, Optional, default: True
            If True, the stage will run until the simulations are equilibrated and perform analysis afterwards.
            If False, the stage will run for the specified runtime and analysis will not be performed.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this number of nanoseconds.
        runtime_constant: float, Optional, default: None
            The runtime_constant (kcal**2 mol**-2 ns*-1) only affects behaviour if running adaptively. This is used
            to calculate how long to run each simulation for based on the current uncertainty of the per-window
            free energy estimate.

        Returns
        -------
        None
        """
        run_nos = self._get_valid_run_nos(run_nos)

        if not adaptive and runtime is None:
            raise ValueError(
                "If adaptive equilibration detection is disabled, a runtime must be supplied."
            )
        # Check adaptive and runtime settings
        if adaptive and runtime is not None:
            raise ValueError(
                "If adaptive equilibration detection is enabled, a runtime cannot be supplied."
            )

        if not adaptive and runtime_constant:
            raise ValueError(
                "Runtime constant can only be supplied if running adaptively."
            )

        if runtime_constant:
            self.recursively_set_attr("runtime_constant", runtime_constant, silent=True)

        # Run in the background with threading so that user can continuously check
        # the status of the Stage object
        self.run_thread = _threading.Thread(
            target=self._run_without_threading,
            args=(run_nos, adaptive, runtime),
            name=str(self),
        )
        self.run_thread.start()

    def kill(self) -> None:
        """Kill all running simulations."""
        # Stop the run loop
        if self.running:
            self._logger.info("Killing all lambda windows")
            self.kill_thread = True
            for win in self.lam_windows:
                win.kill()

    def _run_without_threading(
        self,
        run_nos: _List[int],
        adaptive: bool = True,
        runtime: _Optional[float] = None,
        max_runtime: float = 60,
    ) -> None:
        """Run the ensemble of simulations constituting the stage (optionally with adaptive
        equilibration detection), and, if using adaptive equilibration detection, perform
        analysis once finished.  This function is called by run() with threading, so that
        the function can be run in the background and the user can continuously check
        the status of the Stage object.

        Parameters
        ----------
        run_nos : List[int]
            The run numbers to run.
        adaptive : bool, Optional, default: True
            If True, the stage will run until the simulations are equilibrated and perform analysis afterwards.
            If False, the stage will run for the specified runtime and analysis will not be performed.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this number of nanoseconds.
        max_runtime : float, Optional, default: 30
            The maximum runtime for a single simulation during an adaptive simulation, in ns. Only used when adaptive == True.

        Returns
        -------
        None
        """
        try:
            # Reset self.kill_thread so we can restart after killing
            self.kill_thread = False

            if not adaptive and runtime is None:
                raise ValueError(
                    "If adaptive equilibration detection is disabled, a runtime must be supplied."
                )
            if adaptive and runtime is not None:
                raise ValueError(
                    "If adaptive equilibration detection is enabled, a runtime cannot be supplied."
                )

            if not adaptive:
                self._logger.info(
                    f"Starting {self}. Adaptive equilibration = {adaptive}..."
                )

            elif adaptive:
                self._logger.info(
                    f"Starting {self}. Adaptive equilibration = {adaptive}..."
                )
                if runtime is None:
                    runtime = 0.2  # ns

            # Run initial SOMD simulations
            for win in self.lam_windows:
                win.run(run_nos=run_nos, runtime=runtime)  # type: ignore
                win._update_log()
                self._dump()

            # Periodically check the simulations and analyse/ resubmit as necessary
            # Copy to ensure that we don't modify self.lam_windows when updating self.running_wins
            self.running_wins = self.lam_windows.copy()
            self._dump()

            # Run the appropriate run loop
            if adaptive:
                # Allocate simulation time to achieve maximum efficiency
                self._run_loop_adaptive_efficiency(
                    run_nos=run_nos, max_runtime=max_runtime
                )
                # Check that equilibration has been achieved and resubmit if required
                self._run_loop_adaptive_equilibration_multiwindow(
                    run_nos=run_nos, max_runtime=max_runtime
                )
            else:
                self._run_loop_non_adaptive()

            # All simulations are now finished, so perform final analysis
            self._logger.info(f"All simulations in {self} have finished.")

        except Exception as e:
            self._logger.exception("")
            raise e

    def _run_loop_non_adaptive(self, cycle_pause: int = 60) -> None:
        """The run loop for non-adaptive runs. Simply wait
        until all lambda windows have finished running.

        Parameters
        ----------
        cycle_pause : int, Optional, default: 60
            The number of seconds to wait between checking the status of the simulations.
        """

        while self.running_wins:
            _sleep(cycle_pause)  # Check every 60 seconds
            # Check if we've requested to kill the thread
            if self.kill_thread:
                self._logger.info("Kill thread requested: exiting run loop")
                return

            # Update the queue before checking the simulations
            self.virtual_queue.update()

            # Check if everything has finished
            for win in self.running_wins:
                # Check if the window has now finished - calling win.running updates the win._running attribute
                if not win.running:
                    self._logger.info(f"{win} has finished at {win.tot_simtime:.3f} ns")
                    self.running_wins.remove(win)

                    # Write status after checking for running and equilibration, as the
                    # _running and _equilibrated attributes have now been updated
                    win._update_log()
                    self._dump()

    def _run_loop_adaptive_efficiency(
        self,
        run_nos: _List[int],
        cycle_pause: int = 60,
        max_runtime: float = 30,  # ns
    ) -> None:
        """
        Run loop which allocates sampling time in order to achieve maximal estimation
        efficiency of the free energy difference.

        Parameters
        ----------
        run_nos : List[int]
            The run numbers to run.
        cycle_pause : int, Optional, default: 60
            The number of seconds to wait between checking the status of the simulations.
        max_runtime : float, Optional, default: 30
            The maximum runtime for a single simulation during an adaptive simulation, in ns.
        """

        while not self._maximally_efficient:
            self._logger.info(
                "Maximum efficiency for given runtime constant not achieved."
                " Allocating simulation time to achieve maximum efficiency..."
            )

            # Firstly, wait til all window have finished
            while self.running_wins:
                _sleep(cycle_pause)  # Check every 60 seconds
                # Check if we've requested to kill the thread
                if self.kill_thread:
                    self._logger.info("Kill thread requested: exiting run loop")
                    return
                # Update the queue before checking the simulations
                self.virtual_queue.update()
                for win in self.running_wins:
                    if not win.running:
                        self.running_wins.remove(win)
                    # Write status after checking for running and equilibration, as the
                    # _running and _equilibrated attributes have now been updated
                    win._update_log()
                    self._dump()

            # Now all windows have finished, check if we have reached the maximum
            # efficiency and resubmit if not
            # Get the gradient data and extract the per-lambda SEM of the free energy change
            gradient_data = _GradientData(
                lam_winds=self.lam_windows, equilibrated=False
            )
            smooth_dg_sems = gradient_data.get_time_normalised_sems(
                origin="inter_delta_g", smoothen=True
            )

            # For each window, calculate the predicted most efficient run time. See if we have reached this
            # and if not, resubmit for more simulation time
            for i, win in enumerate(self.lam_windows):
                # Note that normalised_sem_dg has already been multiplied by sqrt(tot_simtime)
                normalised_sem_dg = smooth_dg_sems[i]
                predicted_run_time_max_eff = (
                    1 / _np.sqrt(self.runtime_constant * self.relative_simulation_cost)  # type: ignore
                ) * normalised_sem_dg
                actual_run_time = win.get_tot_simtime(run_nos=run_nos)
                win._logger.info(
                    f"Predicted maximum efficiency run time for is {predicted_run_time_max_eff:.3f} ns"
                )
                win._logger.info(f"Actual run time is {actual_run_time} ns")
                # Make sure that we don't exceed the maximum per-simulation runtime
                if predicted_run_time_max_eff > max_runtime * win.ensemble_size:
                    win._logger.info(
                        f"Predicted maximum efficiency run time per window is "
                        f"{predicted_run_time_max_eff / win.ensemble_size}, which exceeds the maximum runtime of "
                        f"{max_runtime} ns. Running to the maximum runtime instead."
                    )
                    predicted_run_time_max_eff = max_runtime * win.ensemble_size
                if actual_run_time < predicted_run_time_max_eff:
                    resubmit_time = (
                        predicted_run_time_max_eff - actual_run_time
                    ) / win.ensemble_size
                    # Resubmit a maximum of the current total simulation time again. This avoids issues with
                    # overestimating the best runtime intially and then resubmitting for too long
                    if resubmit_time > actual_run_time / win.ensemble_size:
                        resubmit_time = actual_run_time / win.ensemble_size
                    resubmit_time = (
                        _ceil(resubmit_time * 10) / 10
                    )  # Round up to the nearest 0.1 ns
                    if (
                        resubmit_time > 0
                    ):  # We have not reached the maximum efficiency, so resubmit for the remaining time
                        win._logger.info(
                            f"Window has not reached maximum efficiency. Resubmitting for {resubmit_time:.3f} ns"
                        )
                        win.run(run_nos=run_nos, runtime=resubmit_time)
                        self.running_wins.append(win)
                else:  # We have reached or exceeded the maximum efficiency runtime
                    win._logger.info(
                        f"Window has reached the most efficient run time at {actual_run_time}. "
                        "No further simulation required"
                    )

            # If there are no running lambda windows, we must have reached the maximum efficiency
            if not self.running_wins:
                self._maximally_efficient = True
                self._logger.info(
                    "Maximum efficiency for given runtime constant of "
                    f"{self.runtime_constant} kcal**2 mol**-2 ns**-1 achieved"
                )

    def _run_loop_adaptive_equilibration_multiwindow(
        self,
        run_nos: _List[int],
        cycle_pause: int = 60,
        max_runtime: float = 30,  # seconds  # ns
        check_equil_fn: _Callable = _check_equil_multiwindow_paired_t,
        check_equil_kwargs: _Dict[str, _Any] = {},
    ) -> None:
        """
        Run loop which detects equilibration using the check_equil_multiwindow_modified_geweke method.
        This checks if equilibration has been achieved over the whole stage, and if not,
        halves the runtime of the simulations and using the adaptive efficiency loop before
        re-checking for equilibration.

        Parameters
        ----------
        run_nos : List[int]
            The run numbers to run.
        cycle_pause : int, Optional, default: 60
            The number of seconds to wait between checking the status of the simulations.
        max_runtime : float, Optional, default: 30
            The maximum runtime for a single simulation during an adaptive simulation, in ns.
        check_equil_fn : Callable, Optional, default: dummy_check_equil_multiwindow
            The function to use to check for equilibration. This should be a function which
            takes a list of lambda windows and returns a boolen to indicate if equilibration
            over all lambda windows has been achieved. This should also set the _equilibrated and
            _equil_time attributes of each lambda window.
        check_equil_kwargs : Dict[str, Any], Optional, default: {}
            Any keyword arguments to pass to the check_equil_fn function.
        """
        # Check that all lambda windows have the correct equilibration detection method
        if any(
            [
                window.check_equil != _dummy_check_equil_multiwindow
                for window in self.lam_windows
            ]
        ):
            raise ValueError(
                "Not all lambda windows have the correct equilibration detection method. "
                "This should be set to dummy_check_equil_multiwindow"
            )
        if not self.runtime_constant:
            raise ValueError(
                "Cannot run adaptive equilibration multiwindow without a runtime constant"
            )

        # Assumes that we have not already equilibrated
        while not self.is_equilibrated(run_nos=run_nos):
            if self.kill_thread:
                # Check if we've requested to kill the thread
                self._logger.info("Kill thread requested: exiting run loop")
                return

            # Check if we have reached equilibration
            self._wait_ignoring_thread()

            self._logger.info(
                "Checking for equilibration with the check_equil_multiwindow algorithm..."
            )
            equilibrated, fractional_equil_time = check_equil_fn(
                self.lam_windows,
                output_dir=self.output_dir,
                run_nos=run_nos,
                **check_equil_kwargs,
            )
            if equilibrated:
                total_simtime = self.get_tot_simtime(run_nos=run_nos)
                tot_equil_time = total_simtime * fractional_equil_time  # type: ignore
                self._logger.info("Equilibration achieved")
                self._logger.info(
                    f"Fractional equilibration time is {fractional_equil_time}"
                )
                self._logger.info(f"Total equilibration time is {tot_equil_time}")
                self._logger.info(f"Total simulation time is {total_simtime}")
                self._logger.info(f"Runtime constant is {self.runtime_constant}")
                break
            else:
                self._logger.info(
                    "Equilibration not achieved. Quartering runtime constant and running adaptive efficiency loop"
                )
                # Quarter the simulation constant to double the predicted runtime,
                # and run the adaptive efficiency loop again
                self.runtime_constant /= 4
                self._maximally_efficient = False
                # Make all windows running, as is required for the adaptive efficiency loop
                self.running_wins = self.lam_windows.copy()
                self._run_loop_adaptive_efficiency(
                    run_nos=run_nos, cycle_pause=cycle_pause, max_runtime=max_runtime
                )

    def get_optimal_lam_vals(
        self,
        er_type: str = "root_var",
        delta_er: _Optional[float] = None,
        n_lam_vals: _Optional[int] = None,
        run_nos: _List[int] = [1],
    ) -> _np.ndarray:
        """
        Get the optimal lambda values for the stage, based on the
        integrated SEM, and create plots.

        Parameters
        ----------
        er_type: str, optional, default="root_var"
            Whether to integrate the standard error of the mean ("sem") or root
            variance of the gradients ("root_var") to calculate the optimal
            lambda values.
        delta_er : float, optional, default=None
            If er_type == "root_var", the desired integrated root variance of the gradients
            between each lambda value, in kcal mol^(-1). If er_type == "sem", the
            desired integrated standard error of the mean of the gradients between each lambda
            value, in kcal mol^(-1) ns^(1/2). A sensible default for sem is 0.1 kcal mol-1 ns1/2,
            and for root_var is 1 kcal mol-1.  If not provided, the number of lambda windows must be
            provided with n_lam_vals.
        n_lam_vals : int, optional, default=None
            The number of lambda values to sample. If not provided, delta_er must be provided.
        run_nos : List[int], optional, default=[1]
            The run numbers to use for the calculation. Only 1 is run by default, so by default
            we only analyse 1. If using er_type = "SEM", more than one run must be specified.
        Returns
        -------
        optimal_lam_vals : np.ndarray
            List of optimal lambda values for the stage.
        """
        # Check that we have more than one run if using delta_er == "sem"
        if er_type == "sem" and len(run_nos) == 1:
            raise ValueError(
                "If using er_type = 'sem', more than one run must be specified, as the "
                "SEM is calculated using between-run errors by default."
            )

        self._logger.info(
            f"Calculating optimal lambda values with er_type = {er_type} and delta_er = {delta_er}..."
        )
        unequilibrated_gradient_data = _GradientData(
            lam_winds=self.lam_windows, equilibrated=False, run_nos=run_nos
        )
        for plot_type in [
            "mean",
            "stat_ineff",
            "integrated_sem",
            "integrated_var",
            "pred_best_simtime",
        ]:
            _plot_gradient_stats(
                gradients_data=unequilibrated_gradient_data,
                output_dir=self.output_dir,
                plot_type=plot_type,
            )
        _plot_gradient_hists(
            gradients_data=unequilibrated_gradient_data, output_dir=self.output_dir
        )
        return unequilibrated_gradient_data.calculate_optimal_lam_vals(
            er_type=er_type, delta_er=delta_er, n_lam_vals=n_lam_vals
        )

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

    def analyse(
        self,
        slurm: bool = False,
        run_nos: _Optional[_List[int]] = None,
        get_frnrg: bool = True,
        subsampling: bool = False,
        fraction: float = 1,
        plot_rmsds: bool = False,
    ) -> _Union[_Tuple[float, float], _Tuple[None, None]]:
        r"""Analyse the results of the ensemble of simulations. Requires that
        all lambda windows have equilibrated.

        Parameters
        ----------
        slurm : bool, optional, default=False
            Whether to use slurm for the analysis.
        run_nos : List[int], Optional, default: None
            The run numbers to analyse. If None, all runs will be analysed.
        get_frnrg : bool, optional, default=True
            If True, the free energy will be calculated with MBAR, otherwise
            this will be skipped.
        subsampling: bool, optional, default=False
            If True, the free energy will be calculated by subsampling using
            the methods contained within pymbar.
        fraction: float, optional, default=1
            The fraction of the data to use for analysis. For example, if
            fraction=0.5, only the first half of the data will be used for
            analysis. If fraction=1, all data will be used. Note that unequilibrated
            data is discarded from the beginning of simulations in all cases.
        plot_rmsds: bool, optional, default=False
            Whether to plot RMSDS. This is slow and so defaults to False.

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
        run_nos = self._get_valid_run_nos(run_nos)

        # Check that this is not still running
        if self.running:
            raise RuntimeError("Cannot perform analysis as the Stage is still running")

        # Check that none of the simulations have failed
        failed_sims_list = self.failed_simulations
        if failed_sims_list:
            self._logger.error(
                "Unable to perform analysis as several simulations did not complete successfully"
            )
            self._logger.error("Please check the output in the following directories:")
            for failed_sim in failed_sims_list:
                self._logger.error(failed_sim.base_dir)
            raise RuntimeError(
                "Unable to perform analysis as several simulations did not complete successfully"
            )

        # Check that all simulations have equilibrated
        for win in self.lam_windows:
            if not win.equilibrated:
                raise RuntimeError(
                    "Not all lambda windows have equilibrated. Analysis cannot be performed. "
                    "If you are running non-adaptively, please use the `set_equilibration_time` method "
                    "to set the equilibration time manually."
                )
            if win.equil_time is None:
                raise RuntimeError(
                    "Despite equilibration being detected, no equilibration time was found."
                )

        if get_frnrg:
            self._logger.info(
                f"Computing free energy changes using the MBAR for runs {run_nos}"
            )

            # Remove unequilibrated data from the equilibrated output directory
            for win in self.lam_windows:
                win._write_equilibrated_simfiles()

            # Run MBAR and compute mean and 95 % C.I. of free energy
            if not slurm:
                free_energies, errors, mbar_outfiles, _ = _run_mbar(
                    run_nos=run_nos,
                    output_dir=self.output_dir,
                    percentage_end=fraction * 100,
                    percentage_start=0,
                    subsampling=subsampling,
                    equilibrated=True,
                )
            else:
                jobs, mbar_outfiles, tmp_files = _submit_mbar_slurm(
                    output_dir=self.output_dir,
                    virtual_queue=self.virtual_queue,
                    slurm_config=self.analysis_slurm_config,
                    run_nos=run_nos,
                    percentage_end=fraction * 100,
                    percentage_start=0,
                    subsampling=subsampling,
                    equilibrated=True,
                )

                free_energies, errors, *_ = _collect_mbar_slurm(
                    output_dir=self.output_dir,
                    run_nos=run_nos,
                    jobs=jobs,
                    mbar_out_files=mbar_outfiles,
                    virtual_queue=self.virtual_queue,
                    tmp_files=tmp_files,
                )

            mean_free_energy = _np.mean(free_energies)
            # Gaussian 95 % C.I.
            conf_int = (
                _stats.t.interval(
                    0.95,
                    len(free_energies) - 1,
                    mean_free_energy,
                    scale=_stats.sem(free_energies),
                )[1]
                - mean_free_energy
            )  # 95 % C.I.

            # Write overall MBAR stats to file
            with open(f"{self.output_dir}/overall_stats.dat", "a") as ofile:
                if get_frnrg:
                    ofile.write(
                        "###################################### Free Energies ########################################\n"
                    )
                    ofile.write(
                        f"Mean free energy: {mean_free_energy: .3f} + /- {conf_int:.3f} kcal/mol\n"
                    )
                    for i in range(len(free_energies)):
                        ofile.write(
                            f"Free energy from run {i + 1}: {free_energies[i]: .3f} +/- {errors[i]:.3f} kcal/mol\n"
                        )
                    ofile.write(
                        "Errors are 95 % C.I.s based on the assumption of a Gaussian distribution of free energies\n"
                    )
                    ofile.write(f"Runs analysed: {run_nos}\n")

            # Plot overlap matrices and PMFs
            _plot_overlap_mats(
                output_dir=self.output_dir,
                nlam=len(self.lam_windows),
                mbar_outfiles=mbar_outfiles,
            )
            _plot_mbar_pmf(mbar_outfiles, self.output_dir)
            equilibrated_gradient_data = _GradientData(
                lam_winds=self.lam_windows, equilibrated=True
            )
            _plot_overlap_mats(
                output_dir=self.output_dir,
                nlam=len(self.lam_windows),
                predicted=True,
                gradient_data=equilibrated_gradient_data,
            )

        # Plot RMSDS
        if plot_rmsds:
            self._logger.info("Plotting RMSDs")
            _plot_rmsds(
                lam_windows=self.lam_windows,
                output_dir=self.output_dir,
                selection="resname LIG and (not name H*)",
            )

        # Analyse the gradient data and make plots
        self._logger.info("Plotting gradients data")
        equilibrated_gradient_data = _GradientData(
            lam_winds=self.lam_windows, equilibrated=True, run_nos=run_nos
        )
        for plot_type in [
            "mean",
            "stat_ineff",
            "integrated_sem",
            "integrated_var",
            "pred_best_simtime",
        ]:
            _plot_gradient_stats(
                gradients_data=equilibrated_gradient_data,
                output_dir=self.output_dir,
                plot_type=plot_type,
            )
        _plot_gradient_hists(
            gradients_data=equilibrated_gradient_data,
            output_dir=self.output_dir,
            run_nos=run_nos,
        )
        _plot_gradient_timeseries(
            gradients_data=equilibrated_gradient_data,
            output_dir=self.output_dir,
            run_nos=run_nos,
        )

        # Make plots of equilibration time
        self._logger.info("Plotting equilibration times")
        _plot_equilibration_time(
            lam_windows=self.lam_windows, output_dir=self.output_dir
        )

        # Check and plot the Gelman-Rubin stat
        rhat_dict = _check_equil_multiwindow_gelman_rubin(
            lambda_windows=self.lam_windows, output_dir=self.output_dir
        )
        rhat_equil = {lam: rhat < 1.1 for lam, rhat in rhat_dict.items()}
        for lam, equil in rhat_equil.items():
            if not equil:
                self._logger.warning(
                    f"The Gelman-Rubin statistic for lambda = {lam} is greater than 1.1. "
                    "This suggests that the repeat simulations have not converged to the "
                    "same distirbution and there is a sampling issue."
                )

        # Write out stats
        with open(f"{self.output_dir}/overall_stats.dat", "a") as ofile:
            for win in self.lam_windows:
                ofile.write(
                    f"Equilibration time for lambda = {win.lam}: {win.equil_time:.3f} ns per simulation\n"
                )
                ofile.write(
                    f"Total time simulated for lambda = {win.lam}: {win.sims[0].tot_simtime:.3f} ns per simulation\n"
                )

        if get_frnrg:
            self._logger.info(
                f"Overall free energy changes: {free_energies} kcal mol-1"
            )  # type: ignore
            self._logger.info(f"Overall errors: {errors} kcal mol-1")  # type: ignore
            self._logger.info(f"Analysed runs: {run_nos}")
            # Update the interally-stored results
            self._delta_g = free_energies
            self._delta_g_er = errors
            return free_energies, errors  # type: ignore
        else:
            return None, None

    def get_results_df(self, save_csv: bool = True) -> _pd.DataFrame:
        """
        Return the results in dataframe format

        Parameters
        ----------
        save_csv : bool, optional, default=True
            Whether to save the results as a csv file

        Returns
        -------
        results_df : pd.DataFrame
            A dataframe containing the results
        """
        # Call the superclass method but avoid trying to get results from
        # the sub_sim_runners, which do not have calculated free energy changes
        return super().get_results_df(save_csv=save_csv, add_sub_sim_runners=False)

    def analyse_convergence(
        self,
        slurm: bool = False,
        run_nos: _Optional[_List[int]] = None,
        mode: str = "cumulative",
        fraction: float = 1,
        equilibrated: bool = True,
    ) -> _Tuple[_np.ndarray, _np.ndarray]:
        """
        Get a timeseries of the total free energy change of the
        stage against total simulation time. Also plot this.
        This is kept separate from the analyse method as it is
        expensive to run.

        Parameters
        ----------
        slurm: bool, optional, default=False
            Whether to use slurm for the analysis.
        run_nos : List[int], Optional, default: None
            The run numbers to analyse. If None, all runs will be analysed.
        mode : str, optional, default="cumulative"
            "cumulative" or "block". The type of averaging to use. In both cases,
            20 MBAR evaluations are performed.
        fraction : float, optional, default=1
            The fraction of the total simulation time to use for the analysis.
            For example, if fraction=0.5, only the first 50 % of the simulation
            time will be used for the analysis.
        equilibrated: bool, optional, default=True
            Whether to analyse only the equilibrated data (True) or all data (False)

        Returns
        -------
        slurm: bool, optional, default=False
            Whether to use slurm for the analysis.
        fracts : np.ndarray
            The fraction of the total (equilibrated) simulation time for each value of dg_overall.
        dg_overall : np.ndarray
            The overall free energy change for the stage for each value of total (equilibrated)
            simtime for each of the ensemble size repeats.
        """
        run_nos = self._get_valid_run_nos(run_nos)

        self._logger.info("Analysing convergence...")

        # Check the assumption that all simulation times are the same
        av_simtime = self.get_tot_simtime(run_nos=run_nos) / len(run_nos)
        if not all(
            [
                _np.isclose(
                    self.get_tot_simtime(run_nos=[run_no]), av_simtime, rtol=1e-2
                )
                for run_no in run_nos
            ]
        ):
            raise RuntimeError(
                "Not all simulation times are the same. Convergence analysis cannot be performed."
            )

        # Get the dg_overall in terms of fraction of the total simulation time
        # Use steps of 5 % of the total simulation time
        fracts = _np.arange(0.05, 1.05, 0.05)
        # Only analyse up to specified fraction of total simulation data
        fracts = fracts * fraction
        end_percents = fracts * 100
        dg_overall = _np.zeros(len(fracts))

        # If cumulative mode, start from start from the beginning of the simulation each time
        if mode == "cumulative":
            start_percents = _np.zeros(len(fracts))
        elif mode == "block":
            start_percents = _np.arange(0.00, 1.00, 0.05) * 100 * fraction
        else:
            raise ValueError("mode must be 'cumulative' or 'block'")

        # Make sure to re-write the equilibrated simfiles
        if equilibrated:
            for win in self.lam_windows:
                win._write_equilibrated_simfiles()

        if not slurm:
            # Now run mbar with multiprocessing to speed things up
            with _get_context("spawn").Pool() as pool:
                results = pool.starmap(
                    _run_mbar,
                    [
                        (
                            self.output_dir,
                            run_nos,
                            end_percent,
                            start_percent,
                            False,  # Subsample
                            True,  # Delete output files
                            equilibrated,  # Equilibrated
                        )
                        for start_percent, end_percent in zip(
                            start_percents, end_percents
                        )
                    ],
                )
        else:  # Use slurm
            frac_jobs = []
            results = []
            for start_percent, end_percent in zip(start_percents, end_percents):
                frac_jobs.append(
                    _submit_mbar_slurm(
                        output_dir=self.output_dir,
                        virtual_queue=self.virtual_queue,
                        slurm_config=self.analysis_slurm_config,
                        run_nos=run_nos,
                        percentage_end=end_percent,
                        percentage_start=start_percent,
                        subsampling=False,
                        equilibrated=equilibrated,
                    )
                )

            for frac_job in frac_jobs:
                jobs, mbar_outfiles, tmp_simfiles = frac_job
                results.append(
                    _collect_mbar_slurm(
                        output_dir=self.output_dir,
                        run_nos=run_nos,
                        jobs=jobs,
                        mbar_out_files=mbar_outfiles,
                        virtual_queue=self.virtual_queue,
                        tmp_files=tmp_simfiles,
                    )
                )

        dg_overall = _np.array(
            [result[0] for result in results]
        ).transpose()  # result[0] is a 2D array for a given percent
        mbar_grads = [
            result[3] for result in results
        ]  # result[3] is a Dict of gradient data for a given percent

        self._logger.info(f"Overall free energy changes: {dg_overall} kcal mol-1")
        self._logger.info(f"Fractions of (equilibrated) simulation time: {fracts}")

        # Save the convergence information as an attribute
        self._delta_g_convergence = dg_overall
        self._delta_g_convergence_fracts = fracts

        # Plot the overall convergence and the squared SEM of the free energy change
        for plot in [_plot_convergence, _plot_sq_sem_convergence]:
            plot(
                fracts,
                dg_overall,
                self.get_tot_simtime(run_nos=run_nos),
                (
                    self.equil_time if equilibrated else 0
                ),  # Already per member of the ensemble
                self.output_dir,
                len(run_nos),
            )

        # Plot the convergence of the MBAR gradients
        _plot_mbar_gradient_convergence(
            fracts=fracts,
            mbar_grads=mbar_grads,
            simtime_per_run=self.get_tot_simtime(
                run_nos=[1]
            ),  # Assumes all simulation times the same
            equil_time_per_run=self.equil_time if equilibrated else 0,
            output_dir=self.output_dir,
        )

        return fracts, dg_overall

    def setup(self) -> None:
        raise NotImplementedError("Stages are set up when they are created")

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

    def set_simfile_option(self, option: str, value: str) -> None:
        setattr(self.engine_config, option, value)
        self.engine_config.get_somd_config(self.input_dir)
        super().set_simfile_option(option, value)

    def wait(self) -> None:
        """Wait for the stage to finish running."""
        # Override the base class method so that we can update the
        # virtual queue
        # Give the simulations a chance to start
        _sleep(30)
        self.virtual_queue.update()
        while self.running:
            _sleep(30)  # Check every 30 seconds
            self.virtual_queue.update()

    def _wait_ignoring_thread(self) -> None:
        """Wait for the stage to finish running, ignoring the thread."""
        _sleep(30)
        self.virtual_queue.update()
        # Superclass implementation of running ignores the thread
        while super().running:
            _sleep(30)
            self.virtual_queue.update()

    @property
    def running(self) -> bool:
        """Check if the stage is running."""
        # Override the base class method so that we can also check for active threads
        live_thread = (
            self.run_thread.is_alive() if self.run_thread is not None else False
        )
        return super().running or live_thread

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

        old_lam_vals_attrs = self.lam_windows[0].__dict__
        self._logger.info("Deleting old lambda windows and creating new ones...")
        self._sub_sim_runners = []
        lam_val_weights = self.lam_val_weights
        for i, lam_val in enumerate(self.lam_vals):
            lam_base_dir = _os.path.join(self.output_dir, f"lambda_{lam_val:.3f}")
            new_lam_win = _LamWindow(
                lam=lam_val,
                lam_val_weight=lam_val_weights[i],
                virtual_queue=self.virtual_queue,
                runtime_constant=self.runtime_constant,
                relative_simulation_cost=self.relative_simulation_cost,
                ensemble_size=self.ensemble_size,
                base_dir=lam_base_dir,
                input_dir=self.input_dir,
                stream_log_level=self.stream_log_level,
                slurm_config=self.slurm_config,
                analysis_slurm_config=self.analysis_slurm_config,
                engine_config=self.engine_config.copy(),
            )
            # Overwrite the default equilibration detection algorithm
            new_lam_win.check_equil = old_lam_vals_attrs["check_equil"]
            self.lam_windows.append(new_lam_win)

    def clean(self, clean_logs=False) -> None:
        """
        Clean the simulation runner by deleting all files
        with extensions matching self.__class__.run_files in the
        base and output dirs, and resetting the total runtime to 0.
        Also flush the virtual queue to remove any remaining simulations.

        Parameters
        ----------
        clean_logs : bool, default=False
            If True, also delete the log files.
        """
        self.virtual_queue._flush()
        super().clean(clean_logs=clean_logs)


class StageContextManager:
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
