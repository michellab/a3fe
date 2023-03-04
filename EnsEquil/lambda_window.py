"""Functionality for running simulations at a single lambda value."""

import logging as _logging
import numpy as _np
import os as _os
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

from .detect_equil import check_equil_block_gradient as _check_equil_block_gradient, check_equil_chodera as _check_equil_chodera
from .simulation import Simulation as _Simulation
from ._utils import VirtualQueue as _VirtualQueue 

class LamWindow():
    """A class to hold and manipulate a set of SOMD simulations at a given lambda value."""

    equil_detection_methods={"block_gradient": _check_equil_block_gradient,
                               "chodera": _check_equil_chodera}

    def __init__(self, lam: float,
                 virtual_queue: _VirtualQueue,
                 block_size: float=1,
                 equil_detection: str="block_gradient",
                 gradient_threshold: _Optional[float]=None,
                 ensemble_size: int=5, 
                 input_dir: _Optional[str] = None,
                 output_dir: _Optional[str] = None,
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
        if input_dir is None:
            input_dir = _os.path.join(_os.getcwd(), "input")
        self.input_dir = input_dir
        if output_dir is None:
            output_dir = _os.path.join(_os.getcwd(), "output")
        self.output_dir = output_dir
        self._equilibrated: bool=False
        self.equil_time: _Optional[float]=None
        self._running: bool=False
        self.tot_simtime: float=0  # ns

        # Create the required simulations for this lambda value
        self.sims=[]
        for i in range(1, ensemble_size + 1):
            sim=_Simulation(lam, i, self.virtual_queue, input_dir, output_dir, stream_log_level)
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
        self._running=False

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
