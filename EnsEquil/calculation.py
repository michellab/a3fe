"""Functionality for setting up and running an entire ABFE calculation,
consisting of two legs (bound and unbound) and multiple stages."""

import logging as _logging
from multiprocessing import Pool as _Pool
import os as _os
import pickle as _pkl
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

from .leg import Leg as _Leg, LegType as _LegType, PreparationStage as _PreparationStage
from .stage import Stage as _Stage, StageContextManager as _StageContextManager
from ._simulation_runner import SimulationRunner as _SimulationRunner

class Calculation(_SimulationRunner):
    """
    Class to set up and run an entire ABFE calculation, consisting of two legs
    (bound and unbound) and multiple stages.
    """

    required_input_files = ["run_somd.sh",
                            "protein.pdb",
                            "ligand.pdb",
                            "template_config.cfg"] # Waters.pdb is optional

    #required_legs = [_LegType.BOUND, _LegType.FREE]
    required_legs = [_LegType.FREE]

    def __init__(self, 
                 block_size: float = 1,
                 equil_detection: str = "block_gradient",
                 gradient_threshold: _Optional[float] = None,
                 ensemble_size: int = 5,
                 input_dir: _Optional[str] = None,
                 base_dir: _Optional[str] = None,
                 stream_log_level: int = _logging.INFO) -> None:
        """
        Instantiate a calculation based on files in the input dir. If calculation.pkl exists in the
        base directory, the calculation will be loaded from this file and any arguments
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
        base_dir : str, Optional, default: None
            Path to the base directory in which to set up the legs and stages. If None,
            this is set to the current working directory.
        input_dir : str, Optional, default: None
            Path to directory containing input files for the simulations. If None, this
            is set to `current_working_directory/input`.
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            calculation object and its child objects.

        Returns
        -------
        None
        """
        super().__init__(base_dir=base_dir,
                         input_dir=input_dir,
                         output_dir=None,
                         stream_log_level=stream_log_level)
        
        if not self.loaded_from_pickle:
            self.block_size = block_size
            self.equil_detection = equil_detection
            self.gradient_threshold = gradient_threshold
            self.ensemble_size = ensemble_size
            self.setup_complete: bool = False
            
            # Validate the input
            self._validate_input()

            # Save the state and update log
            self._update_log()
            self._dump()

    def _validate_input(self) -> None:
        """Check that the required input files are present in the input directory."""
        # Check backwards, as we care about the most advanced preparation stage
        for prep_stage in reversed(_PreparationStage):
            files_absent = False
            for leg_type in Calculation.required_legs:
                for file in _Leg.required_input_files[leg_type][prep_stage]:
                    if not _os.path.isfile(f"{self.input_dir}/{file}"):
                        files_absent = True
            # We have the required files for this prep stage for both legs, and this is the most 
            # advanced prep stage that files are present for
            if not files_absent:
                self.prep_stage = prep_stage
                self._logger.info(f"Found all required input files for preparation stage {prep_stage.name.lower()}")
                return 
        # We didn't find all required files for any of the prep stages
        raise ValueError(f"Could not find all required input files for " \
                          f"any preparation stage. Required files are: {_Leg.required_input_files[_LegType.BOUND]}" \
                            f"and {_Leg.required_input_files[_LegType.FREE]}")

    def setup(self) -> None:
        """ Set up the calculation."""

        if self.setup_complete:
            self._logger.info("Setup already complete. Skipping...")
            return

        # Set up the legs
        self.legs = []
        for leg_type in reversed(Calculation.required_legs):
            self._logger.info(f"Setting up {leg_type.name.lower()} leg...")
            leg = _Leg(leg_type=leg_type,
                       block_size = self.block_size,
                       equil_detection=self.equil_detection,
                       gradient_threshold=self.gradient_threshold,
                       ensemble_size=self.ensemble_size,
                       input_dir=self.input_dir,
                       base_dir=_os.path.join(self.base_dir, leg_type.name.lower()),
                       stream_log_level=self.stream_log_level)
            self.legs.append(leg)
            leg.setup()

        # Save the state
        self.setup_complete = True
        self._dump()

    def get_optimal_lam_vals(self, simtime:float = 0.1) -> None:
        """
        Determine the optimal lambda windows for each stage of the calculation
        by running short simulations at each lambda value and analysing them

        Parameters
        ----------
        simtime : float, Optional, default: 0.1
            The length of the short simulations to run, in ns.
        
        Returns
        -------
        None
        """
        # First, run all the simulations for a 100 ps
        self.run(analyse=False, adaptive=False, runtime=simtime)

        # Then, determine the optimal lambda windows
        for leg in self.legs:
            leg.get_optimal_lam_vals()

        # Save state
        self._dump()

    def run(self, analyse:bool = True, adaptive:bool=True, runtime:_Optional[float]=None) -> None:
        """
        Run all stages in parallel and perform analysis once finished.

        Parameters
        ----------
        analyse : bool, Optional, default: True
            If True, the analysis will be performed after the simulations are finished, and each stage will
            also be analysed individually.
        adaptive : bool, Optional, default: True
            If True, the stages will run until the simulations are equilibrated and perform analysis afterwards.
            If False, the stages will run for the specified runtime and analysis will not be performed.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this number of nanoseconds. 

        Returns
        -------
        None
        """
        if not self.setup_complete:
            raise ValueError("The calculation has not been set up yet. Please call setup() first.")
        # Unforunately, this has to be done at the level of stages rather than legs, because we can't
        # use pool.starmap at the level of legs as well as at the level of calculation
        # Get all the stages
        stages = []
        for leg in self.legs:
            stages.extend(leg.stages)

        # Run in parallel, using stage context manager behind the scenes
        # to ensure that stages are killed if the calculation is killed e.g. by
        # a KeyboardInterrupt
        self._logger.info(f"Running {len(stages)} stages in parallel...")
        with _Pool() as pool:
            pool.starmap(self._run_stage, [(stage, analyse, adaptive, runtime) for stage in stages])

        if analyse:
            self.analyse()

    def _run_stage(self, stage: _Stage, analyse:bool=True, adaptive:bool = True, runtime:_Optional[float]=None) -> None:
        """
        Run a stage of the calculation.

        Parameters
        ----------
        stage : _Stage
            The stage to run.
        analyse : bool, Optional, default: True
            If True, each stage will perform analysis after the simulations are finished.
        adaptive : bool, default: True
            If True, the stage will run until the simulations are equilibrated and perform analysis afterwards.
            If False, the stage will run for the specified runtime and analysis will not be performed.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this number of nanoseconds. 

        Returns
        -------
        None
        """
        # Check that the stage is not already running
        if stage.running:
            raise ValueError(f"Stage {stage} is already running.")
        # Run the stage
        self._logger.info(f"Running stage {stage.stage_type.name.lower()}, adaptive={adaptive}, runtime={runtime}...")
        stage._run_without_threading(analyse=analyse, adaptive=adaptive, runtime=runtime)

    def analyse(self) -> None:
        pass
