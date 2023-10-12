"""Functionality for setting up and running an entire ABFE calculation,
consisting of two legs (bound and unbound) and multiple stages."""

__all__ = ["Calculation"]

import logging as _logging
import multiprocessing as _mp
import os as _os
import shutil as _shutil
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import pandas as _pd

from ..analyse.exceptions import AnalysisError as _AnalysisError
from ._simulation_runner import SimulationRunner as _SimulationRunner
from .enums import LegType as _LegType
from .enums import PreparationStage as _PreparationStage
from .leg import Leg as _Leg


class Calculation(_SimulationRunner):
    """
    Class to set up and run an entire ABFE calculation, consisting of two legs
    (bound and unbound) and multiple stages.
    """

    required_input_files = [
        "run_somd.sh",
        "protein.pdb",
        "ligand.sdf",
        "template_config.cfg",
    ]  # Waters.pdb is optional

    required_legs = [_LegType.FREE, _LegType.BOUND]

    def __init__(
        self,
        block_size: float = 1,
        equil_detection: str = "block_gradient",
        gradient_threshold: _Optional[float] = None,
        runtime_constant: _Optional[float] = 0.001,
        relative_simulation_cost: float = 1,
        ensemble_size: int = 5,
        input_dir: _Optional[str] = None,
        base_dir: _Optional[str] = None,
        stream_log_level: int = _logging.INFO,
        update_paths: bool = True,
    ) -> None:
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
        runtime_constant : float, Optional, default: 0.001
            The runtime constant to use for the calculation, in kcal^2 mol^-2 ns^-1.
            This must be supplied if running adaptively. Each window is run until the
            SEM**2 / runtime >= runtime_constant.
        relative_simlation_cost : float, Optional, default: 1
            The relative cost of the simulation for a given runtime. This is used to calculate the
            predicted optimal runtime during adaptive simulations. The recommended use
            is to set this to 1 for the bound leg and to (speed of bound leg / speed of free leg)
            for the free leg.
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
        update_paths: bool, Optional, default: True
            If True, if the simulation runner is loaded by unpickling, then
            update_paths() is called.

        Returns
        -------
        None
        """
        super().__init__(
            base_dir=base_dir,
            input_dir=input_dir,
            output_dir=None,
            stream_log_level=stream_log_level,
            ensemble_size=ensemble_size,
            update_paths=update_paths,
        )

        if not self.loaded_from_pickle:
            self.block_size = block_size
            self.equil_detection = equil_detection
            self.gradient_threshold = gradient_threshold
            self.runtime_constant = runtime_constant
            self.relative_simulation_cost = relative_simulation_cost
            self.setup_complete: bool = False

            # Validate the input
            self._validate_input()

            # Save the state and update log
            self._update_log()
            self._dump()

    @property
    def legs(self) -> _List[_Leg]:
        return self._sub_sim_runners

    @legs.setter
    def legs(self, value) -> None:
        self._logger.info("Modifying/ creating legs")
        self._sub_sim_runners = value

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
                self._logger.info(
                    f"Found all required input files for preparation stage {prep_stage.name.lower()}"
                )
                return
        # We didn't find all required files for any of the prep stages
        raise ValueError(
            f"Could not find all required input files for "
            f"any preparation stage. Required files are: {_Leg.required_input_files[_LegType.BOUND]}"
            f"and {_Leg.required_input_files[_LegType.FREE]}"
        )

    def setup(
        self,
        slurm: bool = True,
        append_to_ligand_selection: str = "",
        use_same_restraints: bool = True,
        short_ensemble_equil: bool = False,
    ) -> None:
        """
        Set up the calculation. This involves parametrising, equilibrating, and
        deriving restraints for the bound leg. Most of the work is done by the
        Leg class.

        Parameters
        ----------
        slurm : bool, default=True
            If True, the setup jobs will be run through SLURM.
        append_to_ligand_selection: str, optional, default = ""
            For the bound leg, this appends the supplied string to the default atom
            selection which chooses the atoms in the ligand to consider as potential anchor
            points. The default atom selection is f'resname {ligand_resname} and not name H*'.
            Uses the mdanalysis atom selection language. For example, 'not name O*' will result
            in an atom selection of f'resname {ligand_resname} and not name H* and not name O*'.
        use_same_restraints: bool, default=True
            If True, the same restraints will be used for all of the bound leg repeats - by default
            , the restraints generated for the first repeat are used. This allows meaningful
            comparison between repeats for the bound leg. If False, the unique restraints are
            generated for each repeat.
        short_ensemble_equil: bool, default=False
            If True, the ensemble equilibration will be run for 0.1 ns instead of 5 ns. This is
            not recommended for production calculations, but is useful for testing.
        """

        if self.setup_complete:
            self._logger.info("Setup already complete. Skipping...")
            return

        # Set up the legs
        self.legs = []
        for leg_type in reversed(Calculation.required_legs):
            self._logger.info(f"Setting up {leg_type.name.lower()} leg...")
            leg = _Leg(
                leg_type=leg_type,
                block_size=self.block_size,
                equil_detection=self.equil_detection,
                runtime_constant=self.runtime_constant,
                relative_simulation_cost=self.relative_simulation_cost,
                gradient_threshold=self.gradient_threshold,
                ensemble_size=self.ensemble_size,
                input_dir=self.input_dir,
                base_dir=_os.path.join(self.base_dir, leg_type.name.lower()),
                stream_log_level=self.stream_log_level,
            )
            self.legs.append(leg)
            leg.setup(
                slurm=slurm,
                append_to_ligand_selection=append_to_ligand_selection,
                use_same_restraints=use_same_restraints,
                short_ensemble_equil=short_ensemble_equil,
            )

        # Save the state
        self.setup_complete = True
        self._dump()

    def get_optimal_lam_vals(
        self,
        simtime: float = 0.1,
        er_type: str = "root_var",
        delta_er: float = 1,
        run_nos: _List[int] = [1],
    ) -> None:
        """
        Determine the optimal lambda windows for each stage of the calculation
        by running short simulations at each lambda value and analysing them. This
        also sets the relative_simulation_effieciency of the free leg simulation
        runners (relative to the bound leg, which is set to 1).

        Parameters
        ----------
        simtime : float, Optional, default: 0.1
            The length of the short simulations to run, in ns.
        er_type: str, optional, default="root_var"
            Whether to integrate the standard error of the mean ("sem") or root
            variance of the gradients ("root_var") to calculate the optimal
            lambda values.
        delta_er : float, default=1
            If er_type == "root_var", the desired integrated root variance of the gradients
            between each lambda value, in kcal mol^(-1). If er_type == "sem", the
            desired integrated standard error of the mean of the gradients between each lambda
            value, in kcal mol^(-1) ns^(1/2). A sensible default for root_var is 1 kcal mol-1,
            and 0,1 kcal mol-1 ns^(1/2) for sem.
        run_nos : List[int], optional, default=[1]
            The run numbers to use for the calculation. Only 1 is run by default, so by default
            we only analyse 1. If using delta_er == "sem", more than one run must be specified.

        Returns
        -------
        None
        """
        # Check that we have more than one run if using delta_er == "sem"
        if er_type == "sem" and len(run_nos) == 1:
            raise ValueError(
                "If using er_type = 'sem', more than one run must be specified, as the "
                "SEM is calculated using between-run errors by default."
            )

        # First, run all the simulations for a 100 ps
        self._logger.info(
            f"Running simulations for {simtime} ns to determine optimal lambda values..."
        )
        self.run(adaptive=False, runtime=simtime, run_nos=run_nos)
        self.wait()

        # Then, determine the optimal lambda windows
        self._logger.info(
            f"Determining optimal lambda values for each leg with er_type = {er_type} and delta_er = {delta_er}..."
        )
        costs = {}
        for leg in self.legs:
            # Set simtime = None to avoid running any more simulations
            cost = leg.get_optimal_lam_vals(
                simtime=None, er_type=er_type, delta_er=delta_er, run_nos=run_nos
            )
            costs[leg.leg_type] = cost

        # Set the relative simulation cost for the free leg simulations
        self._logger.info("Setting relative simulation costs...")
        free_rel_simulation_cost = costs[_LegType.FREE] / costs[_LegType.BOUND]
        for leg in self.legs:
            if leg.leg_type == _LegType.FREE:
                leg.set_attr_values(
                    "relative_simulation_cost", free_rel_simulation_cost, force=True
                )

        # Save state
        self._dump()

    def run(
        self,
        run_nos: _Optional[_List[int]] = None,
        adaptive: bool = True,
        runtime: _Optional[float] = None,
        parallel: bool = True,
    ) -> None:
        """
        Run all stages and perform analysis once finished.

        Parameters
        ----------
        run_nos : List[int], Optional, default: None
            List of run numbers to run. If None, all runs will be run.
        adaptive : bool, Optional, default: True
            If True, the stages will run until the simulations are equilibrated and perform analysis afterwards.
            If False, the stages will run for the specified runtime and analysis will not be performed.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this number of nanoseconds.
        parallel : bool, Optional, default: True
            If True, the stages will run in parallel. If False, the stages will run sequentially.

        Returns
        -------
        None
        """
        if not self.setup_complete:
            raise ValueError(
                "The calculation has not been set up yet. Please call setup() first."
            )
        super().run(
            run_nos=run_nos, adaptive=adaptive, runtime=runtime, parallel=parallel
        )

    def update_run_somd(self) -> None:
        """
        Overwrite the run_somd.sh script in all simulation output dirs with
        the version currently in the calculation input dir.
        """
        master_run_somd = _os.path.join(self.input_dir, "run_somd.sh")
        for leg in self.legs:
            for stage in leg.stages:
                for lambda_window in stage.lam_windows:
                    for simulation in lambda_window.sims:
                        _shutil.copy(master_run_somd, simulation.input_dir)
