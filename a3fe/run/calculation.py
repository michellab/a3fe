"""Functionality for setting up and running an entire ABFE calculation,
consisting of two legs (bound and unbound) and multiple stages."""

__all__ = ["Calculation"]

import logging as _logging
import os as _os
import shutil as _shutil
from typing import List as _List
from typing import Optional as _Optional
from pathlib import Path as _Path

from ._simulation_runner import SimulationRunner as _SimulationRunner
from .enums import LegType as _LegType
from .enums import PreparationStage as _PreparationStage
from .leg import Leg as _Leg
from .system_prep import SystemPreparationConfig as _SystemPreparationConfig

# Notes from the paper - by JJ-2025-05-06 
# The simulations with the ligand in solvent collectively make up the free leg, while those with the receptor–ligand 
# complex make up the bound leg. Sets of calculations where interactions of a given type are introduced or removed 
# are termed stages: receptor–ligand restraints were introduced, charges were scaled, and Lennard-Jones (LJ) terms 
# were scaled in the restrain, discharge, and vanish stages, respectively

class Calculation(_SimulationRunner):
    """
    Class to set up and run an entire ABFE calculation, consisting of two legs
    (bound and unbound) and multiple stages.


    """
    # commented out by JJ 2025-05-03
    # required_input_files = [
    #     "run_somd.sh",
    #     "protein.pdb",
    #     "ligand.sdf",
    #     "template_config.cfg",
    # ]  # Waters.pdb is optional

    required_legs = [_LegType.FREE, _LegType.BOUND]

    def __init__(
        self,
        equil_detection: str = "multiwindow",
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
        equil_detection : str, Optional, default: "multiwindow"
            Method to use for equilibration detection. Options are:
            - "multiwindow": Use the multiwindow paired t-test method to detect equilibration.
                             This is applied on a per-stage basis.
            - "chodera": Use Chodera's method to detect equilibration.
        runtime_constant: float, Optional, default: 0.001
            The runtime_constant (kcal**2 mol**-2 ns*-1) only affects behaviour if running adaptively, and must
            be supplied if running adaptively. This is used to calculate how long to run each simulation for based on
            the current uncertainty of the per-window free energy estimate, as discussed in the docstring of the run() method.
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
            dump=False,
        )

        if not self.loaded_from_pickle:
            self.equil_detection = equil_detection
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
                self._prep_stage = prep_stage
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

    @property
    def prep_stage(self) -> _PreparationStage:
        if self.legs:
            min_prep_stage = _PreparationStage.PREEQUILIBRATED
            for leg in self.legs:
                min_prep_stage = min(
                    [min_prep_stage, leg.prep_stage], key=lambda x: x.value
                )
            self._prep_stage = min_prep_stage

        return self._prep_stage

    @property
    def is_complete(self) -> bool:
        f"""Whether the {self.__class__.__name__} has completed."""
        # Check if the overall_stats.dat file exists
        if _Path(f"{self.output_dir}/overall_stats.dat").is_file():
            return True

        return False

    def setup(
        self,
        bound_leg_sysprep_config: _Optional[_SystemPreparationConfig] = None,
        free_leg_sysprep_config: _Optional[_SystemPreparationConfig] = None,
        skip_preparation: bool = False, 
    ) -> None:
        """
        Set up the calculation. This involves parametrising, equilibrating, and
        deriving restraints for the bound leg. Most of the work is done by the
        Leg class.

        Parameters
        ----------
        bound_leg_sysprep_config: SystemPreparationConfig, opttional, default = None
            The system preparation configuration to use for the bound leg. If None, the default
            configuration is used.
        free_leg_sysprep_config: SystemPreparationConfig, opttional, default = None
            The system preparation configuration to use for the free leg. If None, the default
            configuration is used.
        """
        if self.setup_complete:
            self._logger.info("Setup already complete. Skipping...")
            return

        configs = {
            _LegType.BOUND: bound_leg_sysprep_config,
            _LegType.FREE: free_leg_sysprep_config,
        }

        # Set up the legs
        self.legs = []
        for leg_type in reversed(Calculation.required_legs):
            self._logger.info(f"Setting up {leg_type.name.lower()} leg...")
            leg = _Leg(
                leg_type=leg_type,
                equil_detection=self.equil_detection,
                runtime_constant=self.runtime_constant,
                relative_simulation_cost=self.relative_simulation_cost,
                ensemble_size=self.ensemble_size,
                input_dir=self.input_dir,
                base_dir=_os.path.join(self.base_dir, leg_type.name.lower()),
                stream_log_level=self.stream_log_level,
            )
            self.legs.append(leg)
            leg.setup(configs[leg_type], skip_preparation=skip_preparation)

        # Save the state
        self.setup_complete = True
        self._dump()

    def get_optimal_lam_vals(
        self,
        simtime: float = 0.1,
        er_type: str = "root_var",
        delta_er: float = 1,
        set_relative_sim_cost: bool = True,
        reference_sim_cost: float = 0.21,
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
        set_relative_sim_cost: bool, optional, default=True
            Whether to recursively set the relative simulation cost for the leg and all
            sub simulation runners according to the mean simulation cost of the leg.
        reference_sim_cost: float, optional, default=0.16
            The reference simulation cost to use if set_relative_sim_cost is True, in hr / ns.
            The default of 0.21 is the average bound leg simulation cost from a test set of ligands
            of a range of system sizes on RTX 2080s. This is used to set the relative simulation
            cost according to average_sim_cost / reference_sim_cost.
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
                simtime=None,
                er_type=er_type,
                delta_er=delta_er,
                set_relative_sim_cost=set_relative_sim_cost,
                reference_sim_cost=reference_sim_cost,
                run_nos=run_nos,
            )
            costs[leg.leg_type] = cost

        # Save state
        self._dump()

    def run(
        self,
        run_nos: _Optional[_List[int]] = None,
        adaptive: bool = True,
        runtime: _Optional[float] = None,
        runtime_constant: _Optional[float] = None,
        parallel: bool = True,
    ) -> None:
        """
        Run all stages and perform analysis once finished. If running adaptively,
        cycles of short runs then optimal runtime estimation are performed, where the optimal
        runtime is estimated according to

        .. math::

            t_{\\mathrm{Optimal, k}} = \\sqrt{\\frac{t_{\\mathrm{Current}, k}}{C}}\\sigma_{\\mathrm{Current}}(\\Delta \\widehat{F}_k)

        where:
        - :math:`t_{\\mathrm{Optimal, k}}` is the calculated optimal runtime for lambda window :math:`k`
        - :math:`t_{\\mathrm{Current}, k}` is the current runtime for lambda window :math:`k`
        - :math:`C` is the runtime constant
        - :math:`\sigma_{\\mathrm{Current}}(\\Delta \\widehat{F}_k)` is the current uncertainty in the free energy change contribution for lambda window :math:`k`. This is estimated from inter-run deviations.
        - :math:`\Delta \\widehat{F}_k` is the free energy change contribution for lambda window :math:`k`

        Parameters
        ----------
        run_nos : List[int], Optional, default: None
            List of run numbers to run. If None, all runs will be run.
        adaptive : bool, Optional, default: True
            If True, the stages will run until the simulations are equilibrated and perform analysis afterwards.
            If False, the stages will run for the specified runtime and analysis will not be performed.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this number of nanoseconds.
        runtime_constant: float, Optional, default: None
            The runtime_constant (kcal**2 mol**-2 ns*-1) only affects behaviour if running adaptively. This is used
            to calculate how long to run each simulation for based on the current uncertainty of the per-window
            free energy estimate.
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

        if runtime_constant:
            self.recursively_set_attr("runtime_constant", runtime_constant)

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
                _shutil.copy(master_run_somd, stage.input_dir)
                for lambda_window in stage.lam_windows:
                    for simulation in lambda_window.sims:
                        _shutil.copy(master_run_somd, simulation.input_dir)
