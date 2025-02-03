"""Functionality for managing legs of the calculation."""

__all__ = ["Leg"]

import glob as _glob
import logging as _logging
import os as _os
import pathlib as _pathlib
import shutil as _shutil
import subprocess as _subprocess
from time import sleep as _sleep
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import BioSimSpace.Sandpit.Exscientia as _BSS
import numpy as _np
import pandas as _pd

from ..analyse.plot import plot_convergence as _plot_convergence
from ..analyse.plot import plot_rmsds as _plot_rmsds
from ..analyse.plot import plot_sq_sem_convergence as _plot_sq_sem_convergence
from . import system_prep as _system_prep
from ._restraint import A3feRestraint as _A3feRestraint
from ._simulation_runner import SimulationRunner as _SimulationRunner
from ._utils import get_single_mol as _get_single_mol
from ._virtual_queue import Job as _Job
from ._virtual_queue import VirtualQueue as _VirtualQueue
from .enums import LegType as _LegType
from .enums import PreparationStage as _PreparationStage
from .enums import StageType as _StageType
from .stage import Stage as _Stage
from ..configuration import (
    _BaseSystemPreparationConfig,
    SlurmConfig as _SlurmConfig,
    SomdConfig as _SomdConfig,
)

from .enums import EngineType as _EngineType


class Leg(_SimulationRunner):
    """
    Class set up and run the stages of a leg of the calculation.
    """

    # Required input files for each leg type and preparation stage.
    required_input_files = {}
    for leg_type in _LegType:
        required_input_files[leg_type] = {}
        for prep_stage in _PreparationStage:
            required_input_files[leg_type][prep_stage] = (
                [] + prep_stage.get_simulation_input_files(leg_type)
            )

    required_stages = {
        _LegType.BOUND: [_StageType.RESTRAIN, _StageType.DISCHARGE, _StageType.VANISH],
        _LegType.FREE: [_StageType.DISCHARGE, _StageType.VANISH],
    }

    def __init__(
        self,
        leg_type: _LegType,
        equil_detection: str = "multiwindow",
        runtime_constant: _Optional[float] = 0.005,
        relative_simulation_cost: float = 1,
        ensemble_size: int = 5,
        base_dir: _Optional[str] = None,
        input_dir: _Optional[str] = None,
        stream_log_level: int = _logging.INFO,
        slurm_config: _Optional[_SlurmConfig] = None,
        analysis_slurm_config: _Optional[_SlurmConfig] = None,
        engine_config: _Optional[_SomdConfig] = None,
        engine_type: _EngineType = _EngineType.SOMD,
        update_paths: bool = True,
    ) -> None:
        """
        Instantiate a calculation based on files in the input dir. If leg.pkl exists in the
        base directory, the calculation will be loaded from this file and any arguments
        supplied will be overwritten.

        Parameters
        ----------
        leg_type: a3.LegType
            The type of leg to set up. Options are BOUND or FREE.
        equil_detection : str, Optional, default: "multiwindow"
            Method to use for equilibration detection. Options are:
            - "multiwindow": Use the multiwindow paired t-test method to detect equilibration.
            - "chodera": Use Chodera's method to detect equilibration.
        runtime_constant: float, Optional, default: 0.005
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
        base_dir : str, Optional, default: None
            Path to the base directory in which to set up the stages. If None,
            this is set to the current working directory.
        input_dir : str, Optional, default: None
            Path to directory containing input files for the simulations. If None, this
            is set to `current_working_directory/input`.
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            calculation object and its child objects.
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
        update_paths: bool, optional, default: True
            if true, if the simulation runner is loaded by unpickling, then
            update_paths() is called.

        Returns
        -------
        None
        """
        # Set the leg type, as this is needed in the superclass constructor
        self.leg_type = leg_type
        self.stage_type = self.required_stages[leg_type]

        super().__init__(
            base_dir=base_dir,
            input_dir=input_dir,
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
            self.stage_types = Leg.required_stages[leg_type]
            self.equil_detection = equil_detection
            self.runtime_constant = runtime_constant
            self.relative_simulation_cost = relative_simulation_cost
            self._running: bool = False
            self.jobs: _List[_Job] = []

            # Change the sign of the dg contribution to negative
            # if this is the bound leg
            if self.leg_type == _LegType.BOUND:
                self.dg_multiplier = -1

            # Validate the input
            self._validate_input()

            # Create a virtual queue for the prep jobs
            self.virtual_queue = _VirtualQueue(
                log_dir=self.base_dir, stream_log_level=self.stream_log_level
            )

            # If this is a bound leg, we want to store restraints
            if self.leg_type == _LegType.BOUND:
                self.restraints: _List[_A3feRestraint] = []

            # Save the state and update log
            self._update_log()
            self._dump()

    def __str__(self) -> str:
        return f"Leg (type = {self.leg_type.name})"

    @property
    def stages(self) -> _List[_Stage]:
        return self._sub_sim_runners

    @stages.setter
    def stages(self, value) -> None:
        self._logger.info("Modifying/ creating stages")
        self._sub_sim_runners = value

    def _validate_input(self) -> None:
        """Check that the required files are provided for the leg type and set the preparation stage
        according to the files present."""
        # Check backwards, as we care about the most advanced preparation stage
        for prep_stage in reversed(_PreparationStage):
            files_absent = False
            for file in Leg.required_input_files[self.leg_type][prep_stage]:
                if not _os.path.isfile(f"{self.input_dir}/{file}"):
                    files_absent = True
            # We have the required files for this prep stage, and this is the most
            # advanced prep stage that files are present for
            if not files_absent:
                self._logger.info(
                    f"Found all required input files for preparation stage {prep_stage.name.lower()}"
                )
                self.prep_stage = prep_stage
                return
        # We didn't find all required files for any of the prep stages
        raise ValueError(
            f"Could not find all required input files for leg type {self.leg_type.name} for "
            f"any preparation stage. Required files are: {Leg.required_input_files[self.leg_type]}"
        )

    def setup(
        self,
        sysprep_config: _Optional[_BaseSystemPreparationConfig] = None,
    ) -> None:
        """
        Set up the leg. This involves:
            - Creating the input directories
            - Parameterising the input structures
            - Solvating the input structures
            - Minimising the input structures
            - Heating the input structures
            - Running pre-equilibration simulations (and extracting the
              restraints for the bound leg)
            - Creating the Stage objects

        Parameters
        ----------
        sysprep_config: Optional[BaseSystemPreparationConfig], default: None
            Configuration object for the setup of the leg. If None, the default configuration
            is used.
        """
        self._logger.info("Setting up leg...")

        # First, we need to save the config to the input directory so that this can be reloaded
        # by the slurm jobs.
        cfg = (
            sysprep_config
            if sysprep_config is not None
            else _BaseSystemPreparationConfig()
        )
        cfg.dump(self.input_dir, self.leg_type)

        # Create input directories, parameterise, solvate, minimise, heat and preequil, all
        # depending on the input files present.

        # First, create the input directories
        self.create_stage_input_dirs()

        # Then prepare as required according to the preparation stage
        prep_stage_order = [prep_stage for prep_stage in _PreparationStage]
        while self.prep_stage != _PreparationStage.PREEQUILIBRATED:
            next_prep_stage = prep_stage_order[self.prep_stage.value]
            self._run_prep_stage(next_prep_stage, cfg.slurm)

        # Run separate equilibration simulations for each of the repeats and
        # extract the final structures to give a diverse ensemble of starting
        # conformations. For the bound leg, this also extracts the restraints.
        system = self.run_ensemble_equilibration(sysprep_config=cfg)

        # Write input files
        self.setup_stages(system, sys_prep_config=cfg)

        # Store restraints used. Currenly (and unlike previous versions) we only allow
        # the same restraint to be used for all.
        first_restr = self.restraints[0]
        self.restraints = [first_restr for _ in range(self.ensemble_size)]

        # Create the Stage objects, which automatically set themselves up
        for stage_type in self.required_stages[self.leg_type]:
            self.stages.append(
                _Stage(
                    stage_type=stage_type,
                    equil_detection=self.equil_detection,
                    runtime_constant=self.runtime_constant,
                    relative_simulation_cost=self.relative_simulation_cost,
                    ensemble_size=self.ensemble_size,
                    lambda_values=self.engine_config.default_lambda_values[
                        self.leg_type
                    ][stage_type],
                    base_dir=self.stage_input_dirs[stage_type].replace("/input", ""),
                    input_dir=self.stage_input_dirs[stage_type],
                    output_dir=self.stage_input_dirs[stage_type].replace(
                        "input", "output"
                    ),
                    stream_log_level=self.stream_log_level,
                    slurm_config=self.slurm_config,
                    analysis_slurm_config=self.analysis_slurm_config,
                    engine_config=self.engine_config.copy(),
                    engine_type=self.engine_type,
                )
            )

        self._logger.info("Setup complete.")
        # Save state
        self._dump()

    def get_optimal_lam_vals(
        self,
        simtime: _Optional[float] = 0.1,
        er_type: str = "root_var",
        delta_er: float = 1,
        set_relative_sim_cost: bool = True,
        reference_sim_cost: float = 0.21,
        run_nos: _List[int] = [1],
    ) -> None:
        """
        Determine the optimal lambda windows for each stage of the leg
        by running short simulations at each lambda value and analysing them,
        using only a single run. Optionally, determine the simulation cost
        and recursively set the relative simulation cost according reference_sim_cost.

        Parameters
        ----------
        simtime : float, Optional, default: 0.1
            The length of the short simulations to run, in ns. If None is provided,
            it is assumed that the simulations have already been run and the
            optimal lambda values are extracted from the output files.
        er_type: str, Optional, default="root_var"
            Whether to integrate the standard error of the mean ("sem") or root
            variance of the gradients ("root_var") to calculate the optimal
            lambda values.
        delta_er : float, default=1
            If er_type == "root_var", the desired integrated root variance of the gradients
            between each lambda value, in kcal mol^(-1). If er_type == "sem", the
            desired integrated standard error of the mean of the gradients between each lambda
            value, in kcal mol^(-1) ns^(1/2). A sensible default for root_var is 1 kcal mol-1,
            and 0.1 kcal mol-1 ns^(1/2) for sem.
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
            we only analyse 1. If using delta_er = "sem", more than one run must be specified.

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

        # Check that the leg has been set up
        if not hasattr(self, "stages"):
            raise ValueError(
                "The leg has not been set up yet. Please call setup() first."
            )

        # If simtime is not None, run short simulations
        if simtime is not None:
            self._logger.info(
                f"Running short simulations for {simtime} ns to determine optimal lambda windows..."
            )
            self.run(adaptive=False, runtime=simtime, run_nos=run_nos)
            self.wait()
        else:
            self._logger.info(
                "Simulation time is not 0 - assuming that short simulations have already been run and"
                " extracting optimal lambda values from output files..."
            )

        # Now extract the optimal lambda values
        self._logger.info(
            f"Determining optimal lambda windows for each stage with er_type = {er_type} and delta_er = {delta_er}..."
        )

        # If requested, get the total simulation cost for the leg
        if set_relative_sim_cost:
            mean_simulation_cost = self.get_tot_gpu_time(
                run_nos=run_nos
            ) / self.get_tot_simtime(run_nos=run_nos)

            relative_sim_cost = mean_simulation_cost / reference_sim_cost
            self.recursively_set_attr(
                "relative_simulation_cost", relative_sim_cost, force=True
            )

        for stage in self.stages:
            self._logger.info(f"Determining optimal lambda windows for {stage}...")
            optimal_lam_vals = stage.get_optimal_lam_vals(
                er_type=er_type, delta_er=delta_er, run_nos=run_nos
            )
            # Create new LamWindow objects with the optimal lambda values, then save data
            stage.lam_vals = list(optimal_lam_vals)
            stage.update(
                save_name="lam_val_determination"
            )  # This deletes all of the old LamWindow objects and creates a new output dir

        # Save state
        self._dump()

    def create_stage_input_dirs(self) -> _Dict[_StageType, str]:
        """
        Create the input directories for each stage.

        Returns
        -------
        stage_input_dirs : Dict[StageType, str]
            Dictionary mapping each stage type to the path to its input directory.
        """
        self._logger.info("Creating stage input directories...")
        stage_input_dirs = {}
        for stage_type in self.stage_types:
            input_dir = f"{self.base_dir}/{stage_type.name.lower()}/input"
            _pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)
            stage_input_dirs[stage_type] = input_dir

        self.stage_input_dirs = stage_input_dirs

        return stage_input_dirs

    def _run_prep_stage(
        self, next_prep_stage: _PreparationStage, slurm: bool = True
    ) -> None:
        """
        Run a preparation stage.

        Parameters
        ----------
        next_prep_stage: PreparationStage
            The preparation stage to prepare the input structures to.

        slurm : bool, optional, default=True
            Whether to use SLURM to run the solvation job, by default True.
        """
        fn = next_prep_stage.prep_fn
        job_name = next_prep_stage.name.lower()
        self._logger.info(f"Preparing {job_name} input structures...")

        if slurm:
            self._logger.info(
                f"Preparing {job_name} input structures. Submitting through SLURM..."
            )
            self._run_slurm(
                sys_prep_fn=fn,
                leg_type=self.leg_type,
                engine_type=self.engine_type,
                run_dir=self.input_dir,
                wait=True,
                job_name=job_name,
            )

            # Check that the required input files have been produced, since slurm can fail silently
            for file in next_prep_stage.get_simulation_input_files(self.leg_type):
                if not _os.path.isfile(f"{self.input_dir}/{file}"):
                    raise RuntimeError(
                        f"SLURM job failed to produce {file}. Please check the output of the "
                        f"last slurm log in {self.input_dir} directory for error."
                    )
        else:
            self._logger.info(
                f"Preparing {job_name} input structures. Not using SLURM..."
            )
            fn(
                leg_type=self.leg_type,
                engine_type=self.engine_type,
                input_dir=self.input_dir,
                output_dir=self.input_dir,
            )

        # Update the preparation stage
        self.prep_stage = next_prep_stage

    def run_ensemble_equilibration(
        self,
        sysprep_config: _BaseSystemPreparationConfig,
    ) -> _BSS._SireWrappers._system.System:
        """
        Run 5 ns simulations with SOMD for each of the ensemble_size runs and extract the final structures
        to use as diverse starting points for the production runs. If this is the bound leg, the restraints
        will also be extracted from the simulations and saved to a file. The simulations will be run in a
        subdirectory of the stage base directory called ensemble_equilibration, and the restraints and
        final coordinates will be saved here.

        Parameters
        ----------
        sysprep_config: BaseSystemPreparationConfig
            Configuration object for the setup of the leg.
        """
        # Generate output dirs and copy over the input
        outdirs = [
            f"{self.base_dir}/ensemble_equilibration_{i + 1}"
            for i in range(self.ensemble_size)
        ]

        # Check if we have already run any ensemble equilibration simulations
        outdirs_to_run = [outdir for outdir in outdirs if not _os.path.isdir(outdir)]
        outdirs_already_run = [outdir for outdir in outdirs if _os.path.isdir(outdir)]
        self._logger.info(
            f"Found {len(outdirs_already_run)} ensemble equilibration directories already run."
        )

        for outdir in outdirs_to_run:
            _subprocess.run(["mkdir", "-p", outdir], check=True)
            for input_file in [
                f"{self.input_dir}/{ifile}"
                for ifile in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                    self.leg_type
                )
            ]:
                _subprocess.run(["cp", "-r", input_file, outdir], check=True)

            # Also write a pickle of the config to the output directory
            sysprep_config.dump(outdir, self.leg_type)

        fn = _system_prep.run_ensemble_equilibration

        if sysprep_config.slurm:
            # For each ensemble member to be run, run a 5 ns simulation in a seperate directory
            for i, outdir in enumerate(outdirs_to_run):
                self._logger.info(
                    f"Running ensemble equilibration {i + 1} of {len(outdirs_to_run)}. Submitting through SLURM..."
                )
                self._run_slurm(
                    sys_prep_fn=fn,
                    leg_type=self.leg_type,
                    engine_type=self.engine_type,
                    run_dir=outdir,
                    wait=False,
                    job_name="ensemble_equilibration",
                )

            self.virtual_queue.wait()  # Wait for all jobs to finish

            # Check that the required input files have been produced, since slurm can fail silently
            for i, outdir in enumerate(outdirs_to_run):
                for file in (
                    _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                        self.leg_type
                    )
                    + ["somd.rst7"]
                ):
                    if not _os.path.isfile(f"{outdir}/{file}"):
                        raise RuntimeError(
                            f"SLURM job failed to produce {file}. Please check the output of the "
                            f"last slurm log in {outdir} directory for errors."
                        )

        else:  # Not slurm
            for i, outdir in enumerate(outdirs_to_run):
                self._logger.info(
                    f"Running ensemble equilibration {i + 1} of {len(outdirs_to_run)}..."
                )
                fn(
                    leg_type=self.leg_type,
                    engine_type=self.engine_type,
                    input_dir=self.input_dir,
                    output_dir=outdir,
                )

        # Give the output files unique names
        for i, outdir in enumerate(outdirs_to_run):
            _subprocess.run(
                ["mv", f"{outdir}/somd.rst7", f"{outdir}/somd_{i + 1}.rst7"], check=True
            )

        # Load the system and mark the ligand to be decoupled
        self._logger.info("Loading pre-equilibrated system...")
        pre_equilibrated_system = _BSS.IO.readMolecules(
            [
                f"{self.input_dir}/{file}"
                for file in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                    self.leg_type
                )
            ]
        )

        # Mark the ligand to be decoupled so the restraints searching algorithm works
        lig = _BSS.Align.decouple(pre_equilibrated_system[0], intramol=True)
        pre_equilibrated_system.updateMolecule(0, lig)

        # If this is the bound leg, search for restraints
        if self.leg_type == _LegType.BOUND:
            # For each run, load the trajectory and extract the restraints
            for i, outdir in enumerate(outdirs):
                self._logger.info(f"Loading trajectory for run {i + 1}...")
                top_file = f"{self.input_dir}/{_PreparationStage.PREEQUILIBRATED.get_simulation_input_files(self.leg_type)[0]}"
                traj = _BSS.Trajectory.Trajectory(
                    topology=top_file,
                    trajectory=f"{outdir}/gromacs.xtc",
                    system=pre_equilibrated_system,
                )
                self._logger.info(f"Selecting restraints for run {i + 1}...")
                restraint = _BSS.FreeEnergy.RestraintSearch.analyse(
                    method="BSS",
                    system=pre_equilibrated_system,
                    traj=traj,
                    work_dir=outdir,
                    temperature=298.15 * _BSS.Units.Temperature.kelvin,
                    append_to_ligand_selection=sysprep_config.append_to_ligand_selection,
                )

                # Check that we actually generated a restraint
                if restraint is None:
                    raise ValueError(f"No restraints found for run {i + 1}.")

                # Save the restraints to a text file and store within the Leg object
                with open(f"{outdir}/restraint_{i + 1}.txt", "w") as f:
                    f.write(restraint.toString(engine="SOMD"))  # type: ignore
                self.restraints.append(restraint)

            return pre_equilibrated_system

        else:  # Free leg
            return pre_equilibrated_system

    def setup_stages(
        self,
        pre_equilibrated_system: _BSS._SireWrappers._system.System,  # type: ignore
        sys_prep_config: _BaseSystemPreparationConfig,
    ) -> _Dict[_StageType, _SomdConfig]:
        """
        Set up the engine configurations for each stage of the leg.

        Parameters
        ----------
        pre_equilibrated_system: _BSS._SireWrappers._system.System
            The equilibrated system to run further equilinration on. The final coordinates
            are then used as input for each of the individual runs.
        config: BaseSystemPreparationConfig
            Configuration object for the setup of the leg.

        Returns
        -------
        Dict[StageType, EngineConfig]
            Dictionary mapping stage types to their Engine configurations
        """
        # Get the charge of the ligand
        lig = _get_single_mol(pre_equilibrated_system, "LIG")
        lig_charge = round(lig.charge().value())

        if lig_charge != 0:
            self._logger.info(
                f"The ligand has a charge of {lig_charge}. Using co-alchemical ion approach to maintain neutrality. "
                "Please note: The cutoff type should be PME and the cutoff length can be adjusted to other values, e.g., 10 Å."
            )
        # Figure out where the ligand is in the system
        perturbed_resnum = pre_equilibrated_system.getIndex(lig) + 1

        # Dummy values get overwritten later
        dummy_runtime = 0.001  # ns
        dummy_lam_vals = [0.0]
        if not hasattr(self, "stage_input_dirs"):
            raise AttributeError("No stage input directories have been set.")

        stage_configs = {}
        for stage_type, stage_input_dir in self.stage_input_dirs.items():
            self._logger.info(
                f"Setting up {self.leg_type.name} leg {stage_type.name} stage"
            )
            restraint = self.restraints[0] if self.leg_type == _LegType.BOUND else None
            protocol = _BSS.Protocol.FreeEnergy(
                runtime=dummy_runtime * _BSS.Units.Time.nanosecond,  # type: ignore
                lam_vals=dummy_lam_vals,
                perturbation_type=stage_type.bss_perturbation_type,
            )
            self._logger.info(f"Perturbation type: {stage_type.bss_perturbation_type}")
            # Ensure we remove the velocites to avoid RST7 file writing issues, as before
            _BSS.FreeEnergy.AlchemicalFreeEnergy(
                pre_equilibrated_system,
                protocol,
                engine="SOMD",
                restraint=restraint,
                work_dir=stage_input_dir,
                setup_only=True,
                property_map={"velocity": "foo"},
            )  # We will run outside of BSS

            # Copy input written by BSS to the stage input directory
            for file in _glob.glob(f"{stage_input_dir}/lambda_0.0000/*"):
                _shutil.copy(file, stage_input_dir)
            for file in _glob.glob(f"{stage_input_dir}/lambda_*"):
                _subprocess.run(["rm", "-rf", file], check=True)

            # Create a seperate config for this stage
            stage_config = self.engine_config.copy()

            # Copy the final coordinates from the ensemble equilibration stage to the stage input directory
            # and, if this is the bound stage, read in the restraints
            for i in range(self.ensemble_size):
                ens_equil_output_dir = f"{self.base_dir}/ensemble_equilibration_{i + 1}"
                coordinates_file = f"{ens_equil_output_dir}/somd_{i + 1}.rst7"
                _shutil.copy(coordinates_file, f"{stage_input_dir}/somd_{i + 1}.rst7")

                if self.leg_type == _LegType.BOUND:
                    # Read in the first restraint file
                    with open(
                        f"{ens_equil_output_dir}/restraint_{i + 1}.txt", "r"
                    ) as f:
                        lines = f.readlines()
                        restraint_type, restraint_dict = [
                            item.strip() for item in lines[0].split("=")
                        ]

                    if restraint_type != "boresch restraints dictionary":
                        raise ValueError(
                            f"Only Boresch restraints are supported. Found {restraint_type} restraints."
                        )

                    stage_config.boresch_restraints_dictionary = restraint_dict

            # Set configuration options
            stage_config.perturbed_residue_number = perturbed_resnum
            stage_config.use_boresch_restraints = self.leg_type == _LegType.BOUND
            stage_config.turn_on_receptor_ligand_restraints = (
                self.leg_type == _LegType.BOUND
            )
            stage_config.ligand_charge = (
                -lig_charge
            )  # Use co-alchemical ion approach when there is a charge difference
            stage_configs[stage_type] = stage_config

        # We no longer need to store the large BSS restraint classes.
        self._lighten_restraints()

        return stage_configs

    def _run_slurm(
        self,
        sys_prep_fn: _Callable,
        leg_type: _LegType,
        engine_type: _EngineType,
        wait: bool,
        run_dir: str,
        job_name: str,
        short: _Optional[bool] = None,
    ) -> None:
        """
        Run the supplied function through a slurm job. The function must be in
        the system_prep module.

        Parameters
        ----------
        sys_prep_fn: Callable
            The function to run through slurm.
        leg_type: _LegType
            The type of leg to run the function for.
        engine_type: _EngineType
            The type of engine to run the function for.
        wait: bool
            If True, the function will wait for the job to complete before returning.
        run_dir: str
            The directory to run the job in.
        job_name: str
            The name of the job.
        short: Optional[bool], default=None
            Whether to supply short=True to the slurm job.

        Returns
        -------
        None
        """
        cmd = f"python -c 'from a3fe.run.system_prep import {sys_prep_fn.__name__};"
        cmd += "from a3fe import LegType, EngineType;"
        if short is not None:
            cmd += f'{sys_prep_fn.__name__}(leg_type={leg_type}, engine_type={engine_type}, input_dir=".", output_dir=".", short={short})\''
        else:
            cmd += f'{sys_prep_fn.__name__}(leg_type={leg_type}, engine_type={engine_type}, input_dir=".", output_dir=".")\''

        cmd_list = self.slurm_config.get_submission_cmds(
            cmd=cmd, run_dir=run_dir, script_name=job_name
        )
        slurm_file_base = self.slurm_config.get_slurm_output_file_base(run_dir=run_dir)
        job = self.virtual_queue.submit(cmd_list, slurm_file_base=slurm_file_base)
        self._logger.info(f"Submitted job {job}")
        self.jobs.append(job)
        # Update the virtual queue to submit the job
        self.virtual_queue.update()

        # Always wait untit the job is submitted to the real slurm queue
        while self.virtual_queue._pre_queue:
            self._logger.info(
                f"Waiting for job {job} to be submitted to the real slurm queue"
            )
            _sleep(5 * 60)
            self.virtual_queue.update()

        # Wait for the job to complete if we've specified wait
        if wait:
            while job in self.virtual_queue.queue:
                self._logger.info(f"Waiting for job {job} to complete")
                _sleep(30)
                self.virtual_queue.update()

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
        - :math:`\\sigma_{\\mathrm{Current}}(\\Delta \\widehat{F}_k)` is the current uncertainty in the free energy change contribution for lambda window :math:`k`. This is estimated from inter-run deviations.
        - :math:`\\Delta \\widehat{F}_k` is the free energy change contribution for lambda window :math:`k`

        Parameters
        ----------
        run_nos : Optional[List[int]], default=None
            If specified, only run the specified runs. Otherwise, run all runs.
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
        run_nos = self._get_valid_run_nos(run_nos)
        if runtime_constant:
            self.recursively_set_attr("runtime_constant", runtime_constant, silent=True)

        self._logger.info(
            f"Running run numbers {run_nos} for {self.__class__.__name__}..."
        )
        for stage in self.stages:
            stage.run(run_nos=run_nos, adaptive=adaptive, runtime=runtime)
            if not parallel:
                stage.wait()

    def analyse(
        self,
        slurm: bool = False,
        run_nos: _Optional[_List[int]] = None,
        subsampling=False,
        fraction: float = 1,
        plot_rmsds: bool = False,
    ) -> _Tuple[_np.ndarray, _np.ndarray]:
        """
        Analyse the leg and any sub-simulations, and
        return the overall free energy change.

        Parameters
        ----------
        slurm : bool, optional, default=False
            Whether to use SLURM to run the analysis, by default False.
        run_nos : List[int], Optional, default=None
            A list of the run numbers to analyse. If None, all runs are analysed.
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
        dg_overall : np.ndarray
            The overall free energy change for each of the
            ensemble size repeats.
        er_overall : np.ndarray
            The overall error for each of the ensemble size
            repeats.
        """
        run_nos = self._get_valid_run_nos(run_nos)

        dg_overall, er_overall = super().analyse(
            slurm=slurm,
            run_nos=run_nos,
            subsampling=subsampling,
            fraction=fraction,
            plot_rmsds=plot_rmsds,
        )

        if self.leg_type == _LegType.BOUND:
            if plot_rmsds:
                # We can plot the RMSD of the protein and the distance-to-bound configuration
                self._logger.info(
                    "Analysing RMSD of protein and computing distance-to-bound configuration..."
                )
                selections = [
                    ("protein", None),
                    # This is the distance to bound configuration -
                    # the RMSD of the ligand in the frame of reference of the protein.
                    (
                        # "protein and (around 5 resname LIG)",
                        "protein",
                        "resname LIG and (not name H*)",
                    ),
                ]
                for stage in self.stages:
                    for selection, group_selection in selections:
                        _plot_rmsds(
                            lam_windows=stage.lam_windows,
                            output_dir=stage.output_dir,
                            selection=selection,
                            group_selection=group_selection,
                        )

            # We need to add on the restraint corrections. There are no errors associated with these.
            rest_corrs = _np.array(
                [
                    self.restraints[run_no - 1].getCorrection().value()
                    for run_no in run_nos
                ]
            )
            # Write out restraint
            with open(f"{self.output_dir}/restraint_corrections.txt", "w") as ofile:
                for run_no in run_nos:
                    ofile.write(
                        f"{run_no} {self.restraints[run_no - 1].getCorrection().value()} kcal / mol\n"
                    )
            self._logger.info(f"Restraint corrections: {rest_corrs} kcal / mol")

            # Correct overall DG
            dg_overall += rest_corrs

            # Update the internally stored dGs after the restraint corrections and save the restraint corrections
            self._delta_g = dg_overall
            self._restratint_corrections = rest_corrs

        return dg_overall, er_overall

    def get_results_df(
        self, save_csv: bool = True, add_sub_sim_runners: bool = True
    ) -> _pd.DataFrame:
        """
        Return the results in dataframe format

        Parameters
        ----------
        save_csv : bool, optional, default=True
            Whether to save the results as a csv file

        add_sub_sim_runners : bool, optional, default=True
            Whether to show the results from the sub-simulation runners.

        Returns
        -------
        results_df : pd.DataFrame
            A dataframe containing the results
        """
        results_df = super().get_results_df(
            save_csv=save_csv, add_sub_sim_runners=add_sub_sim_runners
        )
        # Add the restraint corrections if this is the bound leg
        if self.leg_type == _LegType.BOUND:
            if not hasattr(self, "_restratint_corrections"):
                raise AttributeError(
                    "No restraint corrections have been calculated. Please run analyse first."
                )
            new_row = {
                "dg / kcal mol-1": round(self._restratint_corrections.mean(), 2),
                "dg_95_ci / kcal mol-1": 0,
                "tot_simtime / ns": 0,
                "tot_gpu_time / GPU hours": 0,
            }
            results_df.loc["restraint_corr"] = new_row

            self._logger.warning(
                "Remember to add in any required symmetry corrections."
            )

        return results_df

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
        Leg against total simulation time. Also plot this. Keep
        this separate from analyse as it is expensive to run.

        Parameters
        ----------
        slurm: bool, optional, default=False
            Whether to use slurm for the analysis.
        run_nos : Optional[List[int]], default=None
            If specified, only analyse the specified runs. Otherwise, analyse all runs.
        mode : str, optional, default="cumulative"
            "cumulative" or "block". The type of averaging to use. In both cases,
            20 MBAR evaluations are performed.
        fraction: float, optional, default=1
            The fraction of the data to use for analysis. For example, if
            fraction=0.5, only the first half of the data will be used for
            analysis. If fraction=1, all data will be used. Note that unequilibrated
            data is discarded from the beginning of simulations in all cases.
        equilibrated: bool, optional, default=True
            Whether to analyse only the equilibrated data (True) or all data (False)

        Returns
        -------
        fracts : np.ndarray
            The fraction of the total (equilibrated) simulation time for each value of dg_overall.
        dg_overall : np.ndarray
            The overall free energy change for the {self.__class__.__name__} for
            each value of total (equilibrated) simtime for each of the ensemble size repeats.
        """
        run_nos = self._get_valid_run_nos(run_nos)

        self._logger.info(f"Analysing convergence of {self.__class__.__name__}...")

        # Get the dg_overall in terms of fraction of the total simulation time
        # Use steps of 5 % of the total simulation time
        fracts = _np.arange(0.05, 1.05, 0.05)
        # Only analyse up to specified fraction of total simulation data
        fracts = fracts * fraction
        # Create an array to store the overall free energy change
        dg_overall = _np.zeros((len(run_nos), len(fracts)))

        # Now add up the data for each of the sub-simulation runners
        for sub_sim_runner in self._sub_sim_runners:
            _, dgs = sub_sim_runner.analyse_convergence(
                slurm=slurm,
                run_nos=run_nos,
                mode=mode,
                fraction=fraction,
                equilibrated=equilibrated,
            )
            # Decide if the component should be added or subtracted
            # according to the dg_multiplier attribute
            dg_overall += dgs * sub_sim_runner.dg_multiplier

        if self.leg_type == _LegType.BOUND:
            # We need to add on the restraint corrections. There are no errors associated with these.
            rest_corrs = _np.array(
                [
                    self.restraints[run_no - 1].getCorrection().value()
                    for run_no in run_nos
                ]
            )
            self._logger.info(
                f"Correcting convergence plots with restraint corrections: {rest_corrs}"
            )
            # Make sure the shape is correct
            rest_corrs = [rest_corr * _np.ones(len(fracts)) for rest_corr in rest_corrs]
            dg_overall += rest_corrs

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

        return fracts, dg_overall

    def lighten(self) -> None:
        """Lighten the leg by deleting ensemble equilibration output
        and lightening all sub-simulation runners"""
        # Remove the ensemble equilibration directories
        for direct in _pathlib.Path(self.base_dir).glob("ensemble_equilibration*"):
            self._logger.info(f"Deleting {direct}")
            _subprocess.run(["rm", "-rf", direct])

        # Lighten all the sub-simulation runners
        if hasattr(self, "_sub_sim_runners"):
            for sub_sim_runner in self._sub_sim_runners:
                sub_sim_runner.lighten()

    def _lighten_restraints(self) -> None:
        """
        Replace the BioSimSpace restraints with a light-weight version
        which does not store entire systems in memory.
        """
        if self.leg_type == _LegType.BOUND:
            light_restraints = [
                _A3feRestraint(restraint) for restraint in self.restraints
            ]
            self.restraints = light_restraints
