"""Functionality for managing legs of the calculation."""

import glob as _glob
import BioSimSpace.Sandpit.Exscientia as _BSS
from functools import partial as _partial
import logging as _logging
import numpy as _np
import os as _os
import pathlib as _pathlib
import shutil as _shutil
from time import sleep as _sleep
import subprocess as _subprocess
from typing import (
    Dict as _Dict,
    List as _List,
    Tuple as _Tuple,
    Any as _Any,
    Optional as _Optional,
    Callable as _Callable,
)

from ..analyse.plot import plot_convergence as _plot_convergence
from .enums import (
    LegType as _LegType,
    PreparationStage as _PreparationStage,
    StageType as _StageType,
)
from .stage import Stage as _Stage
from .system_prep import (
    parameterise_input as _sysprep_parameterise_input,
    slurm_parameterise_bound as _slurm_parameterise_bound,
    slurm_parameterise_free as _slurm_parameterise_free,
    solvate_input as _sysprep_solvate_input,
    slurm_solvate_bound as _slurm_solvate_bound,
    slurm_solvate_free as _slurm_solvate_free,
    minimise_input as _sysprep_minimise_input,
    slurm_minimise_bound as _slurm_minimise_bound,
    slurm_minimise_free as _slurm_minimise_free,
    heat_and_preequil_input as _sysprep_heat_and_preequil_input,
    slurm_heat_and_preequil_bound as _slurm_heat_and_preequil_bound,
    slurm_heat_and_preequil_free as _slurm_heat_and_preequil_free,
    run_ensemble_equilibration as _sysprep_run_ensemble_equilibration,
    slurm_ensemble_equilibration_bound as _slurm_ensemble_equilibration_bound,
    slurm_ensemble_equilibration_free as _slurm_ensemble_equilibration_free,
    slurm_ensemble_equilibration_bound_short as _slurm_ensemble_equilibration_bound_short,
    slurm_ensemble_equilibration_free_short as _slurm_ensemble_equilibration_free_short,
)
from ..read._process_slurm_files import get_slurm_file_base as _get_slurm_file_base
from ..read._process_somd_files import (
    read_simfile_option as _read_simfile_option,
    write_simfile_option as _write_simfile_option,
)
from ..read._process_bss_systems import rename_lig as _rename_lig
from ._simulation_runner import SimulationRunner as _SimulationRunner
from ._virtual_queue import VirtualQueue as _VirtualQueue, Job as _Job


class Leg(_SimulationRunner):
    """
    Class set up and run the stages of a leg of the calculation.
    """

    # Required input files for each leg type and preparation stage.
    required_input_files = {}
    for leg_type in _LegType:
        required_input_files[leg_type] = {}
        for prep_stage in _PreparationStage:
            required_input_files[leg_type][prep_stage] = [
                "run_somd.sh",
                "template_config.cfg",
            ] + prep_stage.get_simulation_input_files(leg_type)

    required_stages = {
        _LegType.BOUND: [_StageType.RESTRAIN, _StageType.DISCHARGE, _StageType.VANISH],
        _LegType.FREE: [_StageType.DISCHARGE, _StageType.VANISH],
    }

    default_lambda_values = {
        _LegType.BOUND: {
            _StageType.RESTRAIN: [0.000, 0.125, 0.250, 0.375, 0.500, 1.000],
            _StageType.DISCHARGE: [
                0.000,
                0.143,
                0.286,
                0.429,
                0.571,
                0.714,
                0.857,
                1.000,
            ],
            _StageType.VANISH: [
                0.000,
                0.025,
                0.050,
                0.075,
                0.100,
                0.125,
                0.150,
                0.175,
                0.200,
                0.225,
                0.250,
                0.275,
                0.300,
                0.325,
                0.350,
                0.375,
                0.400,
                0.425,
                0.450,
                0.475,
                0.500,
                0.525,
                0.550,
                0.575,
                0.600,
                0.625,
                0.650,
                0.675,
                0.700,
                0.725,
                0.750,
                0.800,
                0.850,
                0.900,
                0.950,
                1.000,
            ],
        },
        _LegType.FREE: {
            _StageType.DISCHARGE: [
                0.000,
                0.143,
                0.286,
                0.429,
                0.571,
                0.714,
                0.857,
                1.000,
            ],
            _StageType.VANISH: [
                0.000,
                0.028,
                0.056,
                0.111,
                0.167,
                0.222,
                0.278,
                0.333,
                0.389,
                0.444,
                0.500,
                0.556,
                0.611,
                0.667,
                0.722,
                0.778,
                0.889,
                1.000,
            ],
        },
    }

    def __init__(
        self,
        leg_type: _LegType,
        block_size: float = 1,
        equil_detection: str = "block_gradient",
        gradient_threshold: _Optional[float] = None,
        runtime_constant: _Optional[float] = None,
        ensemble_size: int = 5,
        base_dir: _Optional[str] = None,
        input_dir: _Optional[str] = None,
        stream_log_level: int = _logging.INFO,
        update_paths: bool = True,
    ) -> None:
        """
        Instantiate a calculation based on files in the input dir. If leg.pkl exists in the
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
        runtime_constant : float, Optional, default: None
            The runtime constant to use for the calculation. This must be supplied if running
            adaptively. Each window is run until the SEM**2 / runtime >= runtime_constant.
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
        update_paths: bool, optional, default: True
            if true, if the simulation runner is loaded by unpickling, then
            update_paths() is called.

        Returns
        -------
        None
        """
        # Set the leg type, as this is needed in the superclass constructor
        self.leg_type = leg_type

        super().__init__(
            base_dir=base_dir,
            input_dir=input_dir,
            stream_log_level=stream_log_level,
            ensemble_size=ensemble_size,
            update_paths=update_paths,
        )

        if not self.loaded_from_pickle:
            self.stage_types = Leg.required_stages[leg_type]
            self.block_size = block_size
            self.equil_detection = equil_detection
            self.gradient_threshold = gradient_threshold
            self.runtime_constant = runtime_constant
            self._running: bool = False
            self.jobs: _List[_Job] = []

            # Change the sign of the dg contribution to negative
            # if this is the bound leg
            if self.leg_type == _LegType.BOUND:
                self.dg_multiplier = -1

            # Validate the input
            self._validate_input()

            # Create a virtual queue for the prep jobs
            self.virtual_queue = _VirtualQueue(log_dir=self.base_dir)

            # If this is a bound leg, we want to store restraints
            if self.leg_type == _LegType.BOUND:
                self.restraints = []

            # Save the state and update log
            self._update_log()
            self._dump()

    def __str__(self) -> str:
        return f"Leg (type = {self.leg_type.name})"

    @property
    def stages(self) -> _List[_Stage]:
        return self._sub_sim_runners

    @stages.setter
    def legs(self, value) -> None:
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
        slurm: bool = True,
        append_to_ligand_selection: str = "",
        use_same_restraints: bool = True,
        short_ensemble_equil: bool = False,
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
        slurm : bool, default: True
            If True, the setup jobs will be run through SLURM.
        append_to_ligand_selection: str, optional, default = ""
            If this is a bound leg, this appends the supplied string to the default atom
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
            not recommended for production runs, but can be useful for testing.
        """
        self._logger.info("Setting up leg...")
        # Create input directories, parameterise, solvate, minimise, heat and preequil, all
        # depending on the input files present.

        # First, create the input directories
        self.create_stage_input_dirs()

        # Then prepare as required according to the preparation stage
        if self.prep_stage == _PreparationStage.STRUCTURES_ONLY:
            self.parameterise_input(slurm=slurm)
        if self.prep_stage == _PreparationStage.PARAMETERISED:
            self.solvate_input(slurm=slurm)  # This also adds ions
        if self.prep_stage == _PreparationStage.SOLVATED:
            system = self.minimise_input(slurm=slurm)
        if self.prep_stage == _PreparationStage.MINIMISED:
            system = self.heat_and_preequil_input(slurm=slurm)
        if self.prep_stage == _PreparationStage.PREEQUILIBRATED:
            # Run separate equilibration simulations for each of the repeats and
            # extract the final structures to give a diverse ensemble of starting
            # conformations. For the bound leg, this also extracts the restraints.
            system = self.run_ensemble_equilibration(
                slurm=slurm,
                append_to_ligand_selection=append_to_ligand_selection,
                short_ensemble_equil=short_ensemble_equil,
            )

        # Write input files
        self.write_input_files(system, use_same_restraints=use_same_restraints)

        # Make sure the stored restraints reflect the restraints used. TODO:
        # make this more robust my using the SOMD functionality to extract
        # results from the simfiles
        if self.leg_type == _LegType.BOUND and use_same_restraints:
            # Use the first restraints
            first_restr = self.restraints[0]
            self.restraints = [first_restr for _ in range(self.ensemble_size)]

        # Create the Stage objects, which automatically set themselves up
        for stage_type in self.required_stages[self.leg_type]:
            self.stages.append(
                _Stage(
                    stage_type=stage_type,
                    block_size=self.block_size,
                    equil_detection=self.equil_detection,
                    gradient_threshold=self.gradient_threshold,
                    runtime_constant=self.runtime_constant,
                    ensemble_size=self.ensemble_size,
                    lambda_values=Leg.default_lambda_values[self.leg_type][stage_type],
                    base_dir=self.stage_input_dirs[stage_type].replace("/input", ""),
                    input_dir=self.stage_input_dirs[stage_type],
                    output_dir=self.stage_input_dirs[stage_type].replace(
                        "input", "output"
                    ),
                    stream_log_level=self.stream_log_level,
                )
            )

        self._logger.info("Setup complete.")
        # Save state
        self._dump()

    def get_optimal_lam_vals(
        self,
        simtime: _Optional[float] = 0.1,
        er_type: str = "sem",
        delta_er: float = 0.1,
        run_nos: _List[int] = [1],
    ) -> None:
        """
        Determine the optimal lambda windows for each stage of the leg
        by running short simulations at each lambda value and analysing them,
        using only a single run.

        Parameters
        ----------
        simtime : float, Optional, default: 0.1
            The length of the short simulations to run, in ns. If None is provided,
            it is assumed that the simulations have already been run and the
            optimal lambda values are extracted from the output files.
        er_type: str, optional, default="sem"
            Whether to integrate the standard error of the mean ("sem") or root
            variance of the gradients ("root_var") to calculate the optimal
            lambda values.
        delta_er : float, default=0.1
            If er_type == "root_var", the desired integrated root variance of the gradients
            between each lambda value, in kcal mol^(-1). If er_type == "sem", the
            desired integrated standard error of the mean of the gradients between each lambda
            value, in kcal mol^(-1) ns^(1/2). A sensible default for root_var is 1 kcal mol-1.
            If not provided, the number of lambda windows must be provided with n_lam_vals.
        run_nos : List[int], optional, default=[1]
            The run numbers to use for the calculation. Only 1 is run by default, so by default
            we only analyse 1.

        Returns
        -------
        None
        """
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
        for stage in self.stages:
            self._logger.info(f"Determining optimal lambda windows for {stage}...")
            optimal_lam_vals = stage.get_optimal_lam_vals(
                er_type=er_type, delta_er=delta_er, run_nos=[1]
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

    # TODO: Avoid all the code duplication in the following functions

    def parameterise_input(self, slurm: bool = True) -> None:
        """
        Paramaterise the input structure, using Open Force Field v.2.0 'Sage'
        for the ligand, AMBER ff14SB for the protein, and TIP3P for the water.
        The resulting system is saved to the input directory.

        Parameters
        ----------
        slurm : bool, optional, default=True
            Whether to use SLURM to run the solvation job, by default True.
        """
        if slurm:
            self._logger.info(
                "Parameterising input structures. Submitting through SLURM..."
            )
            if self.leg_type == _LegType.BOUND:
                job_name = "param_bound"
                fn = _slurm_parameterise_bound
            elif self.leg_type == _LegType.FREE:
                job_name = "param_free"
                fn = _slurm_parameterise_free
            else:
                raise ValueError("Invalid leg type.")
            self._run_slurm(fn, wait=True, run_dir=self.input_dir, job_name=job_name)

            # Check that the required input files have been produced, since slurm can fail silently
            for file in _PreparationStage.PARAMETERISED.get_simulation_input_files(
                self.leg_type
            ):
                if not _os.path.isfile(f"{self.input_dir}/{file}"):
                    raise RuntimeError(
                        f"SLURM job failed to produce {file}. Please check the output of the "
                        f"last slurm log in {self.input_dir} directory for error."
                    )
        else:
            self._logger.info("Parmeterising input structures...")
            _sysprep_parameterise_input(self.leg_type, self.input_dir, self.input_dir)

        # Update the preparation stage
        self.prep_stage = _PreparationStage.PARAMETERISED

    def solvate_input(self, slurm: bool = True) -> None:
        """
        Determine an appropriate (rhombic dodecahedron)
        box size, then solvate the input structure using
        TIP3P water, adding 150 mM NaCl to the system.
        The resulting system is saved to the input directory.

        Parameters
        ----------
        slurm : bool, optional, default=True
            Whether to use SLURM to run the job, by default True.
        """
        if slurm:
            self._logger.info("Solvating input structure. Submitting through SLURM...")
            if self.leg_type == _LegType.BOUND:
                job_name = "solvate_bound"
                fn = _slurm_solvate_bound
            elif self.leg_type == _LegType.FREE:
                job_name = "solvate_free"
                fn = _slurm_solvate_free
            else:
                raise ValueError("Invalid leg type.")
            self._run_slurm(fn, wait=True, run_dir=self.input_dir, job_name=job_name)

            # Check that the required input files have been produced, since slurm can fail silently
            for file in _PreparationStage.SOLVATED.get_simulation_input_files(
                self.leg_type
            ):
                if not _os.path.isfile(f"{self.input_dir}/{file}"):
                    raise RuntimeError(
                        f"SLURM job failed to produce {file}. Please check the output of the "
                        f"last slurm log in {self.input_dir} directory for error."
                    )
        else:
            self._logger.info("Solvating input structure...")
            _sysprep_solvate_input(self.leg_type, self.input_dir, self.input_dir)

        # Update the preparation stage
        self.prep_stage = _PreparationStage.SOLVATED

    def minimise_input(self, slurm: bool = True) -> None:
        """
        Minimise the input structure with GROMACS.

        Parameters
        ----------
        slurm : bool, optional, default=True
            Whether to use SLURM to run the job, by default True.
        """
        if slurm:
            self._logger.info("Minimising input structure. Submitting through SLURM...")
            if self.leg_type == _LegType.BOUND:
                job_name = "minimise_bound"
                fn = _slurm_minimise_bound
            elif self.leg_type == _LegType.FREE:
                job_name = "minimise_free"
                fn = _slurm_minimise_free
            else:
                raise ValueError("Invalid leg type.")
            self._run_slurm(fn, wait=True, run_dir=self.input_dir, job_name=job_name)

            # Check that the required input files have been produced, since slurm can fail silently
            for file in _PreparationStage.MINIMISED.get_simulation_input_files(
                self.leg_type
            ):
                if not _os.path.isfile(f"{self.input_dir}/{file}"):
                    raise RuntimeError(
                        f"SLURM job failed to produce {file}. Please check the output of the "
                        f"last slurm log in {self.input_dir} directory for error."
                    )
        else:
            self._logger.info("Minimising input structure...")
            _sysprep_minimise_input(self.leg_type, self.input_dir, self.input_dir)

        # Update the preparation stage
        self.prep_stage = _PreparationStage.MINIMISED

    def heat_and_preequil_input(self, slurm: bool = True) -> None:
        """
        Heat the input structure from 0 to 298.15 K with GROMACS.

        Parameters
        ----------
        slurm : bool, optional, default=True
            Whether to use SLURM to run the job, by default True.
        """
        if slurm:
            self._logger.info("Heating and equilibrating. Submitting through SLURM...")
            if self.leg_type == _LegType.BOUND:
                job_name = "heat_preequil_bound"
                fn = _slurm_heat_and_preequil_bound
            elif self.leg_type == _LegType.FREE:
                job_name = "heat_preequil_free"
                fn = _slurm_heat_and_preequil_free
            else:
                raise ValueError("Invalid leg type.")
            self._run_slurm(fn, wait=True, run_dir=self.input_dir, job_name=job_name)

            # Check that the required input files have been produced, since slurm can fail silently
            for file in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                self.leg_type
            ):
                if not _os.path.isfile(f"{self.input_dir}/{file}"):
                    raise RuntimeError(
                        f"SLURM job failed to produce {file}. Please check the output of the "
                        f"last slurm log in {self.input_dir} directory for error."
                    )
        else:
            self._logger.info("Heating and equilibrating...")
            _sysprep_heat_and_preequil_input(
                self.leg_type, self.input_dir, self.input_dir
            )

        # Update the preparation stage
        self.prep_stage = _PreparationStage.PREEQUILIBRATED

    def run_ensemble_equilibration(
        self,
        slurm: bool = True,
        append_to_ligand_selection: str = "",
        short_ensemble_equil: bool = False,
    ) -> _BSS._SireWrappers._system.System:
        """
        Run 5 ns simulations with SOMD for each of the ensemble_size runs and extract the final structures
        to use as diverse starting points for the production runs. If this is the bound leg, the restraints
        will also be extracted from the simulations and saved to a file. The simulations will be run in a
        subdirectory of the stage base directory called ensemble_equilibration, and the restraints and
        final coordinates will be saved here.

        Parameters
        ----------
        slurm : bool, optional, default=True
            Whether to use SLURM to run the job, by default True.
        append_to_ligand_selection: str, optional, default = ""
            If this is a bound leg, this appends the supplied string to the default atom
            selection which chooses the atoms in the ligand to consider as potential anchor
            points. The default atom selection is f'resname {ligand_resname} and not name H*'.
            Uses the mdanalysis atom selection language. For example, 'not name O*' will result
            in an atom selection of f'resname {ligand_resname} and not name H* and not name O*'.
        short_ensemble_equil: bool, optional, default=False
            Whether to run a short ensemble equilibration of 0.1 ns instead of the default 5 ns.
            This is not recommended for production runs, but can be useful for testing.
        """
        # Generate output dirs and copy over the input
        outdirs = [
            f"{self.base_dir}/ensemble_equilibration_{i+1}"
            for i in range(self.ensemble_size)
        ]
        for outdir in outdirs:
            _subprocess.run(["mkdir", "-p", outdir], check=True)
            for input_file in [
                f"{self.input_dir}/{ifile}"
                for ifile in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                    self.leg_type
                )
            ]:
                _subprocess.run(["cp", "-r", input_file, outdir], check=True)

        if slurm:
            if self.leg_type == _LegType.BOUND:
                job_name = "ensemble_equil_bound"
                fn = (
                    _slurm_ensemble_equilibration_bound
                    if not short_ensemble_equil
                    else _slurm_ensemble_equilibration_bound_short
                )
            elif self.leg_type == _LegType.FREE:
                job_name = "ensemble_equil_free"
                fn = (
                    _slurm_ensemble_equilibration_free
                    if not short_ensemble_equil
                    else _slurm_ensemble_equilibration_free_short
                )
            else:
                raise ValueError("Invalid leg type.")

            # For each ensemble member, run a 5 ns simulation in a seperate directory

            for i, outdir in enumerate(outdirs):
                self._logger.info(
                    f"Running ensemble equilibration for run {i+1}. Submitting through SLURM..."
                )
                self._run_slurm(fn, wait=False, run_dir=outdir, job_name=job_name)

            self.virtual_queue.wait()  # Wait for all jobs to finish

            # Check that the required input files have been produced, since slurm can fail silently
            for i, outdir in enumerate(outdirs):
                for (
                    file
                ) in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                    self.leg_type
                ):
                    if not _os.path.isfile(f"{outdir}/{file}"):
                        raise RuntimeError(
                            f"SLURM job failed to produce {file}. Please check the output of the "
                            f"last slurm log in {outdir} directory for error."
                        )

        else:  # Not slurm
            for i, outdir in enumerate(outdirs):
                self._logger.info(f"Running ensemble equilibration for run {i+1}.")
                _sysprep_run_ensemble_equilibration(
                    self.leg_type, outdir, outdir, short_ensemble_equil
                )

        # Give the output files unique names
        for i, outdir in enumerate(outdirs):
            _subprocess.run(
                ["mv", f"{outdir}/somd.rst7", f"{outdir}/somd_{i+1}.rst7"], check=True
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
                self._logger.info(f"Loading trajectory for run {i+1}...")
                top_file = f"{self.input_dir}/{_PreparationStage.PREEQUILIBRATED.get_simulation_input_files(self.leg_type)[0]}"
                traj = _BSS.Trajectory.Trajectory(
                    topology=top_file,
                    trajectory=f"{outdir}/gromacs.xtc",
                    system=pre_equilibrated_system,
                )
                self._logger.info(f"Selecting restraints for run {i+1}...")
                restraint = _BSS.FreeEnergy.RestraintSearch.analyse(
                    method="BSS",
                    system=pre_equilibrated_system,
                    traj=traj,
                    work_dir=outdir,
                    temperature=298.15 * _BSS.Units.Temperature.kelvin,
                    append_to_ligand_selection=append_to_ligand_selection,
                )

                # Check that we actually generated a restraint
                if restraint is None:
                    raise ValueError(f"No restraints found for run {i+1}.")

                # Save the restraints to a text file and store within the Leg object
                with open(f"{outdir}/restraint_{i+1}.txt", "w") as f:
                    f.write(restraint.toString(engine="SOMD"))  # type: ignore
                self.restraints.append(restraint)

            return pre_equilibrated_system

        else:  # Free leg
            return pre_equilibrated_system

    def write_input_files(
        self,
        pre_equilibrated_system: _BSS._SireWrappers._system.System,  # type: ignore
        use_same_restraints: bool = True,
    ) -> None:
        """
        Write the required input files to all of the stage input directories.

        Parameters
        ----------
        pre_equilibrated_system: _BSS._SireWrappers._system.System
            The equilibrated system to run further equilinration on. The final coordinates
            are then used as input for each of the individual runs.
        use_same_restraints: bool, default=True
            If True, the same restraints will be used for all of the bound leg repeats - by default
            , the restraints generated for the first repeat are used. This allows meaningful
            comparison between repeats for the bound leg. If False, the unique restraints are
            generated for each repeat.
        """
        # Dummy values get overwritten later
        DUMMY_RUNTIME = 0.001  # ns
        DUMMY_LAM_VALS = [0.0]
        if not hasattr(self, "stage_input_dirs"):
            raise AttributeError("No stage input directories have been set.")

        for stage_type, stage_input_dir in self.stage_input_dirs.items():
            self._logger.info(
                f"Writing input files for {self.leg_type.name} leg {stage_type.name} stage"
            )
            restraint = self.restraints[0] if self.leg_type == _LegType.BOUND else None
            protocol = _BSS.Protocol.FreeEnergy(
                runtime=DUMMY_RUNTIME * _BSS.Units.Time.nanosecond,  # type: ignore
                lam_vals=DUMMY_LAM_VALS,
                perturbation_type=stage_type.bss_perturbation_type,
            )
            self._logger.info(f"Perturbation type: {stage_type.bss_perturbation_type}")
            # Ensure we remove the velocites to avoid RST7 file writing issues, as before
            restrain_fe_calc = _BSS.FreeEnergy.Absolute(
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

            # Copy the run_somd.sh script to the stage input directory
            _shutil.copy(f"{self.input_dir}/run_somd.sh", stage_input_dir)

            # Copy the final coordinates from the ensemble equilibration stage to the stage input directory
            # and, if this is the bound stage, also copy over the restraints
            for i in range(self.ensemble_size):
                ens_equil_output_dir = f"{self.base_dir}/ensemble_equilibration_{i+1}"
                coordinates_file = f"{ens_equil_output_dir}/somd_{i+1}.rst7"
                _shutil.copy(coordinates_file, f"{stage_input_dir}/somd_{i+1}.rst7")
                if self.leg_type == _LegType.BOUND:
                    if (
                        use_same_restraints
                    ):  # Want to use same restraints for all repeats
                        restraint_file = (
                            f"{self.base_dir}/ensemble_equilibration_1/restraint_1.txt"
                        )
                    else:
                        restraint_file = f"{ens_equil_output_dir}/restraint_{i+1}.txt"
                    _shutil.copy(
                        restraint_file, f"{stage_input_dir}/restraint_{i+1}.txt"
                    )

            # Update the template-config.cfg file with the perturbed residue number generated
            # by BSS, as well as the restraints options
            _shutil.copy(f"{self.input_dir}/template_config.cfg", stage_input_dir)

            # Read simfile options
            perturbed_resnum = _read_simfile_option(
                f"{stage_input_dir}/somd.cfg", "perturbed residue number"
            )
            # Temporary fix for BSS bug - perturbed residue number is wrong, but since we always add the
            # ligand first to the system, this should always be 1 anyway
            # TODO: Fix this - raise BSS issue
            perturbed_resnum = "1"
            try:
                use_boresch_restraints = _read_simfile_option(
                    f"{stage_input_dir}/somd.cfg", "use boresch restraints"
                )
            except ValueError:
                use_boresch_restraints = False
            try:
                turn_on_receptor_ligand_restraints_mode = _read_simfile_option(
                    f"{stage_input_dir}/somd.cfg",
                    "turn on receptor-ligand restraints mode",
                )
            except ValueError:
                turn_on_receptor_ligand_restraints_mode = False

            # Now write simfile options
            _write_simfile_option(
                f"{stage_input_dir}/template_config.cfg",
                "perturbed residue number",
                perturbed_resnum,
            )
            _write_simfile_option(
                f"{stage_input_dir}/template_config.cfg",
                "use boresch restraints",
                str(use_boresch_restraints),
            )
            _write_simfile_option(
                f"{stage_input_dir}/template_config.cfg",
                "turn on receptor-ligand restraints mode",
                str(turn_on_receptor_ligand_restraints_mode),
            )

            # Now overwrite the SOMD generated config file with the updated template
            _subprocess.run(
                [
                    "mv",
                    f"{stage_input_dir}/template_config.cfg",
                    f"{stage_input_dir}/somd.cfg",
                ],
                check=True,
            )

            # Set the default lambda windows based on the leg and stage types
            lam_vals = Leg.default_lambda_values[self.leg_type][stage_type]
            lam_vals_str = ", ".join([str(lam_val) for lam_val in lam_vals])
            _write_simfile_option(
                f"{stage_input_dir}/somd.cfg", "lambda array", lam_vals_str
            )

    def _run_slurm(
        self, sys_prep_fn: _Callable, wait: bool, run_dir: str, job_name: str
    ) -> None:
        """
        Run the supplied function through a slurm job. The function must be in
        the _system_prep module.

        Parameters
        ----------
        sys_prep_fn: Callable
            The function to run through slurm.
        wait: bool
            If True, the function will wait for the job to complete before returning.
        run_dir: str
            The directory to run the job in.
        job_name: str
            The name of the job.

        Returns
        -------
        None
        """
        # Write the slurm script
        # Get the header from run_somd.sh
        header_lines = []
        with open(f"{self.input_dir}/run_somd.sh", "r") as file:
            for line in file.readlines():
                if line.startswith("#SBATCH") or line.startswith("#!/bin/bash"):
                    header_lines.append(line)
                else:
                    break

        # Add lines to run the python function and write out
        header_lines.append(
            f"\npython -c 'from EnsEquil.run.system_prep import {sys_prep_fn.__name__}; {sys_prep_fn.__name__}()'"
        )
        slurm_file = f"{run_dir}/{job_name}.sh"
        with open(slurm_file, "w") as file:
            file.writelines(header_lines)

        # Submit to the virtual queue
        cmd = f"--chdir={run_dir} {run_dir}/{job_name}.sh"  # The virtual queue adds sbatch
        slurm_file_base = _get_slurm_file_base(slurm_file)
        job = self.virtual_queue.submit(cmd, slurm_file_base=slurm_file_base)
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
        parallel: bool = True,
    ) -> None:
        """
        Run all stages and perform analysis once finished.

        Parameters
        ----------
        run_nos : Optional[List[int]], default=None
            If specified, only run the specified runs. Otherwise, run all runs.
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
        run_nos = self._get_valid_run_nos(run_nos)

        self._logger.info(
            f"Running run numbers {run_nos} for {self.__class__.__name__}..."
        )
        for stage in self.stages:
            stage.run(run_nos=run_nos, adaptive=adaptive, runtime=runtime)
            if not parallel:
                stage.wait()

    def analyse(
        self, run_nos: _Optional[_List[int]], subsampling=False
    ) -> _Tuple[_np.ndarray, _np.ndarray]:
        f"""
        Analyse the leg and any sub-simulations, and
        return the overall free energy change.

        Parameters
        ----------
        subsampling: bool, optional, default=False
            If True, the free energy will be calculated by subsampling using
            the methods contained within pymbar.

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
            run_nos=run_nos, subsampling=subsampling
        )

        if self.leg_type == _LegType.BOUND:
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

            # Update the internally stored dGs after the restraint corrections
            self._delta_g = dg_overall

        return dg_overall, er_overall

    def analyse_convergence(
        self, run_nos: _Optional[_List[int]] = None
    ) -> _Tuple[_np.ndarray, _np.ndarray]:
        f"""
        Get a timeseries of the total free energy change of the
        {self.__class__.__name__} against total simulation time. Also plot this.
        Keep this separate from analyse as it is expensive to run.

        Parameters
        ----------
        run_nos : Optional[List[int]], default=None
            If specified, only analyse the specified runs. Otherwise, analyse all runs.

        Returns
        -------
        fracts : np.ndarray
            The fraction of the total equilibrated simulation time for each value of dg_overall.
        dg_overall : np.ndarray
            The overall free energy change for the {self.__class__.__name__} for
            each value of total equilibrated simtime for each of the ensemble size repeats.
        """
        run_nos = self._get_valid_run_nos(run_nos)

        self._logger.info(f"Analysing convergence of {self.__class__.__name__}...")

        # Get the dg_overall in terms of fraction of the total simulation time
        # Use steps of 5 % of the total simulation time
        fracts = _np.arange(0.05, 1.05, 0.05)
        # Create an array to store the overall free energy change
        dg_overall = _np.zeros((len(run_nos), len(fracts)))

        # Now add up the data for each of the sub-simulation runners
        for sub_sim_runner in self._sub_sim_runners:
            _, dgs = sub_sim_runner.analyse_convergence(run_nos=run_nos)
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
        self._logger.info(f"Fractions of equilibrated simulation time: {fracts}")

        # Plot the overall convergence
        _plot_convergence(
            fracts,
            dg_overall,
            self.get_tot_simtime(run_nos=run_nos),
            self.equil_time,
            self.output_dir,
            len(run_nos),
        )

        return fracts, dg_overall

    def lighten(self) -> None:
        f"""Lighten the leg by deleting ensemble equilibration output
        and lightening all sub-simulation runners"""
        # Remove the ensemble equilibration directories
        for direct in _pathlib.Path(self.base_dir).glob("ensemble_equilibration*"):
            print("DIRECTORY TO REMOVE", direct)
            self._logger.info(f"Deleting {direct}")
            _subprocess.run(["rm", "-rf", direct])

        # Lighten all the sub-simulation runners
        if hasattr(self, "_sub_sim_runners"):
            for sub_sim_runner in self._sub_sim_runners:
                sub_sim_runner.lighten()
