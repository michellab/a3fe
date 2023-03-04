"""Functionality for managing legs of the calculation."""

import glob as _glob
from inspect import Parameter
import BioSimSpace.Sandpit.Exscientia as _BSS
from enum import Enum as _Enum
import logging as _logging
from multiprocessing import Pool as _Pool
import os as _os
import pathlib as _pathlib
import pickle as _pkl
import shutil as _shutil
import subprocess as _subprocess
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

from .stage import Stage as _Stage, StageType as _StageType

class LegType(_Enum):
    """The type of leg in the calculation."""
    BOUND = 1
    FREE = 2

class PreparationStage(_Enum):
    """The stage of preparation of the input files."""
    STRUCTURES_ONLY = 1
    PARAMETERISED = 2
    SOLVATED = 3
    MINIMISED = 4
    PREEQUILIBRATED = 5

    @property
    def file_suffix(self) -> str:
        """Return the suffix to use for the files in this stage."""
        if self == PreparationStage.STRUCTURES_ONLY:
            return ""
        elif self == PreparationStage.PARAMETERISED:
            return "_param"
        elif self == PreparationStage.SOLVATED:
            return "_solv"
        elif self == PreparationStage.MINIMISED:
            return "_min"
        elif self == PreparationStage.PREEQUILIBRATED:
            return "_preequil"
        else:
            raise ValueError(f"Unknown preparation stage: {self}")
        
    def get_simulation_input_files(self, leg_type: LegType) -> _List[str]:
        """Return the input files required for the simulation in this stage."""
        if self == PreparationStage.STRUCTURES_ONLY:
            if leg_type == LegType.BOUND:
                return ["protein.pdb", "ligand.pdb"]
            elif leg_type == LegType.FREE:
                return ["ligand.pdb"]
        else:
            return [f"{leg_type.name.lower()}{self.file_suffix}.{file_type}" for file_type in ["prm7", "rst7"]]

class Leg():
    """
    Class set up and run the stages of a leg of the calculation.
    """
    # Required input files for each leg type and preparation stage.
    required_input_files = {}
    for leg_type in LegType:
        required_input_files[leg_type] = {}
        for prep_stage in PreparationStage:
            required_input_files[leg_type][prep_stage] = ["run_somd.sh"] + prep_stage.get_simulation_input_files(leg_type)

    required_stages = {LegType.BOUND: [_StageType.RESTRAIN, _StageType.DISCHARGE, _StageType.VANISH],
                        LegType.FREE: [_StageType.DISCHARGE, _StageType.VANISH]}

    def __init__(self, 
                 leg_type: LegType,
                 block_size: float = 1,
                 equil_detection: str = "block_gradient",
                 gradient_threshold: _Optional[float] = None,
                 ensemble_size: int = 5,
                 input_dir: _Optional[str] = None,
                 base_dir: _Optional[str] = None,
                 stream_log_level: int = _logging.INFO) -> None:
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
        ensemble_size : int, Optional, default: 5
            Number of simulations to run in the ensemble.
        input_dir : str, Optional, default: None
            Path to directory containing input files for the simulations. If None, this
            is set to `current_working_directory/input`.
        base_dir : str, Optional, default: None
            Path to the base directory in which to set up the stages. If None,
            this is set to the current working directory.
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            calculation object and its child objects.

        Returns
        -------
        None
        """
        # Check if we are starting from a previous simulation
        if _os.path.isfile(f"{base_dir}/leg.pkl"):
            print("Loading previous leg. Any arguments will be overwritten...")
            with open(f"{base_dir}/leg.pkl", "rb") as file:
                self.__dict__ = _pkl.load(file)

        else:  # No pkl file to resume from
            print("Creating new leg...")

            # Set up the calculation
            self.leg_type = leg_type
            self.stage_types = Leg.required_stages[leg_type]
            self.block_size = block_size
            self.equil_detection = equil_detection
            self.gradient_threshold = gradient_threshold
            self.ensemble_size = ensemble_size
            if input_dir is None:
                input_dir = _os.path.join(_os.getcwd(), "input")
            self.input_dir = input_dir
            if base_dir is None:
                base_dir = _os.path.join(_os.getcwd(), f"{leg_type.name.lower()}")
            _pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
            self.base_dir = base_dir
            self._running: bool = False
            self.running_stages: _List[_Stage] = []
            if base_dir is None:
                self.base_dir = f"./{leg_type.name.lower()}"
            else:
                self.base_dir = base_dir

            # Set up logging
            self.stream_log_level = stream_log_level
            self._logger = _logging.getLogger(str(self))
            # For the file handler, we want to log everything
            file_handler = _logging.FileHandler(f"{self.base_dir}/{self.leg_type.name.lower()}.log")
            file_handler.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            file_handler.setLevel(_logging.DEBUG)
            self._logger.addHandler(file_handler)
            # For the stream handler, we want to log at the user-specified level
            stream_handler = _logging.StreamHandler()
            stream_handler.setFormatter(_logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
            stream_handler.setLevel(stream_log_level)
            self._logger.addHandler(stream_handler)

            # Validate the input
            self._validate_input()

            # Save state
            self._dump()

    def __str__(self) -> str:
        return f"Leg (type = {self.leg_type.name}, repeats = {self.ensemble_size}))"

    def _validate_input(self) -> None:
        """Check that the required files are provided for the leg type and set the preparation stage
        according to the files present."""
        # Check backwards, as we care about the most advanced preparation stage
        for prep_stage in reversed(PreparationStage):
            files_absent = False
            for file in Leg.required_input_files[self.leg_type][prep_stage]:
                if not _os.path.isfile(f"{self.input_dir}/{file}"):
                    files_absent = True
            # We have the required files for this prep stage, and this is the most 
            # advanced prep stage that files are present for
            if not files_absent:
                self._logger.info(f"Found all required input files for preparation stage {prep_stage.name.lower()}")
                self.prep_stage = prep_stage
                return 
        # We didn't find all required files for any of the prep stages
        raise ValueError(f"Could not find all required input files for leg type {self.leg_type.name} for " \
                          f"any preparation stage. Required files are: {Leg.required_input_files[self.leg_type]}")


    def setup(self) -> None:
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
        """
        self._logger.info("Setting up leg...")
        # Create input directories, parameterise, solvate, minimise, heat and preequil, all
        # depending on the input files present.
        # First, create the input directories
        self.create_stage_input_dirs()
        # Then load in the input files
        if self.prep_stage == PreparationStage.STRUCTURES_ONLY:
            system = self.parameterise_input()
        else:
            system = _BSS.IO.readMolecules([f"{self.input_dir}/{file}" for file in self.prep_stage.get_simulation_input_files(self.leg_type)])
        # Now, process the input files depending on the preparation stage
        if self.prep_stage == PreparationStage.PARAMETERISED:
            system = self.solvate_input(system) # This also adds ions
        if self.prep_stage == PreparationStage.SOLVATED:
            system = self.minimise_input(system)
        if self.prep_stage == PreparationStage.MINIMISED:
            system = self.heat_and_preequil_input(system)

        # If this is the bound leg, extract the restraints from 5 ns simulations
        if self.leg_type == LegType.BOUND:
            self.extract_restraints(system)

        # Write input files
        self.write_input_files(system)

        # Create the Stage objects, which automatically set themselves up
        self.stages = []
        for stage_type in self.required_stages[self.leg_type]:
            self.stages.append(_Stage(stage_type=stage_type,
                                      block_size=self.block_size,
                                      equil_detection=self.equil_detection,
                                      gradient_threshold=self.gradient_threshold,
                                      ensemble_size=self.ensemble_size,
                                      input_dir=self.stage_input_dirs[stage_type],
                                      output_dir=self.stage_input_dirs[stage_type].replace("input", "output"),
                                      stream_log_level=self.stream_log_level))

        self._logger.info("Setup complete.")
        # Save state
        self._dump()


    def get_optimal_lambda_windows(self, simtime:float = 0.1) -> None:
        """
        Determine the optimal lambda windows for each stage of the leg
        by running short simulations at each lambda value and analysing them.

        Parameters
        ----------
        simtime : float, Optional, default: 0.1
            The length of the short simulations to run, in ns.
        
        Returns
        -------
        None
        """
        # Check that the leg has been set up
        if not hasattr(self, "legs"):
            raise ValueError("The calculation has not been set up yet. Please call setup() first.")

        # Run in parallel
        def get_optimal_lambda_windows_stage(stage: _Stage, simtime:float) -> None:
            self._logger.info(f"Running simulations to determine optimal lambda windows for {stage}...")
            # Run the stage for a given amount of time and wait for it to finish
            stage.run(adaptive=False, runtime=simtime)
            stage.wait()
            optimal_lam_vals = stage.get_optimal_lambda_values()
            # Save the old data, then create new LamWindow objects with the optimal lambda values
            stage.mv_output(name="lam_val_determination")
            stage.lam_vals = list(optimal_lam_vals) # This deletes all of the old LamWindow objects and creates a new output dir
            stage.update() # This deletes all of the old LamWindow objects and creates a new output dir

        with _Pool() as pool:
            pool.starmap(get_optimal_lambda_windows_stage, [(stage, simtime) for stage in self.stages])

        # Run the simulations
        for stage in self.stages:
            self._logger.info(f"Running simulations to determine optimal lambda windows for {stage}...")
            # Run the stage for a given amount of time and wait for it to finish
            stage.run(adaptive=False, simtime=simtime)
            stage.wait()
            optimal_lam_vals = stage.get_optimal_lam_vals()
            # Save the old data, then create new LamWindow objects with the optimal lambda values
            stage.mv_output(name="lam_val_determination")
            stage.lam_vals = optimal_lam_vals # This deletes all of the old LamWindow objects and creates a new output dir
            stage.update() # This deletes all of the old LamWindow objects and creates a new output dir

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

    def parameterise_input(self) -> _BSS._SireWrappers._system.System:
        """
        Paramaterise the input structure, using Open Force Field v.2.0 'Sage'
        for the ligand, AMBER ff14SB for the protein, and TIP3P for the water.
        The resulting system is saved to the input directory.
        
        Returns
        -------
        parameterised_system : _BSS._SireWrappers._system.System
            Parameterised system.
        """
        FORCEFIELDS = {"ligand": "openff_unconstrained-2.0.0", 
                       "protein": "ff14SB", 
                       "water": "tip3p"}

        self._logger.info("Parameterising input...")
        # Parameterise the ligand
        self._logger.info("Parameterising ligand...")
        lig = _BSS.IO.readMolecules(f"{self.input_dir}/ligand.pdb")[0]
        param_lig = _BSS.Parameters.parameterise(molecule=lig, forcefield=FORCEFIELDS["ligand"]).getMolecule()


        # If bound, then parameterise the protein and waters and add to the system
        if self.leg_type == LegType.BOUND:
            # Parameterise the protein
            self._logger.info("Parameterising protein...")
            protein = _BSS.IO.readMolecules(f"{self.input_dir}/protein.pdb")[0]
            param_protein = _BSS.Parameters.parameterise(molecule=protein, 
                                                         forcefield=FORCEFIELDS["protein"]).getMolecule()

            # Parameterise the waters
            self._logger.info("Parameterising crystallographic waters...")
            waters = _BSS.IO.readMolecules(f"{self.input_dir}/waters.pdb")
            param_waters = []
            for water in waters:
                param_waters.append(_BSS.Parameters.parameterise(molecule=water, 
                                                                 water_model=FORCEFIELDS["water"],
                                                                 forcefield=FORCEFIELDS["protein"]).getMolecule())

            # Create the system
            self._logger.info("Assembling parameterised system...")
            parameterised_system = param_lig + param_protein
            for water in param_waters:
                parameterised_system += water

        # This is the free leg, so just turn the ligand into a system
        else:
            parameterised_system = param_lig.toSystem()

        # Set the parameterisation stage
        self.prep_stage = PreparationStage.PARAMETERISED
        # Save the system
        self._logger.info("Saving parameterised system...")
        _BSS.IO.saveMolecules(f"{self.base_dir}/input/{self.leg_type.name.lower()}{self.prep_stage.file_suffix}",
                               parameterised_system, 
                               fileformat=["prm7", "rst7"])

        return parameterised_system

    def solvate_input(self, parameterised_system: _BSS._SireWrappers._system.System) -> _BSS._SireWrappers._system.System:
        """
        Determine an appropriate (rhombic dodecahedron) 
        box size, then solvate the input structure using
        TIP3P water, adding 150 mM NaCl to the system. 
        The resulting system is saved to the input directory.
        
        Parameters
        ----------
        parameterised_system : _BSS._SireWrappers._system.System
            Parameterised system.
        
        Returns
        -------
        solvated_system : _BSS._SireWrappers._system.System
            Solvated system.
        """
        WATER_MODEL = "tip3p"
        ION_CONC = 0.15 # M

        # Determine the box size
        # Taken from https://github.com/michellab/BioSimSpaceTutorials/blob/main/01_introduction/02_molecular_setup.ipynb
        # Get the minimium and maximum coordinates of the bounding box that
        # minimally encloses the protein.
        self._logger.info("Determining optimal rhombic dodecahedral box...")
        box_min, box_max = parameterised_system.getAxisAlignedBoundingBox()

        # Work out the box size from the difference in the coordinates.
        box_size = [y - x for x, y in zip(box_min, box_max)]

        # Add 15 A padding to the box size in each dimension.
        padding = 15 * _BSS.Units.Length.angstrom

        # Work out an appropriate box. This will used in each dimension to ensure
        # that the cutoff constraints are satisfied if the molecule rotates.
        box_length = max(box_size) + 2*padding
        box, angles = _BSS.Box.rhombicDodecahedronHexagon(box_length)

        self._logger.info(f"Solvating system with {WATER_MODEL} water and {ION_CONC} M NaCl...")
        solvated_system = _BSS.Solvent.solvate(model=WATER_MODEL,
                                               molecule=parameterised_system,
                                               box=box, 
                                               angles=angles, 
                                               ion_conc=ION_CONC)

        # Set the preparation stage
        self.prep_stage = PreparationStage.SOLVATED
        # Save the system
        self._logger.info("Saving solveted system")
        _BSS.IO.saveMolecules(f"{self.base_dir}/input/{self.leg_type.name.lower()}{self.prep_stage.file_suffix}",
                               solvated_system, 
                               fileformat=["prm7", "rst7"])

        return solvated_system

    def minimise_input(self, solvated_system: _BSS._SireWrappers._system.System) -> _BSS._SireWrappers._system.System:
        """
        Minimise the input structure with GROMACS. The resulting system is saved to the input directory.
        
        Parameters
        ----------
        solvated_system : _BSS._SireWrappers._system.System
            Solvated system.
        
        Returns
        -------
        minimised_system : _BSS._SireWrappers._system.System
            Minimised system.
        """
        STEPS = 1000 # This is the default for _BSS
        self._logger.info(f"Minimising input structure with {STEPS} steps...")
        protocol = _BSS.Protocol.Minimisation(steps=STEPS)
        minimised_system = self._run_process(solvated_system, protocol, prep_stage=PreparationStage.MINIMISED)
        return minimised_system

    def heat_and_preequil_input(self, minimised_system: _BSS._SireWrappers._system.System) -> _BSS._SireWrappers._system.System:
        """ 
        Heat the input structure from 0 to 298.15 K with GROMACS. The resulting system is saved to the input directory.
        
        Parameters
        ----------
        minimised_system : _BSS._SireWrappers._system.System
            Minimised system.
        
        Returns
        -------
        preequilibrated_system : _BSS._SireWrappers._system.System
            Pre-Equilibrated system.
        """
        RUNTIME_SHORT_NVT = 5 # ps
        RUNTIME_NVT = 50 # ps 
        RUNTIME_NPT = 200 # ps
        END_TEMP = 298.15 # K

        self._logger.info(f"PMEMD NVT equilibration for {RUNTIME_SHORT_NVT} ps while restraining all non-solvent atoms")
        protocol = _BSS.Protocol.Equilibration(
                                        runtime=RUNTIME_SHORT_NVT*_BSS.Units.Time.picosecond, 
                                        temperature_start=0*_BSS.Units.Temperature.kelvin, 
                                        temperature_end=END_TEMP*_BSS.Units.Temperature.kelvin,
                                        restraint="all"
                                        )
        equil1 = self._run_process(minimised_system, protocol)

        # If this is the bound leg, carry out step with backbone restraints
        if self.leg_type == LegType.BOUND:
            self._logger.info(f"PMEMD NVT equilibration for {RUNTIME_NVT} ps while restraining all backbone atoms")
            protocol = _BSS.Protocol.Equilibration(
                                            runtime=RUNTIME_NVT*_BSS.Units.Time.picosecond, 
                                            temperature=END_TEMP*_BSS.Units.Temperature.kelvin, 
                                            restraint="backbone"
                                            )
            equil2 = self._run_process(equil1, protocol)

        else: # Free leg - skip the backbone restraint step
            equil2 = equil1

        self._logger.info(f"PMEMD NVT equilibration for {RUNTIME_NVT} ps without restraints")
        protocol = _BSS.Protocol.Equilibration(
                                        runtime=RUNTIME_NVT*_BSS.Units.Time.picosecond, 
                                        temperature=END_TEMP*_BSS.Units.Temperature.kelvin,
                                        )
        equil3 = self._run_process(equil2, protocol)

        self._logger.info(f"PMEMD NPT equilibration for {RUNTIME_NPT} ps while restraining non-solvent heavy atoms")
        protocol = _BSS.Protocol.Equilibration(
                                        runtime=RUNTIME_NPT*_BSS.Units.Time.picosecond, 
                                        pressure=1*_BSS.Units.Pressure.atm,
                                        temperature=END_TEMP*_BSS.Units.Temperature.kelvin,
                                        restraint="heavy",
                                        )
        equil4 = self._run_process(equil3, protocol)

        self._logger.info(f"PMEMD NPT equilibration for {RUNTIME_NPT} ps without restraints")
        protocol = _BSS.Protocol.Equilibration(
                                        runtime=RUNTIME_NPT*_BSS.Units.Time.picosecond, 
                                        pressure=1*_BSS.Units.Pressure.atm,
                                        temperature=END_TEMP*_BSS.Units.Temperature.kelvin,
                                        )
        preequilibrated_system = self._run_process(equil4, protocol, prep_stage=PreparationStage.PREEQUILIBRATED)

        return preequilibrated_system

    def _run_process(self, system: _BSS._SireWrappers._system.System,
                     protocol: _BSS.Protocol._protocol.Protocol,
                     prep_stage: _Optional[PreparationStage] = None) -> _BSS._SireWrappers._system.System:
        """
        Run a process with GROMACS.
        
        Parameters
        ----------
        system : _BSS._SireWrappers._system.System
            System to run the process on.
        protocol : _BSS._Protocol._protocol.Protocol
            Protocol to run the process with.
        prep_stage : _Optional[PreparationStage]
            Preparation stage that the leg will be in if the process 
            completes successfully. If this is supplied, the leg's
            preparation stage will be updated and the files saved
            upon completion of the process.
        
        Returns
        -------
        system : _BSS._SireWrappers._system.System
            System after the process has been run.
        """
        process = _BSS.Process.Gromacs(system, protocol)
        process.start()
        process.wait()
        if process.isError():
            self._logger.error(process.stdout())
            self._logger.error(process.stderr())
            raise _BSS._Exceptions.ThirdPartyError("The process failed.")
        system = process.getSystem()
        # Save the system if a suffix is supplied
        if prep_stage is not None:
            # Update the leg's preparation stage
            self.prep_stage = prep_stage
            # Save the files
            file_name = f"{self.leg_type.name.lower()}_{prep_stage.file_suffix}"
            self._logger.info(f"Saving {file_name} PRM7 and RST7 files to {self.base_dir}/input")
            _BSS.IO.saveMolecules(f"{self.base_dir}/input/{file_name}",
                                system, fileformat=["prm7", "rst7"])
        return system

    def extract_restraints(self, pre_equilibrated_system: _BSS._SireWrappers._system.System) -> None:
        """
        Run 5 ns simulations with SOMD for each of the ensemble_size runs and extract the restraints.
        The simulations will be run in a subdirectory of the stage base directory called restraint_search,
        and the restraints and final coordinates will be saved here.
        
        Parameters
        ----------
        pre_equilibrated_system : _BSS._SireWrappers._system.System
            Pre-equilibrated system.
        
        Returns
        -------
        None
        """
        # Temporary hack
        old_system = pre_equilibrated_system.copy()

        RESTRAINT_SEARCH_TIME = 0.01 # ns
        # Mark the ligand to be decoupled
        lig = _BSS.Align.decouple(pre_equilibrated_system[0], intramol=True)
        self._logger.info(f"Selecting ligand {lig} for decoupling in restraint search simulations")
        pre_equilibrated_system.updateMolecule(0,lig)
        protocol = _BSS.Protocol.Production(timestep=2*_BSS.Units.Time.femtosecond, # 2 fs timestep as 4 fs seems to cause instability even with HMR
                                             runtime=RESTRAINT_SEARCH_TIME*_BSS.Units.Time.nanosecond)

        self._logger.info(f"Running {self.ensemble_size} SOMD simulations for {RESTRAINT_SEARCH_TIME} ns to extract restraints")
        # Repeat this for each of the ensemble_size repeats
        for i in range(self.ensemble_size):
            self._logger.info(f"Running SOMD restraint search simulation {i+1} of {self.ensemble_size}")
            restraint_search = _BSS.FreeEnergy.RestraintSearch(pre_equilibrated_system, protocol=protocol,
                                                            engine='SOMD', work_dir=f"{self.base_dir}/restraint_search",)
            restraint_search.start()
            # After waiting for the restraint search to finish, extract the final system with new coordinates, and the restraints
            restraint_search.wait()
            final_system = restraint_search._process.getSystem()
            restraint = restraint_search.analyse(method='BSS', block=True)

            # Save the final coordinates 
            # TODO: Fix this
            # Temporarily ignore issue with extracting the final system
            self._logger.info(f"Saving somd_{i+1}.rst7 and restraint_{i+1}.txt to {self.base_dir}/restraint_search")
            #_BSS.IO.saveMolecules(f"{self.base_dir}/restraint_search/somd_{i+1}", final_system, fileformat=["RST7"])
            _BSS.IO.saveMolecules(f"{self.base_dir}/restraint_search/somd_{i+1}", old_system, fileformat=["rst7"])

            # Save the restraints to a text file and store within the Leg object
            with open(f"{self.base_dir}/restraint_search/restraint_{i+1}.txt", "w") as f:
                f.write(restraint.toString(engine="SOMD"))
            if not hasattr(self, "restraints"):
                self.restraints = [restraint]
            else:
                self.restraints.append(restraint)

    def write_input_files(self, pre_equilibrated_system: _BSS._SireWrappers._system.System) -> None:
        """
        Write the required input files to all of the stage input directories.
        """
        # Dummy values get overwritten later
        DUMMY_RUNTIME = 0.001 # ns
        DUMMY_LAM_VALS = [0.0]
        if not hasattr(self, "stage_input_dirs"):
            raise AttributeError("No stage input directories have been set.")

        for stage_type, stage_input_dir in self.stage_input_dirs.items():
            self._logger.info(f"Writing input files for {self.leg_type.name} leg {stage_type.name} stage")
            self._logger.info(f"Writing input files for {self.leg_type.name} leg {stage_type.name} stage")
            restraint = self.restraints[0] if self.leg_type == LegType.BOUND else None
            protocol = _BSS.Protocol.FreeEnergy(runtime=DUMMY_RUNTIME*_BSS.Units.Time.nanosecond, 
                                                lam_vals=DUMMY_LAM_VALS, 
                                                perturbation_type=stage_type.bss_perturbation_type)
            restrain_fe_calc = _BSS.FreeEnergy.Absolute(pre_equilibrated_system, 
                                                        protocol,
                                                        engine='SOMD', 
                                                        restraint=restraint,
                                                        work_dir=stage_input_dir,
                                                        setup_only=True) # We will run outside of BSS

            # Copy input written by BSS to the stage input directory
            for file in _glob.glob(f"{stage_input_dir}/lambda_0.0000/*"):
                _shutil.copy(file, stage_input_dir)
            _subprocess.run(["rm", "-rf", f"{stage_input_dir}/lambda_*"], check=True)

            # Copy the run_somd.sh script to the stage input directory
            _shutil.copy(f"{self.input_dir}/run_somd.sh", stage_input_dir)

            # If this is the bound stage, copy the restraints and the final coordinates
            # after the restraint search to the stage input directory
            if self.leg_type == LegType.BOUND:
                for i in range(self.ensemble_size):
                    restraint_file = f"{self.base_dir}/restraint_search/restraint_{i+1}.txt"
                    _shutil.copy(restraint_file, f"{stage_input_dir}/restraint_{i+1}.txt")
                    coordinates_file = f"{self.base_dir}/restraint_search/somd_{i+1}.rst7"
                    _shutil.copy(coordinates_file, f"{stage_input_dir}/somd_{i+1}.rst7")


    def run(self) -> None:
        """Run all stages and perform analysis once finished."""
        self._logger.info("Running all stages")

        def run_stage(stage: _Stage) -> None:
            self._logger.info(f"Starting {stage}...")
            stage.run()

        # Run in parallel
        with _Pool() as pool:
            pool.map(run_stage, self.stages)

        self.analyse()

    def analyse(self) -> None:
        pass

    def _update_log(self) -> None:
        """ Update the status log file with the current status of the ensemble. """
        self._logger.debug("##############################################")
        for var in vars(self):
            self._logger.debug(f"{var}: {getattr(self, var)}")
        self._logger.debug("##############################################")

    def _dump(self) -> None:
        """ Dump the current state of the ensemble to a pickle file."""
        with open(f"{self.base_dir}/leg.pkl", "wb") as ofile:
            _pkl.dump(self.__dict__, ofile)
