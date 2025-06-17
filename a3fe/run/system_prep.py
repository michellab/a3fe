"""Functionality for running preparation simulations."""

__all__ = [
    "SystemPreparationConfig",
    "parameterise_input",
    "solvate_input",
    "minimise_input",
    "heat_and_preequil_input",
    "run_ensemble_equilibration",
]

import pathlib as _pathlib
import pickle as _pkl
import warnings as _warnings
from typing import Optional as _Optional

import BioSimSpace.Sandpit.Exscientia as _BSS
from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field

from ..read._process_bss_systems import rename_lig as _rename_lig
from ._utils import check_has_wat_and_box as _check_has_wat_and_box
from .enums import LegType as _LegType
from .enums import PreparationStage as _PreparationStage
from .enums import StageType as _StageType


class SystemPreparationConfig(_BaseModel):
    """
    Pydantic model for holding system preparation configuration.

    Attributes
    ----------
    slurm: bool
        Whether to use SLURM for the preparation.
    forcefields : dict
        Forcefields to use for the ligand, protein, and water.
    water_model : str
        Water model to use.
    ion_conc : float
        Ion concentration in M.
    steps : int
        Number of steps for the minimisation.
    runtime_short_nvt : int
        Runtime for the short NVT equilibration in ps.
    runtime_nvt : int
        Runtime for the NVT equilibration in ps.
    end_temp : float
        End temperature for the NVT equilibration in K.
    runtime_npt : int
        Runtime for the NPT equilibration in ps.
    runtime_npt_unrestrained : int
        Runtime for the unrestrained NPT equilibration in ps.
    ensemble_equilibration_time : int
        Ensemble equilibration time in ps.
    append_to_ligand_selection: str
        If this is a bound leg, this appends the supplied string to the default atom
        selection which chooses the atoms in the ligand to consider as potential anchor
        points. The default atom selection is f'resname {ligand_resname} and not name H*'.
        Uses the mdanalysis atom selection language. For example, 'not name O*' will result
        in an atom selection of f'resname {ligand_resname} and not name H* and not name O*'.
    use_same_restraints: bool
        If True, the same restraints will be used for all of the bound leg repeats - by default
        , the restraints generated for the first repeat are used. This allows meaningful
        comparison between repeats for the bound leg. If False, the unique restraints are
        generated for each repeat.
    membrane_protein: bool
        If True, the protein is a membrane protein requiring special NPT settings.
    """

    slurm: bool = _Field(True)
    forcefields: dict = {
        "ligand": "openff_unconstrained-2.0.0",
        "protein": "ff14SB",
        "water": "tip3p",
    }
    water_model: str = "tip3p"
    ion_conc: float = _Field(0.15, ge=0, lt=1)  # M
    steps: int = _Field(1000, gt=0, lt=100_000)  # This is the default for _BSS
    runtime_short_nvt: int = _Field(5, gt=0, lt=500)  # ps
    runtime_nvt: int = _Field(50, gt=0, lt=5_000)  # ps
    end_temp: float = _Field(298.15, gt=0, lt=350)  # K
    runtime_npt: int = _Field(400, gt=0, lt=40_000)  # ps
    runtime_npt_unrestrained: int = _Field(1000, gt=0, lt=100_000)  # ps
    ensemble_equilibration_time: int = _Field(5000, gt=0, lt=50_000)  # ps
    append_to_ligand_selection: str = _Field(
        "",
        description="Atom selection to append to the ligand selection during restraint searching.",
    )
    use_same_restraints: bool = _Field(
        True,
        description="Whether to use the same restraints for all repeats of the bound leg. Note "
        "that this should be used if you plan to run adaptively.",
    )
    lambda_values: dict = {
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
    membrane_protein: bool = _Field(
        False,
        description="Whether the protein is a membrane protein requiring special configuration settings.",
    )

    class Config:
        """
        Pydantic model configuration.
        """

        extra = "forbid"
        validate_assignment = True

    def get_membrane_nvt_config(self) -> dict:
        """Get the temperature coupling configuration for membraneNVT equilibration runs."""
        if self.membrane_protein:
            return {
                "tcoupl": "v-rescale",
                "tc-grps": "Protein non-Protein",
                "tau-t": "1.0 1.0",
                "ref-t": "303.15 303.15",
                "annealing": "no no",
            }
        return {}

    def get_membrane_npt_config(self) -> dict:
        """Get the pressure coupling configuration for membrane equilibration runs."""
        if self.membrane_protein:
            return {
                "pcoupltype": "semiisotropic",
                "tau-p": "5.0",
                "compressibility": "4.5e-5 4.5e-5",
                "ref-p": "1.0 1.0",
                "rcoulomb": "1.2",
                "rvdw": "1.2",
                "rlist": "1.2",
                "rvdw-switch": "1.0",
                "vdw-modifier": "Force-switch",
                "nstcomm": "100",
                "comm-mode": "linear",
                "nstxout-compressed": "5000",
            }
        return {}

    def get_membrane_bound_equil_config(self) -> dict:
        """Get the temperature and pressure coupling configuration for membrane equilibration runs."""
        if self.membrane_protein:
            return {
                **self.get_membrane_nvt_config(),
                **self.get_membrane_npt_config(),
            }
        return {}

    def get_tot_simtime(self, n_runs: int, leg_type: _LegType) -> int:
        """
        Get the total simulation time for the ensemble equilibration.

        Parameters
        ----------
        n_runs : int
            Number of ensemble equilibration runs.
        leg_type : LegType
            The type of the leg.

        Returns
        -------
        int
            Total simulation time in ps.
        """

        # See functions below for where these times are used.
        tot_simtime = 0
        tot_simtime += self.runtime_short_nvt
        tot_simtime += (
            self.runtime_nvt * 2 if leg_type == _LegType.BOUND else self.runtime_nvt
        )
        tot_simtime += self.runtime_npt * 2
        tot_simtime += self.runtime_npt_unrestrained
        tot_simtime += self.ensemble_equilibration_time * n_runs
        return tot_simtime

    def save_pickle(self, save_dir: str, leg_type: _LegType) -> None:
        """
        Save the configuration to a pickle file.

        Parameters
        ----------
        save_dir : str
            Directory to save the pickle file to.

        leg_type : LegType
            The type of the leg. Used to name the pickle file.
        """
        # First, convert to dict
        model_dict = self.model_dump()

        # Save the dict to a pickle file
        save_path = save_dir + "/" + self.get_file_name(leg_type)
        with open(save_path, "wb") as f:
            _pkl.dump(model_dict, f)

    @classmethod
    def from_pickle(
        cls, save_dir: str, leg_type: _LegType
    ) -> "SystemPreparationConfig":
        """
        Load the configuration from a pickle file.

        Parameters
        ----------
        save_dir : str
            Directory to load the pickle file from.

        leg_type : LegType
            The type of the leg. Used to decide the name of the pickle
            file to load.

        Returns
        -------
        SystemPreparationConfig
            Loaded configuration.
        """

        # Load the dict from the pickle file
        load_path = save_dir + "/" + cls.get_file_name(leg_type)
        with open(load_path, "rb") as f:
            model_dict = _pkl.load(f)

        # Create the model from the dict
        return cls.parse_obj(model_dict)

    @staticmethod
    def get_file_name(leg_type: _LegType) -> str:
        """Get the name of the pickle file for the configuration."""
        return f"system_preparation_config_{leg_type.name.lower()}.pkl"


def parameterise_input(
    leg_type: _LegType, input_dir: str, output_dir: str
) -> _BSS._SireWrappers._system.System:  # type: ignore
    """
    Paramaterise the input structure, using Open Force Field v.2.0 'Sage'
    for the ligand, AMBER ff14SB for the protein, and TIP3P for the water.
    The resulting system is saved to the input directory.

    Parameters
    ----------
    leg_type : LegType
        The type of the leg.
    input_dir : str
        The path to the input directory, where the required files are located.
    output_dir : str
        The path to the output directory, where the files will be saved.

    Returns
    -------
    parameterised_system : _BSS._SireWrappers._system.System
        Parameterised system.
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    print("Parameterising input...")
    # Parameterise the ligand
    print("Parameterising ligand...")
    lig_sys = _BSS.IO.readMolecules(f"{input_dir}/ligand.sdf")
    # Ensure that the ligand is named "LIG"
    _rename_lig(lig_sys, "LIG")
    # Check charge of the ligand
    lig = lig_sys[0]
    lig_charge = round(lig.charge().value())
    if lig_charge != 0:
        _warnings.warn(
            f"Ligand has a charge of {lig_charge}. Co-alchemical ion approach will be used."
            " Ensure that your box is large enough to avoid artefacts."
        )

    # Only include ligand charge if we're using gaff (OpenFF doesn't need it)
    param_args = {"molecule": lig, "forcefield": cfg.forcefields["ligand"]}
    if "gaff" in cfg.forcefields["ligand"]:
        param_args["net_charge"] = lig_charge

    param_lig = _BSS.Parameters.parameterise(**param_args).getMolecule()

    # If bound, then parameterise the protein and waters and add to the system
    if leg_type == _LegType.BOUND:
        # Parameterise the protein
        print("Parameterising protein...")
        protein = _BSS.IO.readMolecules(f"{input_dir}/protein.pdb")[0]
        param_protein = _BSS.Parameters.parameterise(
            molecule=protein, forcefield=cfg.forcefields["protein"]
        ).getMolecule()

        # Parameterise the waters, if they are supplied
        # Check that waters are supplied
        param_waters = []
        if _pathlib.Path(f"{input_dir}/waters.pdb").exists():
            print("Crystallographic waters detected. Parameterising...")
            waters = _BSS.IO.readMolecules(f"{input_dir}/waters.pdb")
            for water in waters:
                param_waters.append(
                    _BSS.Parameters.parameterise(
                        molecule=water,
                        water_model=cfg.forcefields["water"],
                        forcefield=cfg.forcefields["protein"],
                    ).getMolecule()
                )

        # Create the system
        print("Assembling parameterised system...")
        parameterised_system = param_lig + param_protein
        for water in param_waters:
            parameterised_system += water

    # This is the free leg, so just turn the ligand into a system
    else:
        parameterised_system = param_lig.toSystem()

    # Save the system
    print("Saving parameterised system...")
    _BSS.IO.saveMolecules(
        f"{output_dir}/{leg_type.name.lower()}{_PreparationStage.PARAMETERISED.file_suffix}",
        parameterised_system,
        fileformat=["prm7", "rst7"],
    )

    return parameterised_system


def solvate_input(
    leg_type: _LegType, input_dir: str, output_dir: str
) -> _BSS._SireWrappers._system.System:  # type: ignore
    """
    Determine an appropriate (rhombic dodecahedron)
    box size, then solvate the input structure using
    TIP3P water, adding 150 mM NaCl to the system.
    The resulting system is saved to the input directory.

    Parameters
    ----------
    leg_type : LegType
        The type of the leg.
    input_dir : str
        The path to the input directory, where the required files are located.
    output_dir : str
        The path to the output directory, where the files will be saved.

    Returns
    -------
    solvated_system : _BSS._SireWrappers._system.System
        Solvated system.
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    # Load the parameterised system
    print("Loading parameterised system...")
    parameterised_system = _BSS.IO.readMolecules(
        [
            f"{input_dir}/{file}"
            for file in _PreparationStage.PARAMETERISED.get_simulation_input_files(
                leg_type
            )
        ]
    )

    # Determine the box size
    # Taken from https://github.com/michellab/BioSimSpaceTutorials/blob/main/01_introduction/02_molecular_setup.ipynb
    # Get the minimium and maximum coordinates of the bounding box that
    # minimally encloses the protein.
    print("Determining optimal rhombic dodecahedral box...")
    # Want to get box size based on complex/ ligand, exlcuding any crystallographic waters
    non_waters = [mol for mol in parameterised_system if mol.nAtoms() != 3]  # type: ignore
    dry_system = _BSS._SireWrappers._system.System(non_waters)  # type: ignore
    box_min, box_max = dry_system.getAxisAlignedBoundingBox()

    # Work out the box size from the difference in the coordinates.
    box_size = [y - x for x, y in zip(box_min, box_max)]

    # Add 15 A padding to the box size in each dimension.
    padding = 15 * _BSS.Units.Length.angstrom

    # Work out an appropriate box. This will used in each dimension to ensure
    # that the cutoff constraints are satisfied if the molecule rotates.
    box_length = max(box_size) + 2 * padding
    box, angles = _BSS.Box.rhombicDodecahedronHexagon(box_length)

    # Exclude waters if they are too far from the protein. These are unlikely
    # to be important for the simulation and including them would require a larger
    # box. Exclude if further than 10 A from the protein.
    try:
        waters_to_exclude = [
            wat
            for wat in parameterised_system.search(
                "water and not (water within 10 of protein)"
            ).molecules()
        ]
        # If we have failed to convert to molecules (old BSS bug), then do this for each molecule.
        if hasattr(waters_to_exclude[0], "toMolecule"):
            waters_to_exclude = [wat.toMolecule() for wat in waters_to_exclude]
        print(
            f"Excluding {len(waters_to_exclude)} waters that are over 10 A from the protein"
        )
    except ValueError:
        waters_to_exclude = []
    parameterised_system.removeMolecules(waters_to_exclude)

    print(f"Solvating system with {cfg.water_model} water and {cfg.ion_conc} M NaCl...")
    solvated_system = _BSS.Solvent.solvate(
        model=cfg.water_model,
        molecule=parameterised_system,
        box=box,
        angles=angles,
        ion_conc=cfg.ion_conc,
    )

    # Save the system
    print("Saving solvated system")
    _BSS.IO.saveMolecules(
        f"{output_dir}/{leg_type.name.lower()}{_PreparationStage.SOLVATED.file_suffix}",
        solvated_system,
        fileformat=["prm7", "rst7"],
    )

    return solvated_system


def minimise_input(
    leg_type: _LegType, input_dir: str, output_dir: str
) -> _BSS._SireWrappers._system.System:  # type: ignore
    """
    Minimise the input structure with GROMACS.

    Parameters
    ----------
    leg_type : LegType
        The type of the leg.
    input_dir : str
        The path to the input directory, where the required files are located.
    output_dir : str
        The path to the output directory, where the files will be saved.

    Returns
    -------
    minimised_system : _BSS._SireWrappers._system.System
        Minimised system.
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    # Load the solvated system
    print("Loading solvated system...")
    solvated_system = _BSS.IO.readMolecules(
        [
            f"{input_dir}/{file}"
            for file in _PreparationStage.SOLVATED.get_simulation_input_files(leg_type)
        ]
    )

    # Check that it is actually solvated in a box of water
    _check_has_wat_and_box(solvated_system)

    # Minimise
    print(f"Minimising input structure with {cfg.steps} steps...")
    protocol = _BSS.Protocol.Minimisation(steps=cfg.steps)
    minimised_system = run_process(solvated_system, protocol)

    # Save, renaming the velocity property to foo so avoid saving velocities. Saving the
    # velocities sometimes causes issues with the size of the floats overflowing the RST7
    # format.
    _BSS.IO.saveMolecules(
        f"{output_dir}/{leg_type.name.lower()}{_PreparationStage.MINIMISED.file_suffix}",
        minimised_system,
        fileformat=["prm7", "rst7"],
        property_map={"velocity": "foo"},
    )

    return minimised_system


def heat_and_preequil_input(
    leg_type: _LegType, input_dir: str, output_dir: str
) -> _BSS._SireWrappers._system.System:  # type: ignore
    """
    Heat the input structure from 0 to 298.15 K with GROMACS.

    Parameters
    ----------
    leg_type : LegType
        The type of the leg.
    input_dir : str
        The path to the input directory, where the required files are located.
    output_dir : str
        The path to the output directory, where the files will be saved.

    Returns
    -------
    preequilibrated_system : _BSS._SireWrappers._system.System
        Pre-Equilibrated system.
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    # Load the minimised system
    print("Loading minimised system...")
    minimised_system = _BSS.IO.readMolecules(
        [
            f"{input_dir}/{file}"
            for file in _PreparationStage.MINIMISED.get_simulation_input_files(leg_type)
        ]
    )

    # Check that it is solvated and has a box
    _check_has_wat_and_box(minimised_system)

    print(
        f"NVT equilibration for {cfg.runtime_short_nvt} ps while restraining all non-solvent atoms"
    )
    protocol = _BSS.Protocol.Equilibration(
        runtime=cfg.runtime_short_nvt * _BSS.Units.Time.picosecond,
        temperature_start=0 * _BSS.Units.Temperature.kelvin,
        temperature_end=cfg.end_temp * _BSS.Units.Temperature.kelvin,
        restraint="all",
    )
    # Only use membrane NVT config for bound leg (which has protein)
    membrane_nvt_config = (
        cfg.get_membrane_nvt_config() if leg_type == _LegType.BOUND else {}
    )
    equil1 = run_process(minimised_system, protocol, extra_options=membrane_nvt_config)

    # If this is the bound leg, carry out step with backbone restraints
    if leg_type == _LegType.BOUND:
        print(
            f"NVT equilibration for {cfg.runtime_nvt} ps while restraining all backbone atoms"
        )
        protocol = _BSS.Protocol.Equilibration(
            runtime=cfg.runtime_nvt * _BSS.Units.Time.picosecond,
            temperature=cfg.end_temp * _BSS.Units.Temperature.kelvin,
            restraint="backbone",
        )
        equil2 = run_process(equil1, protocol, extra_options=membrane_nvt_config)

    else:  # Free leg - skip the backbone restraint step
        equil2 = equil1

    print(f"NVT equilibration for {cfg.runtime_nvt} ps without restraints")
    protocol = _BSS.Protocol.Equilibration(
        runtime=cfg.runtime_nvt * _BSS.Units.Time.picosecond,
        temperature=cfg.end_temp * _BSS.Units.Temperature.kelvin,
    )
    equil3 = run_process(equil2, protocol, extra_options=membrane_nvt_config)

    print(
        f"NPT equilibration for {cfg.runtime_npt} ps while restraining non-solvent heavy atoms"
    )

    # Create basic npt protocol
    protocol_args = {
        "runtime": cfg.runtime_npt * _BSS.Units.Time.picosecond,
        "pressure": 1 * _BSS.Units.Pressure.atm,
        "temperature": cfg.end_temp * _BSS.Units.Temperature.kelvin,
    }
    # membrane protein does not need heavy atom restraints.
    if not cfg.membrane_protein:
        protocol_args["restraint"] = "heavy"
    protocol = _BSS.Protocol.Equilibration(**protocol_args)

    # NPT configuration: bound leg needs full config, free leg only needs pressure coupling
    if leg_type == _LegType.BOUND:
        membrane_npt_config = cfg.get_membrane_bound_equil_config()
    else:
        membrane_npt_config = cfg.get_membrane_npt_config()

    equil4 = run_process(equil3, protocol, extra_options=membrane_npt_config)

    print(f"NPT equilibration for {cfg.runtime_npt_unrestrained} ps without restraints")
    protocol = _BSS.Protocol.Equilibration(
        runtime=cfg.runtime_npt_unrestrained * _BSS.Units.Time.picosecond,
        pressure=1 * _BSS.Units.Pressure.atm,
        temperature=cfg.end_temp * _BSS.Units.Temperature.kelvin,
    )
    preequilibrated_system = run_process(
        equil4, protocol, extra_options=membrane_npt_config
    )

    # Save, renaming the velocity property to foo so avoid saving velocities. Saving the
    # velocities sometimes causes issues with the size of the floats overflowing the RST7
    # format.
    _BSS.IO.saveMolecules(
        f"{output_dir}/{leg_type.name.lower()}{_PreparationStage.PREEQUILIBRATED.file_suffix}",
        preequilibrated_system,
        fileformat=["prm7", "rst7"],
        property_map={"velocity": "foo"},
    )

    return preequilibrated_system


def run_ensemble_equilibration(
    leg_type: _LegType, input_dir: str, output_dir: str, short: bool = False
) -> None:
    """
    Run the ensemble equilibration for the given leg type.

    Parameters
    ----------
    leg_type : LegType
        The type of the leg.
    input_dir : str
        The path to the input directory, where the required files are located.
    output_dir : str
        The path to the output directory, where the files will be saved.
    short : bool, optional, default=False
        Whether to run the short version of the ensemble equilibration, by default False.
        This is used during testing. The short runtime is 0.1 ns.

    Returns
    -------
    None
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    # Load the pre-equilibrated system
    print("Loading pre-equilibrated system...")
    pre_equilibrated_system = _BSS.IO.readMolecules(
        [
            f"{input_dir}/{file}"
            for file in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                leg_type
            )
        ]
    )

    # Check that it is solvated in a box of water
    _check_has_wat_and_box(pre_equilibrated_system)

    # Mark the ligand to be decoupled in the absolute binding free energy calculation
    lig = _BSS.Align.decouple(pre_equilibrated_system[0], intramol=True)

    # Check that this is actually a ligand
    if lig.nAtoms() > 100 or lig.nAtoms() < 5:
        raise ValueError(
            f"The first molecule in the bound system has {lig.nAtoms()} atoms and is likely not a ligand. "
            "Please check that the ligand is the first molecule in the bound system."
        )

    # Check that the name is correct
    if lig._sire_object.name().value() != "LIG":
        raise ValueError(
            f"The name of the ligand in the bound system is {lig._sire_object.name().value()} and is not LIG. "
            "Please check that the ligand is the first molecule in the bound system or rename the ligand."
        )
    print(f"Selecting ligand {lig} for decoupling")

    # Update the system
    pre_equilibrated_system.updateMolecule(0, lig)

    # Create the protocol
    protocol = _BSS.Protocol.Production(
        timestep=2
        * _BSS.Units.Time.femtosecond,  # 2 fs timestep as 4 fs seems to cause instability even with HMR
        runtime=cfg.ensemble_equilibration_time * _BSS.Units.Time.picosecond,
    )

    # Run - assuming that this will be in the appropriate ensemble equilibration directory
    print(
        f"Running ensemble equilibration simulation with GROMACS for {cfg.ensemble_equilibration_time} ps"
    )
    if leg_type == _LegType.BOUND:
        work_dir = output_dir
        membrane_equil_config = cfg.get_membrane_bound_equil_config()
    else:
        work_dir = None
        membrane_equil_config = cfg.get_membrane_npt_config()

    final_system = run_process(
        pre_equilibrated_system,
        protocol,
        work_dir=work_dir,
        extra_options=membrane_equil_config,
    )

    # Save the coordinates only, renaming the velocity property to foo so avoid saving velocities. Saving the
    # velocities sometimes causes issues with the size of the floats overflowing the RST7
    # format.
    print(f"Saving somd.rst7 to {output_dir}")
    _BSS.IO.saveMolecules(
        f"{output_dir}/somd",
        final_system,
        fileformat=["rst7"],
        property_map={"velocity": "foo"},
    )


def run_process(
    system: _BSS._SireWrappers._system.System,
    protocol: _BSS.Protocol._protocol.Protocol,
    work_dir: _Optional[str] = None,
    extra_options: _Optional[dict] = None,
) -> _BSS._SireWrappers._system.System:
    """
    Run a process with GROMACS, raising informative
    errors in the event of a failure.

    Parameters
    ----------
    system : _BSS._SireWrappers._system.System
        System to run the process on.
    protocol : _BSS._Protocol._protocol.Protocol
        Protocol to run the process with.
    work_dir : str, optional
        The working directory to run the process in. If none,
        a temporary directory will be created.
    extra_options : dict, optional
        Extra options to pass to the GROMACS process, such as
        membrane-specific configuration. If None, no extra options are passed.

    Returns
    -------
    system : _BSS._SireWrappers._system.System
        System after the process has been run.
    """
    if extra_options is None:
        extra_options = {}
    process = _BSS.Process.Gromacs(
        system, protocol, work_dir=work_dir, extra_options=extra_options
    )
    process.start()
    process.wait()
    import time

    time.sleep(10)
    if process.isError():
        print(process.stdout())
        print(process.stderr())
        process.getStdout()
        process.getStderr()
        raise _BSS._Exceptions.ThirdPartyError("The process failed.")
    system = process.getSystem(block=True)
    if system is None:
        print(process.stdout())
        print(process.stderr())
        process.getStdout()
        process.getStderr()
        raise _BSS._Exceptions.ThirdPartyError("The final system is None.")

    return system


# Partial versions of functions for use with slurm. Avoid using functools partial
# due to issues with getting the name of the fn.


def slurm_parameterise_bound() -> None:
    """Parameterise the input structures for the bound leg"""
    parameterise_input(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")


def slurm_parameterise_free() -> None:
    """Parameterise the input structures for the free leg"""
    parameterise_input(leg_type=_LegType.FREE, input_dir=".", output_dir=".")


def slurm_solvate_bound() -> None:
    """Perform solvation for the bound leg input."""
    solvate_input(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")


def slurm_solvate_free() -> None:
    """Perform solvation for the free leg input."""
    solvate_input(_LegType.FREE, input_dir=".", output_dir=".")


def slurm_minimise_bound() -> None:
    """Perform minimisation for the bound leg"""
    minimise_input(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")


def slurm_minimise_free() -> None:
    """Perform minimisation for the free leg"""
    minimise_input(leg_type=_LegType.FREE, input_dir=".", output_dir=".")


def slurm_heat_and_preequil_bound() -> None:
    """Perform heating and minimisation for the bound leg"""
    heat_and_preequil_input(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")


def slurm_heat_and_preequil_free() -> None:
    """Perform heating and minimisation for the free leg"""
    heat_and_preequil_input(leg_type=_LegType.FREE, input_dir=".", output_dir=".")


def slurm_ensemble_equilibration_bound() -> None:
    """Perform ensemble equilibration for the bound leg"""
    run_ensemble_equilibration(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")


def slurm_ensemble_equilibration_free() -> None:
    """Perform ensemble equilibration for the free leg"""
    run_ensemble_equilibration(leg_type=_LegType.FREE, input_dir=".", output_dir=".")


def slurm_ensemble_equilibration_bound_short() -> None:
    """Perform ensemble equilibration for the bound leg"""
    run_ensemble_equilibration(
        leg_type=_LegType.BOUND, input_dir=".", output_dir=".", short=True
    )


def slurm_ensemble_equilibration_free_short() -> None:
    """Perform ensemble equilibration for the free leg"""
    run_ensemble_equilibration(
        leg_type=_LegType.FREE, input_dir=".", output_dir=".", short=True
    )
