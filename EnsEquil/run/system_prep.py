"""Functionality for running preparation simulations."""

import BioSimSpace.Sandpit.Exscientia as _BSS
from functools import partial as _partial
import pathlib as _pathlib
from typing import Callable as _Callable

from .enums import LegType as _LegType, PreparationStage as _PreparationStage
from .. read._process_bss_systems import rename_lig as _rename_lig

def parameterise_input(leg_type: _LegType, 
                       input_dir: str, 
                       output_dir: str) -> _BSS._SireWrappers._system.System: # type: ignore
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
    FORCEFIELDS = {"ligand": "openff_unconstrained-2.0.0", 
                    "protein": "ff14SB", 
                    "water": "tip3p"}

    print("Parameterising input...")
    # Parameterise the ligand
    print("Parameterising ligand...")
    lig_sys = _BSS.IO.readMolecules(f"{input_dir}/ligand.sdf")
    # Ensure that the ligand is named "LIG"
    _rename_lig(lig_sys, "LIG")
    param_lig = _BSS.Parameters.parameterise(molecule=lig_sys[0], forcefield=FORCEFIELDS["ligand"]).getMolecule()

    # If bound, then parameterise the protein and waters and add to the system
    if leg_type == _LegType.BOUND:
        # Parameterise the protein
        print("Parameterising protein...")
        protein = _BSS.IO.readMolecules(f"{input_dir}/protein.pdb")[0]
        param_protein = _BSS.Parameters.parameterise(molecule=protein, 
                                                        forcefield=FORCEFIELDS["protein"]).getMolecule()

        # Parameterise the waters, if they are supplied
        # Check that waters are supplied
        param_waters = []
        if _pathlib.Path(f"{input_dir}/waters.pdb").exists():
            print("Crystallographic waters detected. Parameterising...")
            waters = _BSS.IO.readMolecules(f"{input_dir}/waters.pdb")
            for water in waters:
                param_waters.append(_BSS.Parameters.parameterise(molecule=water, 
                                                                water_model=FORCEFIELDS["water"],
                                                                forcefield=FORCEFIELDS["protein"]).getMolecule())

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
    _BSS.IO.saveMolecules(f"{output_dir}/{leg_type.name.lower()}{_PreparationStage.PARAMETERISED.file_suffix}",
                            parameterised_system, 
                            fileformat=["prm7", "rst7"])

    return parameterised_system
    

def solvate_input(leg_type: _LegType,
                  input_dir: str,
                  output_dir: str) -> _BSS._SireWrappers._system.System: # type: ignore
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
    WATER_MODEL = "tip3p"
    ION_CONC = 0.15 # M

    # Load the parameterised system
    print("Loading parameterised system...")
    parameterised_system = _BSS.IO.readMolecules(f"{input_dir}/{leg_type.name.lower()}{_PreparationStage.PARAMETERISED.file_suffix}*")

    # Determine the box size
    # Taken from https://github.com/michellab/BioSimSpaceTutorials/blob/main/01_introduction/02_molecular_setup.ipynb
    # Get the minimium and maximum coordinates of the bounding box that
    # minimally encloses the protein.
    print("Determining optimal rhombic dodecahedral box...")
    # Want to get box size based on complex/ ligand, exlcuding any crystallographic waters
    non_waters = [mol for mol in parameterised_system if mol.nAtoms() != 3] # type: ignore
    dry_system = _BSS._SireWrappers._system.System(non_waters) # type: ignore
    box_min, box_max = dry_system.getAxisAlignedBoundingBox()

    # Work out the box size from the difference in the coordinates.
    box_size = [y - x for x, y in zip(box_min, box_max)]

    # Add 15 A padding to the box size in each dimension.
    padding = 15 * _BSS.Units.Length.angstrom

    # Work out an appropriate box. This will used in each dimension to ensure
    # that the cutoff constraints are satisfied if the molecule rotates.
    box_length = max(box_size) + 2*padding
    box, angles = _BSS.Box.rhombicDodecahedronHexagon(box_length)

    print(f"Solvating system with {WATER_MODEL} water and {ION_CONC} M NaCl...")
    solvated_system = _BSS.Solvent.solvate(model=WATER_MODEL,
                                            molecule=parameterised_system,
                                            box=box, 
                                            angles=angles, 
                                            ion_conc=ION_CONC) 

    # Save the system
    print("Saving solvated system")
    _BSS.IO.saveMolecules(f"{output_dir}/{leg_type.name.lower()}{_PreparationStage.SOLVATED.file_suffix}",
                            solvated_system, 
                            fileformat=["prm7", "rst7"])

    return solvated_system

# Partial versions of functions for use with slurm

def slurm_solvate_bound() -> None:
    """
    Perform solvation for the bound leg input.
    """
    solvate_input(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")

def slurm_solvate_free() -> None:
    """
    Perform solvation for the free leg input.
    """
    solvate_input(_LegType.FREE, input_dir=".", output_dir=".")

