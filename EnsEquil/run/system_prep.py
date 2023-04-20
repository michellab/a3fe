"""Functionality for running preparation simulations."""

import BioSimSpace.Sandpit.Exscientia as _BSS
import pathlib as _pathlib
from typing import Optional as _Optional

from .enums import LegType as _LegType, PreparationStage as _PreparationStage
from .. read._process_bss_systems import rename_lig as _rename_lig
from ._utils import check_has_wat_and_box as _check_has_wat_and_box

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
    parameterised_system = _BSS.IO.readMolecules([f"{input_dir}/{file}" for file in _PreparationStage.PARAMETERISED.get_simulation_input_files(leg_type)])

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

def minimise_input(leg_type: _LegType,
                   input_dir: str,
                   output_dir: str) -> _BSS._SireWrappers._system.System: # type: ignore
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
    STEPS = 1000 # This is the default for _BSS
    
    # Load the solvated system
    print("Loading solvated system...")
    solvated_system = _BSS.IO.readMolecules([f"{input_dir}/{file}" for file in _PreparationStage.SOLVATED.get_simulation_input_files(leg_type)])

    # Check that it is actually solvated in a box of water
    _check_has_wat_and_box(solvated_system)

    # Minimise
    print(f"Minimising input structure with {STEPS} steps...")
    protocol = _BSS.Protocol.Minimisation(steps=STEPS)
    minimised_system = run_process(solvated_system, protocol)

    # Save, renaming the velocity property to foo so avoid saving velocities. Saving the
    # velocities sometimes causes issues with the size of the floats overflowing the RST7
    # format.
    _BSS.IO.saveMolecules(f"{output_dir}/{leg_type.name.lower()}{_PreparationStage.MINIMISED.file_suffix}",
                            minimised_system, 
                            fileformat=["prm7", "rst7"], property_map={"velocity" : "foo"})

    return minimised_system

def heat_and_preequil_input(leg_type: _LegType,
                            input_dir: str,
                            output_dir: str) -> _BSS._SireWrappers._system.System: # type: ignore
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
    RUNTIME_SHORT_NVT = 5 # ps
    RUNTIME_NVT = 50 # ps 
    END_TEMP = 298.15 # K
    RUNTIME_NPT = 400 # ps
    RUNTIME_NPT_UNRESTRAINED = 1000 # ps

    # Load the minimised system
    print("Loading minimised system...")
    minimised_system = _BSS.IO.readMolecules([f"{input_dir}/{file}" for file in _PreparationStage.MINIMISED.get_simulation_input_files(leg_type)])

    # Check that it is solvated and has a box
    _check_has_wat_and_box(minimised_system)

    print(f"NVT equilibration for {RUNTIME_SHORT_NVT} ps while restraining all non-solvent atoms")
    protocol = _BSS.Protocol.Equilibration(
                                    runtime=RUNTIME_SHORT_NVT*_BSS.Units.Time.picosecond, 
                                    temperature_start=0*_BSS.Units.Temperature.kelvin, 
                                    temperature_end=END_TEMP*_BSS.Units.Temperature.kelvin,
                                    restraint="all"
                                    )
    equil1 = run_process(minimised_system, protocol)

    # If this is the bound leg, carry out step with backbone restraints
    if leg_type == _LegType.BOUND:
        print(f"NVT equilibration for {RUNTIME_NVT} ps while restraining all backbone atoms")
        protocol = _BSS.Protocol.Equilibration(
                                        runtime=RUNTIME_NVT*_BSS.Units.Time.picosecond, 
                                        temperature=END_TEMP*_BSS.Units.Temperature.kelvin, 
                                        restraint="backbone"
                                        )
        equil2 = run_process(equil1, protocol)

    else: # Free leg - skip the backbone restraint step
        equil2 = equil1

    print(f"NVT equilibration for {RUNTIME_NVT} ps without restraints")
    protocol = _BSS.Protocol.Equilibration(
                                    runtime=RUNTIME_NVT*_BSS.Units.Time.picosecond, 
                                    temperature=END_TEMP*_BSS.Units.Temperature.kelvin,
                                    )
    equil3 = run_process(equil2, protocol)

    print(f"NPT equilibration for {RUNTIME_NPT} ps while restraining non-solvent heavy atoms")
    protocol = _BSS.Protocol.Equilibration(
                                    runtime=RUNTIME_NPT*_BSS.Units.Time.picosecond, 
                                    pressure=1*_BSS.Units.Pressure.atm,
                                    temperature=END_TEMP*_BSS.Units.Temperature.kelvin,
                                    restraint="heavy",
                                    )
    equil4 = run_process(equil3, protocol)

    print(f"NPT equilibration for {RUNTIME_NPT_UNRESTRAINED} ps without restraints")
    protocol = _BSS.Protocol.Equilibration(
                                    runtime=RUNTIME_NPT_UNRESTRAINED*_BSS.Units.Time.picosecond, 
                                    pressure=1*_BSS.Units.Pressure.atm,
                                    temperature=END_TEMP*_BSS.Units.Temperature.kelvin,
                                    )
    preequilibrated_system = run_process(equil4, protocol)

    # Save, renaming the velocity property to foo so avoid saving velocities. Saving the
    # velocities sometimes causes issues with the size of the floats overflowing the RST7
    # format.
    _BSS.IO.saveMolecules(f"{output_dir}/{leg_type.name.lower()}{_PreparationStage.PREEQUILIBRATED.file_suffix}",
                            preequilibrated_system, 
                            fileformat=["prm7", "rst7"], property_map={"velocity" : "foo"})

    return preequilibrated_system

def run_ensemble_equilibration(leg_type: _LegType,
                               input_dir: str,
                               output_dir: str) -> None:
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
    
    Returns
    -------
    None
    """
    ENSEMBLE_EQUILIBRATION_TIME = 5 # ns

    # Load the pre-equilibrated system
    print("Loading pre-equilibrated system...")
    pre_equilibrated_system = _BSS.IO.readMolecules([f"{input_dir}/{file}" for file in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(leg_type)])

    # Check that it is solvated in a box of water
    _check_has_wat_and_box(pre_equilibrated_system)

    # Mark the ligand to be decoupled in the absolute binding free energy calculation
    lig = _BSS.Align.decouple(pre_equilibrated_system[0], intramol=True)

    # Check that this is actually a ligand
    if lig.nAtoms() > 100 or lig.nAtoms() < 5:
        raise ValueError(f"The first molecule in the bound system has {lig.nAtoms()} atoms and is likely not a ligand. " \
                            "Please check that the ligand is the first molecule in the bound system.")

    # Check that the name is correct
    if lig._sire_object.name().value() != "LIG":
        raise ValueError(f"The name of the ligand in the bound system is {lig._sire_object.name().value()} and is not LIG. " \
                            "Please check that the ligand is the first molecule in the bound system or rename the ligand.")
    print(f"Selecting ligand {lig} for decoupling")

    # Update the system
    pre_equilibrated_system.updateMolecule(0,lig)
    
    # Create the protocol
    protocol = _BSS.Protocol.Production(timestep=2*_BSS.Units.Time.femtosecond, # 2 fs timestep as 4 fs seems to cause instability even with HMR
                                            runtime=ENSEMBLE_EQUILIBRATION_TIME*_BSS.Units.Time.nanosecond)

    # Run - assuming that this will be in the appropriate ensemble equilibration directory
    print(f"Running SOMD ensemble equilibration simulation for {ENSEMBLE_EQUILIBRATION_TIME} ns")
    if leg_type == _LegType.BOUND:
        work_dir = output_dir
    else:
        work_dir = None
    final_system = run_process(pre_equilibrated_system, protocol, work_dir=work_dir)

    # Save the coordinates only, renaming the velocity property to foo so avoid saving velocities. Saving the
    # velocities sometimes causes issues with the size of the floats overflowing the RST7
    # format.
    print(f"Saving somd.rst7 to {output_dir}")
    _BSS.IO.saveMolecules(f"{output_dir}/somd",
                          final_system,
                          fileformat=["rst7"], 
                          property_map={"velocity" : "foo"})

def run_process(system: _BSS._SireWrappers._system.System,
                 protocol: _BSS.Protocol._protocol.Protocol,
                 work_dir: _Optional[str]=None) -> _BSS._SireWrappers._system.System:
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
    
    Returns
    -------
    system : _BSS._SireWrappers._system.System
        System after the process has been run.
    """
    process = _BSS.Process.Gromacs(system, protocol, work_dir=work_dir)
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

# Partial versions of functions for use with slurm. Avoid using functools partial due to issues with getting the name of the fn.

def slurm_parameterise_bound() -> None:
    """ Parameterise the input structures for the bound leg """
    parameterise_input(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")

def slurm_parameterise_free() -> None:
    """ Parameterise the input structures for the free leg """
    parameterise_input(leg_type=_LegType.FREE, input_dir=".", output_dir=".")

def slurm_solvate_bound() -> None:
    """ Perform solvation for the bound leg input.  """
    solvate_input(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")

def slurm_solvate_free() -> None:
    """ Perform solvation for the free leg input.  """
    solvate_input(_LegType.FREE, input_dir=".", output_dir=".")

def slurm_minimise_bound() -> None:
    """ Perform minimisation for the bound leg"""
    minimise_input(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")

def slurm_minimise_free() -> None:
    """ Perform minimisation for the free leg"""
    minimise_input(leg_type=_LegType.FREE, input_dir=".", output_dir=".")

def slurm_heat_and_preequil_bound() -> None:
    """ Perform heating and minimisation for the bound leg"""
    heat_and_preequil_input(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")

def slurm_heat_and_preequil_free() -> None:
    """ Perform heating and minimisation for the free leg"""
    heat_and_preequil_input(leg_type=_LegType.FREE, input_dir=".", output_dir=".")

def slurm_ensemble_equilibration_bound() -> None:
    """ Perform ensemble equilibration for the bound leg"""
    run_ensemble_equilibration(leg_type=_LegType.BOUND, input_dir=".", output_dir=".")

def slurm_ensemble_equilibration_free() -> None:
    """ Perform ensemble equilibration for the free leg"""
    run_ensemble_equilibration(leg_type=_LegType.FREE, input_dir=".", output_dir=".")