"""Helper functions for processing BSS systems."""

import BioSimSpace.Sandpit.Exscientia as _BSS
from BioSimSpace.Sandpit.Exscientia._SireWrappers import Molecule as _Molecule
from sire.legacy import Mol as _SireMol


def rename_lig(
    bss_system: _BSS._SireWrappers._system.System, new_name: str = "LIG"
) -> None:  # type: ignore
    """Rename the ligand in a BSS system.

    Parameters
    ----------
    bss_system : BioSimSpace.Sandpit.Exscientia._SireWrappers._system.System
        The BSS system.
    new_name : str
        The new name for the ligand.
    Returns
    -------
    None
    """
    # Ensure that we only have one molecule
    if len(bss_system) != 1:
        raise ValueError("BSS system must only contain one molecule.")

    # Extract the sire object for the single molecule
    mol = _Molecule(bss_system[0])
    mol_sire = mol._sire_object

    # Create an editable version of the sire object
    mol_edit = mol_sire.edit()

    # Rename the molecule and the residue to the supplied name
    resname = _SireMol.ResName(new_name)  # type: ignore
    mol_edit = mol_edit.residue(_SireMol.ResIdx(0)).rename(resname).molecule()  # type: ignore
    mol_edit = mol_edit.edit().rename(new_name).molecule()

    # Commit the changes and update the system
    mol._sire_object = mol_edit.commit()
    bss_system.updateMolecule(0, mol)
