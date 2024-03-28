"""
A light-weight restraint class for holding required restraint information
without the entire system, as done by BioSimSpace.
"""

from dataclasses import dataclass as _dataclass
from typing import Protocol as _Protocol

import BioSimSpace.Sandpit.Exscientia as _BSS


class Restraint(_Protocol):
    """Defines the required functionality for a restraint object."""

    def getCorrection(self, method: str) -> float: ...

    def toString(self, engine: str) -> str: ...


@_dataclass
class A3feRestraint:
    """
    A light-weight restraint class for holding restraint information.
    Unlike the BioSimSpace restraint class, this does not store the entire
    system.

    Attributes
    ----------
    somd_restr_string : str
        The restraint string for the SOMD engine.
    analytical_corr : _BSS.Types._energy.Energy
        The analytical correction, in kcal/mol.
    numerical_corr : _BSS.Types._energy.Energy
        The numerical correction, in kcal/mol.
    """

    somd_restr_string: str
    analytical_corr: _BSS.Types._energy.Energy
    numerical_corr: _BSS.Types._energy.Energy

    def __init__(self, bss_restraint: _BSS.FreeEnergy._restraint.Restraint):
        """
        Initialize the restraint.

        Parameters
        ----------
        bss_restraint : BioSimSpace.Sandpit.Exscientia.FreeEnergy._restraint.Restraint
            The BioSimSpace restraint object.
        """
        self.somd_restr_string = bss_restraint.toString(engine="SOMD")
        self.analytical_corr = bss_restraint.getCorrection(method="analytical")
        self.numerical_corr = bss_restraint.getCorrection(method="numerical")

    def getCorrection(self, method: str = "analytical") -> _BSS.Types._energy.Energy:
        """
        Get the correction for a given method.

        Parameters
        ----------
        method : str
            The method to get the correction for.

        Returns
        -------
        BSS.Types._energy.Energy
            The correction, in kcal/mol.
        """
        if method == "analytical":
            return self.analytical_corr
        elif method == "numerical":
            return self.numerical_corr
        else:
            raise ValueError(f"Unknown method: {method}")

    def toString(self, engine: str = "SOMD") -> str:
        """
        Get the restraint string for a given engine.

        Parameters
        ----------
        engine : str
            The engine to get the restraint string for.

        Returns
        -------
        str
            The restraint string.
        """
        if engine == "SOMD":
            return self.somd_restr_string
        else:
            raise ValueError(f"Unknown engine: {engine}")
