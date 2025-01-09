"""Configuration classes for SOMD engine configuration."""

__all__ = [
    "SomdConfig",
]

import os as _os
import logging as _logging
from typing import Dict as _Dict, Literal as _Literal, List as _List, Union as _Union, Optional as _Optional, Any as _Any
from math import isclose as _isclose        
from pydantic import (
    BaseModel as _BaseModel, 
    Field as _Field, 
    ConfigDict as _ConfigDict, 
    validator as _validator, 
    root_validator as _root_validator,
    field_validator as _field_validator
)

from ._engine_runner_config import EngineRunnerConfig as _EngineRunnerConfig


class SomdConfig(_EngineRunnerConfig, _BaseModel):
    """
    Pydantic model for holding SOMD engine configuration.
    """
    
    ### Integrator - ncycles modified as required by a3fe ###
    nmoves: int = _Field(25000, description="Number of moves per cycle")
    timestep: float = _Field(4.0, description="Timestep in femtoseconds(fs)")
    runtime: _Union[int, float] = _Field(..., description="Runtime in nanoseconds(ns)")

    input_dir: str = _Field(..., description="Input directory containing simulation config files")
    @staticmethod
    def _calculate_ncycles(runtime: float, time_per_cycle: float) -> int:
        """
        Calculate the number of cycles given a runtime and time per cycle.

        Parameters
        ----------
        runtime : float
            Runtime in nanoseconds.
        time_per_cycle : float
            Time per cycle in nanoseconds.

        Returns
        -------
        int
            Number of cycles.
        """
        return round(runtime / time_per_cycle)

    def _set_n_cycles(self, n_cycles: int) -> None:
        """
        Set the number of cycles in the SOMD config file.

        Parameters
        ----------
        n_cycles : int
            Number of cycles to set in the somd config file.
        """
        with open(_os.path.join(self.input_dir, "somd.cfg"), "r") as ifile:
            lines = ifile.readlines()
        for i, line in enumerate(lines):
            if line.startswith("ncycles ="):
                lines[i] = "ncycles = " + str(n_cycles) + "\n"
                break
        #Now write the new file
        with open(_os.path.join(self.input_dir, "somd.cfg"), "w+") as ofile:
            for line in lines:
                ofile.write(line)

    def _validate_runtime_and_update_config(self) -> None:
        """
        Validate the runtime and update the simulation configuration.

        Need to make sure that runtime is a multiple of the time per cycle
        otherwise actual time could be quite different from requested runtime

        Raises
        ------
        ValueError
            If runtime is not a multiple of the time per cycle.
        """
        time_per_cycle = self.timestep * self.nmoves / 1_000_000  # Convert fs to ns
        # Convert both to float for division
        remainder = float(self.runtime) / float(time_per_cycle)
        if not _isclose(remainder - round(remainder), 0, abs_tol=1e-5):
            raise ValueError(
                f"Runtime must be a multiple of the time per cycle. "
                f"Runtime: {self.runtime} ns, Time per cycle: {time_per_cycle} ns."
            )
        
        # Only try to modify the config file if it exists
        cfg_file = _os.path.join(self.input_dir, "somd.cfg")
        if _os.path.exists(cfg_file):
            # Need to modify the config file to set the correction n_cycles
            n_cycles = self._calculate_ncycles(self.runtime, time_per_cycle)
            self._set_n_cycles(n_cycles)
            print(f"Updated ncycles to {n_cycles} in somd.cfg")

    constraint: str = _Field("hbonds", description="Constraint type, must be hbonds or all-bonds")

    @_validator('constraint')
    def _check_constraint(cls, v):
        if v not in ['hbonds', 'all-bonds']:
            raise ValueError('constraint must be hbonds or all-bonds')
        return v

    hydrogen_mass_factor: float = _Field(
        3.0, 
        alias="hydrogen mass repartitioning factor",
        description="Hydrogen mass repartitioning factor"
    )
    integrator: _Literal["langevinmiddle", "leapfrogverlet"] = _Field("langevinmiddle", description="Integration algorithm")
    # Thermostatting already handled by langevin integrator
    thermostat: bool = _Field(False, description="Enable thermostat")

    @_root_validator(pre=True)
    def _validate_integrator_thermostat(cls, v):
        '''
        Make sure that if integrator is 'langevinmiddle' then thermostat must be False
        '''
        integrator = v.get('integrator')
        thermostat = v.get('thermostat', False)  # Default to False if not provided

        if integrator == "langevinmiddle" and thermostat is not False:
            raise ValueError("thermostat must be False when integrator is langevinmiddle")
        return v
    
    inverse_friction: float = _Field(
        1.0, 
        description="Inverse friction in picoseconds",
        alias="inverse friction"
    )
    temperature: float = _Field(25.0, description="Temperature in Celsius")

    ### Barostat ###
    barostat: bool = _Field(True, description="Enable barostat")
    pressure: float = _Field(1.0, description="Pressure in atm")

    ### Non-Bonded Interactions ###
    cutoff_type: _Literal["PME", "cutoffperiodic"] = _Field(
        "PME",
        description="Type of cutoff to use. Options: PME, cutoffperiodic"
    )
    cutoff_distance: float = _Field(
        10.0,  # Default to PME cutoff distance
        alias="cutoff distance",
        description="Cutoff distance in angstroms"
    )
    reaction_field_dielectric: float | None = _Field(
        None,
        alias="reaction field dielectric",
        description="Reaction field dielectric constant(only for cutoffperiodic)"
    )
    @_validator('cutoff_type')
    def _check_cutoff_type(cls, v):
        if v not in ['PME', 'cutoffperiodic']:
            raise ValueError('cutoff type must be PME or cutoffperiodic')
        return v
    
    @_validator('cutoff_distance', always=True)
    def _set_cutoff_distance(cls, v, values):
        if values.get('cutoff_type') == 'PME':
            return 10.0 if v is None else v
        elif values.get('cutoff_type') == 'cutoffperiodic':
            return 12.0 if v is None else v
        return v
    
    @_validator('reaction_field_dielectric',always=True)
    def _set_reaction_field_dielectric(cls, v, values):
        cutoff_type = values.get('cutoff_type')
        if cutoff_type == 'PME' and v is not None:
            raise ValueError('reaction field dielectric should not be provided when cutoff type is PME')
        elif cutoff_type == 'cutoffperiodic' and v is None:
            return 78.3  # Default dielectric constant for cutoffperiodic
        return v

    @_field_validator("cutoff_distance", mode="before")
    def _validate_cutoff_distance(cls, v, values):
        """
        Validate cutoff distance based on cutoff type.
        """
        cutoff_type = values.data.get("cutoff_type")
        if cutoff_type == "cutoffperiodic":
            return 12.0  # Default for cutoffperiodic
        return v  # Default for PME (10.0)

    model_config = _ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, stream_log_level: int = _logging.INFO, **data):
        _BaseModel.__init__(self, **data)
        _EngineRunnerConfig.__init__(self, stream_log_level=stream_log_level)
        self._validate_runtime_and_update_config()

    def get_config(self) -> _Dict[str, _Any]:
        """
        Get the SOMD configuration as a dictionary.

        Returns
        -------
        config : Dict[str, Any]
            The SOMD configuration dictionary.
        """
        return self.model_dump()

    def get_file_name(self) -> str:
        """
        Get the name of the SOMD configuration file.

        Returns
        -------
        file_name : str
            The name of the SOMD configuration file.
        """
        return "somd.cfg"

    ### Trajectory ###
    buffered_coords_freq: int = _Field(
        5000,
        alias="buffered coordinates frequency",
        description="Frequency of buffered coordinates output"
    )
    center_solute: bool = _Field(
        True,
        alias="center solute",
        description="Center solute in box"
    )

    ### Minimisation ###
    minimise: bool = _Field(True, description="Perform energy minimisation")

    ### Restraints ###
    use_boresch_restraints: bool = _Field(
        False,
        description="UseBoresch restraints mode"
    )
    receptor_ligand_restraints: bool = _Field(
        False,
        description="Turn on receptor-ligand restraints mode"
    )

    ### Alchemistry - restraints added by a3fe ###
    perturbed_residue_number: int = _Field(
        1,
        alias="perturbed residue number",
        description="Residue number to perturb"
    )
    energy_frequency: int = _Field(
        200,
        alias="energy frequency",
        description="Frequency of energy output"
    )

    ### Lambda ###
    lambda_array: _List[float] = _Field(
        default_factory=list,
        description="Lambda array for alchemical perturbation, varies from 0.0 to 1.0 across stages"
    )
    lambda_val: _Optional[float] = _Field(
        None,
        description="Lambda value for current stage"
    )

    ### Alchemical files ###
    morphfile: _Optional[str] = _Field(
        None, description="Path to morph file containing alchemical transformation"
    )
    topfile: _Optional[str] = _Field(
        None, description="Path to topology file for the system"
    )
    crdfile: _Optional[str] = _Field(
        None, description="Path to coordinate file for the system"
    )
    
    extra_options: _Dict[str, str] = _Field(
        default_factory=dict,
        description="Extra options to pass to the SOMD engine"
    )

    def get_somd_config(self, run_dir: str, config_name: str = "somd_config") -> str:
        """
        Generates the SOMD configuration file and returns its path.

        Parameters
        ----------
        run_dir : str
            Directory to write the configuration file to.

        config_name : str, optional, default="somd_config"
            Name of the configuration file to write. Note that when running many jobs from the
            same directory, this should be unique to avoid overwriting the config file.

        Returns
        -------
        str
            Path to the generated configuration file.
        """
        # First, generate the configuration string
        config_lines = [
            "### Integrator ###",
            f"nmoves = {self.nmoves}",
            f"timestep = {self.timestep} * femtosecond",
            f"constraint = {self.constraint}",
            f"hydrogen mass repartitioning factor = {self.hydrogen_mass_factor}",
            f"integrator = {self.integrator}",
            f"inverse friction = {self.inverse_friction} * picosecond",
            f"temperature = {self.temperature} * celsius",
            f"thermostat = {self.thermostat}",
            "",
            "### Barostat ###",
            f"barostat = {self.barostat}",
            f"pressure = {self.pressure} * atm",
            "",
            "### Non-Bonded Interactions ###",
            f"cutoff type = {self.cutoff_type}",
            f"cutoff distance = {self.cutoff_distance} * angstrom",
            "",
            "### Trajectory ###",
            f"buffered coordinates frequency = {self.buffered_coords_freq}",
            f"center solute = {self.center_solute}",
            "",
            "### Minimisation ###",
            f"minimise = {self.minimise}",
            "",
            "### Alchemistry ###",
            f"perturbed residue number = {self.perturbed_residue_number}",
            f"energy frequency = {self.energy_frequency}"
        ]
        # Add any extra options
        if self.extra_options:
            config_lines.extend(["", "### Extra Options ###"])
            for key, value in self.extra_options.items():
                config_lines.append(f"{key} = {value}")

        # Write the configuration to a file
        config_path = _os.path.join(run_dir, f"{config_name}.cfg")
        with open(config_path, "w") as f:
            f.write("\n".join(config_lines) + "\n")

        return config_path
