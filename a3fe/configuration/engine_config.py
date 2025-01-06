"""Configuration classes for SOMD engine configuration."""

__all__ = [
    "SomdConfig",
]

import yaml as _yaml
import os as _os
from typing import Dict as _Dict

from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field
from pydantic import ConfigDict as _ConfigDict


class SomdConfig(_BaseModel):
    """
    Pydantic model for holding SOMD engine configuration.
    """
    
    ### Integrator - ncycles modified as required by a3fe ###
    nmoves: int = _Field(25000, description="Number of moves per cycle")
    ncycles: int = _Field(60, description="Number of cycles")
    timestep: float = _Field(4.0, description="Timestep in femtoseconds")
    constraint: str = _Field("hbonds", description="Constraint type")
    hydrogen_mass_factor: float = _Field(
        3.0, 
        alias="hydrogen mass repartitioning factor",
        description="Hydrogen mass repartitioning factor"
    )
    integrator: str = _Field("langevinmiddle", description="Integration algorithm")
    inverse_friction: float = _Field(
        1.0, 
        description="Inverse friction in picoseconds",
        alias="inverse friction"
    )
    temperature: float = _Field(25.0, description="Temperature in Celsius")
    # Thermostatting already handled by langevin integrator
    thermostat: bool = _Field(False, description="Enable thermostat")

    ### Barostat ###
    barostat: bool = _Field(True, description="Enable barostat")
    pressure: float = _Field(1.0, description="Pressure in atm")

    ### Non-Bonded Interactions ###
    cutoff_type: str = _Field(
        "PME",
        alias="cutoff type",
        description="Type of cutoff to use"
    )
    cutoff_distance: float = _Field(
        10.0,
        alias="cutoff distance",
        description="Cutoff distance in angstroms"
    )
    
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
    extra_options: _Dict[str, str] = _Field(
        default_factory=dict,
        description="Extra options to pass to the SOMD engine"
    )
    model_config = _ConfigDict(validate_assignment=True)

    def get_somd_config(self, run_dir: str, config_name: str = "somd_config") -> str:
        """
        Generates the SOMD configuration file and returns its path.

        Parameters
        ----------
        #content : str
            Content to write to the configuration file.
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
            f"ncycles = {self.ncycles}",
            f"timestep = {self.timestep} * femtosecond",
            f"constraint = {self.constraint}",
            f"hydrogen mass repartitioning factor = {self.hydrogen_mass_factor}",
            f"integrator = {self.integrator}",
            f"inverse friction = {self.inverse_friction} * picosecond",
            f"temperature = {self.temperature} * celsius",
            f"thermostat = {str(self.thermostat)}",
            "",
            "### Barostat ###",
            f"barostat = {str(self.barostat)}",
            f"pressure = {self.pressure} * atm",
            "",
            "### Non-Bonded Interactions ###",
            f"cutoff type = {self.cutoff_type}",
            f"cutoff distance = {self.cutoff_distance} * angstrom",
            "",
            "### Trajectory ###",
            f"buffered coordinates frequency = {self.buffered_coords_freq}",
            f"center solute = {str(self.center_solute)}",
            "",
            "### Minimisation ###",
            f"minimise = {str(self.minimise)}",
            "",
            "### Alchemistry ###",
            f"perturbed residue number = {self.perturbed_residue_number}",
            f"energy frequency = {self.energy_frequency}",
            "",
        ]
        # Add any extra options
        if self.extra_options:
            config_lines.extend(["", "### Extra Options ###"])
            for key, value in self.extra_options.items():
                config_lines.append(f"{key} = {value}")

        # Write the configuration to a file
        config_path = _os.path.join(run_dir, f"{config_name}.cfg")
        with open(config_path, "w") as f:
            f.write("\n".join(config_lines))

        return config_path

    def dump(self, save_dir: str) -> None:
        """
        Dumps the configuration to a YAML file.

        Parameters
        ----------
        save_dir : str
            Directory to save the YAML file to.
        """
        model_dict = self.model_dump()
        
        save_path = save_dir + "/" + self.get_file_name()
        with open(save_path, "w") as f:
            _yaml.dump(model_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, load_dir: str) -> "SomdConfig":
        """
        Loads the configuration from a YAML file.

        Parameters
        ----------
        load_dir : str
            Directory to load the YAML file from.

        Returns
        -------
        SomdConfig
            The loaded configuration.
        """
        with open(load_dir + "/" + cls.get_file_name(), "r") as f:
            model_dict = _yaml.safe_load(f)
        return cls(**model_dict)

    @staticmethod
    def get_file_name() -> str:
        """
        Get the name of the SOMD configuration file.
        """
        return "somd_config.yaml"
