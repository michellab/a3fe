"""Configuration classes for SOMD engine configuration."""

__all__ = [
    "SomdConfig",
    "ENGINE_TYPE_TO_ENGINE_CONFIG",
]

import os as _os
from decimal import Decimal as _Decimal
from typing import (
    Dict as _Dict,
    Literal as _Literal,
    List as _List,
    Union as _Union,
    Optional as _Optional,
)
from pydantic import (
    BaseModel as _BaseModel,
    Field as _Field,
    model_validator as _model_validator,
)

import yaml as _yaml
from abc import ABC as _ABC, abstractmethod as _abstractmethod
from ..run.enums import EngineType as _EngineType


class _EngineConfig(_BaseModel, _ABC):
    """Base class for engine runner configurations."""

    @staticmethod
    @_abstractmethod
    def get_file_name() -> str:
        """
        Get the name of the configuration file.
        """
        pass

    def dump(self, save_dir: str) -> None:
        """
        Dump the configuration to a YAML file using `self.model_dump()`.

        Parameters
        ----------
        save_dir : str
            Directory to dump the configuration to.
        """
        model_dict = self.model_dump()

        save_path = save_dir + "/" + self.get_file_name()
        with open(save_path, "w") as f:
            _yaml.dump(model_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, load_dir: str) -> "_EngineConfig":
        """
        Load a configuration from a YAML file.

        Parameters
        ----------
        load_dir : str
            Directory to load the configuration from.

        Returns
        -------
        config : EngineConfig
            The loaded configuration.
        """
        with open(load_dir + "/" + cls.get_file_name(), "r") as f:
                model_dict = _yaml.safe_load(f)

        return cls(**model_dict)

    @_abstractmethod
    def write_config(
        self,
        run_dir: str,
        lambda_val: float,
        runtime: float,
        top_file: str,
        coord_file: str,
        morph_file: str,
    ) -> None:
        """
        Write the configuration to a file.
        """
        pass

    @_abstractmethod
    def get_run_cmd(self) -> str:
        """
        Get the command to run the simulation.
        """
        pass


# @register_engine_config(_EngineType.SOMD)
class SomdConfig(_EngineConfig):
    """
    Pydantic model for holding SOMD engine configuration.
    """

    ### Integrator - ncycles modified as required by a3fe ###
    timestep: float = _Field(4.0, description="Timestep in femtoseconds(fs)")
    runtime: _Union[int, float] = _Field(
        5.0,
        description="Runtime in nanoseconds(ns), and must be a multiple of timestep",
    )

    ### Constraints ###
    constraint: _Literal["hbonds", "allbonds"] = _Field(
        "hbonds", description="Constraint type, must be hbonds or allbonds"
    )
    hydrogen_mass_factor: float = _Field(
        3.0,
        ge=1.0,
        le=4.0,
        alias="hydrogen mass repartitioning factor",
        description="Hydrogen mass repartitioning factor",
    )
    integrator: _Literal["langevinmiddle", "leapfrogverlet"] = _Field(
        "langevinmiddle", description="Integration algorithm"
    )

    ### Thermostatting already handled by langevin integrator
    thermostat: bool = _Field(
        False,
        description="Enable the thermodstat. Not required if using langevinmiddle integrator",
    )
    inverse_friction: float = _Field(
        1.0,
        ge=0.1,
        lt=10.0,
        description="Inverse friction in picoseconds",
        alias="inverse friction",
    )
    temperature: float = _Field(
        25.0, ge=-200.0, le=1000.0, description="Temperature in Celsius"
    )

    ### Barostat ###
    barostat: bool = _Field(True, description="Enable barostat")
    pressure: float = _Field(1.0, gt=0.0, lt=1000.0, description="Pressure in atm")

    ### Non-Bonded Interactions ###
    cutoff_type: _Literal["cutoffperiodic", "PME"] = _Field(
        "cutoffperiodic",
        description="Type of cutoff to use. Options: PME, cutoffperiodic",
    )
    cutoff_distance: float = _Field(
        12.0,
        alias="cutoff distance",
        ge=6.0,
        le=18.0,
        description="Cutoff distance in angstroms (6-18). Default 12.0 for cutoffperiodic.",
    )
    reaction_field_dielectric: float = _Field(
        78.3,
        alias="reaction field dielectric",
        description="Reaction field dielectric constant (only for cutoffperiodic). "
        "If cutoff type is PME, this value is ignored",
    )
    ### Trajectory ###
    buffered_coords_freq: int = _Field(
        5000,
        alias="buffered coordinates frequency",
        description="Frequency of buffered coordinates output",
    )
    center_solute: bool = _Field(
        True, alias="center solute", description="Center solute in box"
    )

    ### Minimisation ###
    minimise: bool = _Field(True, description="Perform energy minimisation")

    ### Restraints ###
    use_boresch_restraints: bool = _Field(
        False, description="Use Boresch restraints mode"
    )
    turn_on_receptor_ligand_restraints: bool = _Field(
        False, description="Turn on receptor-ligand restraints mode"
    )

    ### Alchemistry - restraints added by a3fe ###
    lambda_values: _Optional[_List[float]] = _Field(
        None,
        description="Lambda array for alchemical perturbation, varies from 0.0 to 1.0 across stage",
    )
    perturbed_residue_number: int = _Field(
        1,
        alias="perturbed residue number",
        ge=1,
        description="Residue number to perturb. Must be >= 1",
    )
    energy_frequency: int = _Field(
        200, alias="energy frequency", description="Frequency of energy output"
    )
    ligand_charge: int = _Field(
        0,
        description="Net charge of the ligand. If non-zero, must use PME for electrostatics.",
    )

    boresch_restraints_dictionary: _Optional[str] = _Field(
        None,
        description="Optional string to hold boresch restraints dictionary content",
    )
    ### Extra options ###
    extra_options: _Dict[str, str] = _Field(
        default_factory=dict, description="Extra options to pass to the SOMD engine"
    )

    @property
    def nmoves(self) -> int:
        """
        Make sure runtime is a multiple of timestep
        """
        # Convert runtime to femtoseconds (ns -> fs)
        runtime_fs = _Decimal(str(self.runtime)) * _Decimal("1_000_000")
        timestep = _Decimal(str(self.timestep))

        # Check if runtime is a multiple of timestep
        remainder = runtime_fs % timestep
        if round(float(remainder), 4) != 0:
            raise ValueError(
                (
                    "Runtime must be a multiple of the timestep. "
                    f"Runtime is {self.runtime} ns ({runtime_fs} fs), "
                    f"and timestep is {self.timestep} fs."
                )
            )

        # Calculate the number of moves
        nmoves = round(float(runtime_fs) / float(timestep))

        return nmoves

    @_model_validator(mode="after")
    def _check_rf_dielectric(self):
        """Warn the user if they've changed the rf dielectric constant but are using PME"""
        if (
            self.cutoff_type == "cutoffperiodic"
            and self.reaction_field_dielectric != 78.3
        ):
            self._logger.warning(
                "You have cutoff_type=cutoffperiodic but set a reaction_field_dielectric. This will result in the use of PME."
                f"This value ({self.reaction_field_dielectric}) will be ignored by the engine."
            )
        return self

    @_model_validator(mode="after")
    def _check_ligand_charge(self):
        if self.ligand_charge != 0 and self.cutoff_type != "PME":
            raise ValueError(
                "Charge difference is non-zero but cutoff type is not PME."
            )
        return self

    @_model_validator(mode="after")
    def _validate_integrator_and_thermo(self):
        integrator = self.integrator
        thermostat = self.thermostat  # Use default value if None

        # 1) integrator='langevinmiddle' => thermostat must be False
        # 2) integrator='leapfrogverlet' => thermostat must be True
        if integrator == "langevinmiddle" and thermostat is True:
            raise ValueError(
                "If integrator is 'langevinmiddle', thermostat must be False."
            )
        elif integrator == "leapfrogverlet" and thermostat is False:
            raise ValueError(
                "If integrator is 'leapfrogverlet', thermostat must be True."
            )
        return self

    @staticmethod
    def get_file_name() -> str:
        """
        Get the name of the SOMD configuration file.
        """
        return "somd.cfg"

    def write_config(
        self,
        run_dir: str,
        lambda_val: float,
        runtime: float,
        top_file: str,
        coord_file: str,
        morph_file: str,
    ) -> None:
        """
        Generates the SOMD configuration file and returns its path.

        Parameters
        ----------
        run_dir : str
            Directory to write the configuration file to.

        lambda_val : float
            Current lambda value

        runtime : float
            Total runtime in nanoseconds.

        top_file : str
            Path to the topology file.

        coord_file : str
            Path to the coordinate file.

        morph_file : str
            Path to the morph file.
        """
        self.runtime = runtime

        if self.lambda_array is None:
            raise ValueError("lambda_array must be set before writing the configuration.")

        config_lines = [
            "### Integrator ###",
            f"timestep = {self.timestep} * femtosecond",
            f"nmoves = {self.nmoves}",
            f"constraint = {self.constraint}",
            f"hydrogen mass repartitioning factor = {self.hydrogen_mass_factor}",
            f"integrator = {self.integrator}",
            f"inverse friction = {self.inverse_friction} * picosecond",
            f"temperature = {self.temperature} * celsius",
            f"thermostat = {self.thermostat}",
            "\n\n### Barostat ###",
            f"barostat = {self.barostat}",
            f"pressure = {self.pressure} * atm",
            "\n\n### Non-Bonded Interactions ###",
            f"cutoff type = {self.cutoff_type}",
            f"cutoff distance = {self.cutoff_distance} * angstrom",
        ]
        if (
            self.cutoff_type == "cutoffperiodic"
            and self.reaction_field_dielectric is not None
        ):
            config_lines.append(
                f"reaction field dielectric = {self.reaction_field_dielectric}"
            )

        config_lines.extend(
            [
                "\n\n### Trajectory ###",
                f"buffered coordinates frequency = {self.buffered_coords_freq}",
                f"center solute = {self.center_solute}",
                "\n\n### Minimisation ###",
                f"minimise = {self.minimise}",
                "\n\n### Alchemistry ###",
                f"perturbed residue number = {self.perturbed_residue_number}",
                f"energy frequency = {self.energy_frequency}",
                f"ligand charge = {self.ligand_charge}",
                f"lambda array = {', '.join(str(x) for x in self.lambda_values)}",
                f"lambda_val = {lambda_val}",
                "\n\n### Restraints ###",
                f"use boresch restraints = {self.use_boresch_restraints}",
                f"turn on receptor-ligand restraints mode = {self.turn_on_receptor_ligand_restraints}"
                "\n\n###Paths###",
                f"morphfile = {morph_file}",
                f"topfile = {top_file}",
                f"crdfile = {coord_file}",
            ]
        )

        # 5) Boresch restraints
        if self.boresch_restraints_dictionary is not None:
            config_lines.extend(["", "### Boresch Restraints Dictionary ###"])
            config_lines.append(
                f"boresch restraints dictionary = {self.boresch_restraints_dictionary}"
            )

        # Add any extra options
        if self.extra_options:
            config_lines.extend(["", "### Extra Options ###"])
            for key, value in self.extra_options.items():
                config_lines.append(f"{key} = {value}")

        # Write the configuration to a file
        run_dir = _os.path.dirname(run_dir)
        config_filename = self.get_file_name()
        config_path = _os.path.join(run_dir, config_filename)
        with open(config_path, "w") as f:
            f.write("\n".join(config_lines) + "\n")

    @classmethod
    def _from_config_file(cls, config_path: str) -> "SomdConfig":
        """Create a SomdConfig instance from an existing configuration file."""
        with open(config_path, "r") as f:
            config_content = f.read()

        config_dict = {}
        for line in config_content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = [x.strip() for x in line.split("=", 1)]

                if key == "lambda array":
                    value = [float(x.strip()) for x in value.split(",")]
                elif "*" in value:
                    value = value.split("*")[0].strip()
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                elif value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    try:
                        if "." in value:
                            value = float(value)
                        elif value.isdigit():
                            value = int(value)
                    except ValueError:
                        pass
                key = key.replace(" ", "_")
                config_dict[key] = value

        return cls(**config_dict)

    def get_run_cmd(self, lam: float) -> str:
        """
        Get the command to run the simulation.
        """
        return f"somd-freenrg -C {self.get_file_name()} -l {lam} -p CUDA"


ENGINE_TYPE_TO_ENGINE_CONFIG = {_EngineType.SOMD: SomdConfig}