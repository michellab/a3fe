"""Configuration classes for SOMD engine configuration."""

__all__ = [
    "SomdConfig",
]

import os as _os
from decimal import Decimal as _Decimal
from typing import Dict as _Dict, Literal as _Literal, List as _List, Union as _Union, Optional as _Optional
from pydantic import (
    BaseModel as _BaseModel, 
    Field as _Field, 
    field_validator as _field_validator, 
    model_validator as _model_validator,
    ValidationInfo as _ValidationInfo
)
from ._engine_runner_config import EngineRunnerConfig as _EngineRunnerConfig

def _get_default_lambda_array(leg: str = "BOUND", stage: str = "VANISH") -> _List[float]:
    
    default_values = {
        "BOUND": {
            "RESTRAIN": [0.0, 1.0],
            "DISCHARGE": [0.0, 0.291, 0.54, 0.776, 1.0],
            "VANISH": [
                0.0, 0.026, 0.054, 0.083, 0.111, 0.14, 0.173, 0.208,
                0.247, 0.286, 0.329, 0.373, 0.417, 0.467, 0.514,
                0.564, 0.623, 0.696, 0.833, 1.0,
            ],
        },
        "FREE": {
            "DISCHARGE": [0.0, 0.222, 0.447, 0.713, 1.0],
            "VANISH": [
                0.0, 0.026, 0.055, 0.09, 0.126, 0.164, 0.202, 0.239,
                0.276, 0.314, 0.354, 0.396, 0.437, 0.478, 0.518,
                0.559, 0.606, 0.668, 0.762, 1.0,
            ],
        },
    }   
    return default_values[leg][stage]

class SomdConfig(_EngineRunnerConfig, _BaseModel):
    """
    Pydantic model for holding SOMD engine configuration.
    """ 
    default_lambda_values: _Dict[str, _Dict[str, _List[float]]] = _Field(
        default={
            "BOUND": {
                "RESTRAIN": [0.0, 1.0],
                "DISCHARGE": [0.0, 0.291, 0.54, 0.776, 1.0],
                "VANISH": [
                    0.0, 0.026, 0.054, 0.083, 0.111, 0.14, 0.173, 0.208,
                    0.247, 0.286, 0.329, 0.373, 0.417, 0.467, 0.514,
                    0.564, 0.623, 0.696, 0.833, 1.0,
                ],
            },
            "FREE": {
                "DISCHARGE": [0.0, 0.222, 0.447, 0.713, 1.0],
                "VANISH": [
                    0.0, 0.026, 0.055, 0.09, 0.126, 0.164, 0.202, 0.239,
                    0.276, 0.314, 0.354, 0.396, 0.437, 0.478, 0.518,
                    0.559, 0.606, 0.668, 0.762, 1.0,
                ],
            },
        },
        description="Default lambda values for each leg and stage type"
    )
    ### Integrator - ncycles modified as required by a3fe ###
    nmoves: int = _Field(25000, description="Number of moves per cycle")
    timestep: float = _Field(4.0, description="Timestep in femtoseconds(fs)")
    runtime: _Union[int, float] = _Field(5.0, description="Runtime in nanoseconds(ns)")

    ### Constraints ###
    constraint: str = _Field("hbonds", description="Constraint type, must be hbonds or allbonds")
    hydrogen_mass_factor: float = _Field(3.0, alias="hydrogen mass repartitioning factor", description="Hydrogen mass repartitioning factor")
    integrator: _Literal["langevinmiddle", "leapfrogverlet"] = _Field("langevinmiddle", description="Integration algorithm")
    
    ### Thermostatting already handled by langevin integrator
    thermostat: bool = _Field(False, description="Enable thermostat")
    inverse_friction: float = _Field(1.0, description="Inverse friction in picoseconds", alias="inverse friction")
    temperature: float = _Field(25.0, description="Temperature in Celsius")

    ### Barostat ###
    barostat: bool = _Field(True, description="Enable barostat")
    pressure: float = _Field(1.0, description="Pressure in atm")

    ### Non-Bonded Interactions ###
    cutoff_type: _Literal["cutoffperiodic", "PME"] = _Field("cutoffperiodic", description="Type of cutoff to use. Options: PME, cutoffperiodic")
    cutoff_distance: float = _Field(12.0, alias="cutoff distance", ge=6.0, le=18.0, description="Cutoff distance in angstroms (6-18). Default 12.0 for cutoffperiodic and 10.0 for PME")
    reaction_field_dielectric: float = _Field(78.3, alias="reaction field dielectric",
        description="Reaction field dielectric constant (only for cutoffperiodic). "
                    "If cutoff type is PME, this value is ignored"
    )
    ### Trajectory ###
    buffered_coords_freq: int = _Field(5000,alias="buffered coordinates frequency",description="Frequency of buffered coordinates output")
    center_solute: bool = _Field(True, alias="center solute", description="Center solute in box")

    ### Minimisation ###
    minimise: bool = _Field(True, description="Perform energy minimisation")

    ### Restraints ###
    use_boresch_restraints: bool = _Field(False, description="Use Boresch restraints mode")
    turn_on_receptor_ligand_restraints: bool = _Field(False, description="Turn on receptor-ligand restraints mode")

    ### Alchemistry - restraints added by a3fe ###
    perturbed_residue_number: int = _Field(1,alias="perturbed residue number",ge=1, description="Residue number to perturb. Must be >= 1")
    energy_frequency: int = _Field(200,alias="energy frequency",description="Frequency of energy output")
    ligand_charge: int = _Field(0, description="Net charge of the ligand. If non-zero, must use PME for electrostatics.")
    charge_difference: int = _Field(0, description="Charge difference between the ligand and the system. If non-zero, must use co-alchemical ion approach.")
    
    ### Lambda ###
    lambda_array: _List[float] = _Field(
        default_factory=_get_default_lambda_array,
        description="Lambda array for alchemical perturbation, varies from 0.0 to 1.0 across stages"
    )
    lambda_val: _Optional[float] = _Field(None, description="Lambda value for current stage")

    ### Alchemical files ###
    morphfile: _Optional[str] = _Field(None, description="Path to morph file containing alchemical transformation")
    topfile: _Optional[str] = _Field(None, description="Path to topology file for the system")
    crdfile: _Optional[str] = _Field(None, description="Path to coordinate file for the system")

    boresch_restraints_dictionary: _Optional[str] = _Field(
        None, 
        #alias="boresch restraints dictionary",
        description="Optional string to hold boresch restraints dictionary content"
    )    
    ### Extra options ###
    extra_options: _Dict[str, str] = _Field(default_factory=dict, description="Extra options to pass to the SOMD engine")    

    @_field_validator('runtime')
    def validate_runtime(cls, v: float, info: _ValidationInfo) -> float:
        """Validate that runtime is a multiple of time per cycle using Decimal for precise division"""
        data = info.data
        if not ('timestep' in data and 'nmoves' in data):
            return v
        
        time_per_cycle = _Decimal(str(data['timestep'])) * _Decimal(str(data['nmoves'])) / _Decimal('1000000')
        runtime_decimal = _Decimal(str(v))
        
        if runtime_decimal % time_per_cycle != 0:
            raise ValueError(
                f"Runtime must be a multiple of the time per cycle. "
                f"Runtime: {v} ns, Time per cycle: {float(time_per_cycle)} ns"
            )
        return float(v)

    @_model_validator(mode="after")
    def _check_cutoff_values(self):
        """
        Issue warnings if the user supplies certain contradictory or unusual combos.
        """
        cutoff_type = self.cutoff_type
        cutoff_distance = self.cutoff_distance
        rfd = self.reaction_field_dielectric

        # 1) Only warn if user set reaction_field_dielectric != 78.3
        if cutoff_type == "cutoffperiodic" and rfd != 78.3:
            self._logger.warning(
                "You have cutoff_type=cutoffperiodic but set a reaction_field_dielectric. "
                f"This value ({rfd}) will be ignored by the engine."
            )

        # 2) Only warn if user picks e.g. cutoff_distance < 6 or > 18
        if cutoff_type == "PME" and not (6.0 <= cutoff_distance <= 18.0):
            self._logger.warning(
                f"For PME, we recommend cutoff_distance in [6.0, 18.0], but you have {cutoff_distance}. "
                "we'll still accept it."
            )
        return self
    
    @_model_validator(mode="after")
    def _check_charge_difference(self):
        if self.charge_difference != 0 and self.cutoff_type != "PME":
            raise ValueError("Charge difference is non-zero but cutoff type is not PME.")
        return self

    @_field_validator('constraint')
    def _check_constraint(cls, v):
        if v not in ['hbonds', 'allbonds']:
            raise ValueError('constraint must be hbonds or allbonds')
        return v    

    @_field_validator('hydrogen_mass_factor')
    def _check_hmf_range(cls, v):
        if not (1.0 <= v <= 4.0):
            raise ValueError('hydrogen_mass_factor must be between 1 and 4.')
        return v

    @_model_validator(mode="before")
    def _validate_integrator_and_thermo(cls,v):
        integrator = v.get("integrator")
        thermostat = v.get("thermostat")
        temperature = v.get("temperature", 25.0)  # Use default value if None
        pressure = v.get("pressure", 1.0)  # Use default value if None
    
        # 1) integrator='langevinmiddle' => thermostat must be False
        # 2) integrator='leapfrogverlet' => thermostat must be True
        if integrator == "langevinmiddle" and thermostat is True:
            raise ValueError("If integrator is 'langevinmiddle', thermostat must be False.")
        elif integrator == "leapfrogverlet" and thermostat is False:
            raise ValueError("If integrator is 'leapfrogverlet', thermostat must be True.")

        # check temperature is in range [-200, 1000]
        if not (-200 <= temperature <= 1000):
            raise ValueError(f"Temperature must be between -200 and 1000 Celsius, got {temperature}")

        # check pressure is in range [0, 1000] atm
        if not (0 <= pressure <= 1000):
            raise ValueError("pressure must be in range [0, 1000] atm.")

        return v

    @_model_validator(mode="after")
    def _check_charge_and_cutoff(self):
        """
        Validate that if ligand_charge != 0, then cutoff_type must be PME.
        """
        ligand_charge = self.ligand_charge
        cutoff_type = self.cutoff_type
        
        if ligand_charge != 0 and cutoff_type != "PME":
            raise ValueError(
                "Ligand charge is non-zero. Must use PME for electrostatics."
            )
        
        return self

    def get_file_name(self) -> str:
        """
        Get the name of the SOMD configuration file.
        """
        return "somd.cfg"

    def get_somd_config(self, run_dir: str) -> str:
        """
        Generates the SOMD configuration file and returns its path.

        Parameters
        ----------
        run_dir : str
            Directory to write the configuration file to.
        """
        config_filename = self.get_file_name()
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
        ]
        if self.cutoff_type == "cutoffperiodic" and self.reaction_field_dielectric is not None:
            config_lines.append(f"reaction field dielectric = {self.reaction_field_dielectric}")    

        config_lines.extend([
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
            f"energy frequency = {self.energy_frequency}",
            f"ligand charge = {self.ligand_charge}",
            f"charge difference = {self.charge_difference}",
            "",
            "### Restraints ###",
            f"use boresch restraints = {self.use_boresch_restraints}",
            f"turn on receptor-ligand restraints mode = {self.turn_on_receptor_ligand_restraints}"
        ])
        # 2) Lambda parameters
        config_lines.extend(["", "### Lambda / Alchemical Settings ###"])
        
        if self.lambda_array:
            lambda_str = f"[{', '.join(str(x) for x in self.lambda_array)}]"
            config_lines.append(f"lambda_array = {lambda_str}")
        if self.lambda_val is not None:
            config_lines.append(f"lambda_val = {self.lambda_val}")

        # 3) Alchemical files path
        config_lines.extend(["", "### Alchemical Files ###"])
        if self.morphfile:
            config_lines.append(f"morphfile = {self.morphfile}")
        if self.topfile:
            config_lines.append(f"topfile = {self.topfile}")
        if self.crdfile:
            config_lines.append(f"crdfile = {self.crdfile}")

        # 5) Boresch restraints 
        if self.boresch_restraints_dictionary is not None:
            config_lines.extend(["", "### Boresch Restraints Dictionary ###"])
            config_lines.append(f"boresch restraints dictionary = {self.boresch_restraints_dictionary}")

        # Add any extra options
        if self.extra_options:
            config_lines.extend(["", "### Extra Options ###"])
            for key, value in self.extra_options.items():
                config_lines.append(f"{key} = {value}")

        # Write the configuration to a file
        config_path = _os.path.join(run_dir, config_filename)
        with open(config_path, "w") as f:
            f.write("\n".join(config_lines) + "\n")
        
        return config_path

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

    def write_somd_config(self, output_dir: str, lam: float) -> str:
        """
        Write out the SOMD configuration file with the current settings.
        """
        self.lambda_val = lam

        # Generate somd.cfg file using the current configuration
        config_path = self.get_somd_config(run_dir=output_dir)

        return config_path
