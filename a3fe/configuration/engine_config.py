"""Configuration classes for SOMD engine configuration."""

__all__ = [
    "SomdConfig",
    "GromacsConfig",
]

import os as _os
import shlex as _shlex
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from decimal import Decimal as _Decimal
from typing import (
    Dict as _Dict,
)
from typing import (
    List as _List,
)
from typing import (
    Literal as _Literal,
)
from typing import (
    Optional as _Optional,
)
from typing import (
    Union as _Union,
)

import yaml as _yaml
from pydantic import (
    BaseModel as _BaseModel,
)
from pydantic import (
    Field as _Field,
)
from pydantic import (
    model_validator as _model_validator,
)


class _EngineConfig(_BaseModel, _ABC):
    """Base class for engine runner configurations."""

    model_config = {
        "populate_by_name": True,
        "validate_assignment": True,
    }

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

    def setup_lambda_arrays(self, stage_type) -> None:
        """
        Make sure GROMACS has the same stages as SOMD.
        """
        pass


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

        if self.lambda_values is None:
            raise ValueError(
                "lambda_array must be set before writing the configuration."
            )

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
                f"turn on receptor-ligand restraints mode = {self.turn_on_receptor_ligand_restraints}",
                "\n\n###Paths###",
                f"morphfile = {_os.path.join(run_dir, morph_file)}",
                f"topfile = {_os.path.join(run_dir, top_file)}",
                f"crdfile = {_os.path.join(run_dir, coord_file)}",
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
                    config_dict["lambda_values"] = value
                    continue
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


class GromacsConfig(_EngineConfig):
    """
    Pydantic model for holding GROMACS engine configuration.
    Based on fragment-opt-abfe-benchmark mdp format with 4fs timestep.
    """

    ### Simulation Type ###
    executable: str = _Field(
        "gmx", description="GROMACS executable, for example 'gmx' or 'gmx_mpi'"
    )
    mdp_type: _Literal["em", "prod"] = _Field("prod", description="Type of simulation")

    ### Run Control ###
    define: _Optional[str] = _Field(
        None, description="Preprocessor defines (e.g., -DPOSRES, -DFLEXIBLE)"
    )
    integrator: _Literal["sd", "steep", "md"] = _Field(
        "sd", description="Integrator type"
    )
    nsteps: int = _Field(5000000, description="Number of MD steps")
    dt: float = _Field(0.002, description="Timestep in ps (2 fs)")
    comm_mode: _Literal["Linear", "Angular", "None"] = _Field(
        "Linear", description="Center of mass motion removal"
    )
    nstcomm: int = _Field(50, description="Frequency for COM removal")
    emtol: _Optional[float] = _Field(
        None, description="Energy minimization tolerance (for EM)"
    )
    emstep: _Optional[float] = _Field(None, description="Initial step size for EM")
    pbc: _Literal["xyz", "xy", "no"] = _Field("xyz", description="PBC type")

    ### Output Control ###
    nstxout: int = _Field(0, description="Write coords to .trr")
    nstvout: int = _Field(0, description="Write velocities to .trr")
    nstfout: int = _Field(0, description="Write forces to .trr")
    nstxout_compressed: int = _Field(500, description="Write compressed trajectory")
    compressed_x_precision: int = _Field(1000, description="Compressed precision")
    nstlog: int = _Field(500, description="Update log file")
    nstenergy: int = _Field(500, description="Save energies")
    nstcalcenergy: int = _Field(50, description="Calculate energies")

    ### Bonds ###
    constraint_algorithm: _Literal["lincs", "shake"] = _Field(
        "lincs", description="Constraint algorithm"
    )
    constraints: _Literal["none", "h-bonds", "all-bonds"] = _Field(
        "h-bonds", description="Constraint type (none/h-bonds/all-bonds)"
    )
    lincs_iter: int = _Field(1, description="LINCS iterations")
    lincs_order: int = _Field(6, description="LINCS order")
    lincs_warnangle: int = _Field(30, description="LINCS warning angle")
    continuation: _Literal["yes", "no"] = _Field("yes", description="Continuation")

    ### Neighbor Searching ###
    cutoff_scheme: _Literal["Verlet", "group"] = _Field(
        "Verlet", description="Cutoff scheme"
    )
    ns_type: _Literal["grid", "simple"] = _Field("grid", description="Neighbor search")
    nstlist: int = _Field(20, description="Update neighbor list")
    rlist: float = _Field(1.2, description="Neighbor list cutoff (nm)")

    ### Electrostatics ###
    coulombtype: _Literal["PME", "Cut-off"] = _Field("PME", description="Coulomb type")
    rcoulomb: float = _Field(1.0, description="Coulomb cutoff (nm)")
    ewald_geometry: _Literal["3d", "3dc"] = _Field("3d", description="Ewald geometry")
    pme_order: int = _Field(4, description="PME order")
    fourierspacing: float = _Field(0.10, description="PME grid spacing (nm)")
    ewald_rtol: float = _Field(1e-6, description="Ewald tolerance")

    ### VDW ###
    vdwtype: _Literal["Cut-off", "PME"] = _Field("Cut-off", description="VdW type")
    vdw_modifier: _Literal["Potential-shift-Verlet", "None"] = _Field(
        "Potential-shift-Verlet", description="VdW modifier"
    )
    verlet_buffer_tolerance: float = _Field(
        0.005, description="Verlet buffer tolerance"
    )
    rvdw: float = _Field(1.0, description="VdW cutoff (nm)")
    DispCorr: _Literal["EnerPres", "Ener", "no"] = _Field(
        "EnerPres", description="Long range dispersion corrections"
    )

    ### Temperature Coupling ###
    tcoupl: _Literal["no", "yes"] = _Field("no", description="Temperature coupling")
    tc_grps: str = _Field("System", description="Temperature coupling groups")
    tau_t: float = _Field(2.0, description="Time constant for T-coupling (ps)")
    ref_t: float = _Field(298.15, description="Reference temperature (K)")

    ### Pressure Coupling ###
    pcoupl: _Literal["no", "Berendsen", "C-rescale", "Parrinello-Rahman"] = _Field(
        "Parrinello-Rahman", description="Pressure coupling"
    )
    pcoupltype: _Literal["isotropic", "semiisotropic"] = _Field(
        "isotropic", description="Pressure coupling type"
    )
    tau_p: float = _Field(2.0, description="Time constant for P-coupling (ps)")
    ref_p: float = _Field(1.01325, description="Reference pressure (bar)")
    compressibility: float = _Field(4.5e-5, description="Compressibility (bar^-1)")
    refcoord_scaling: _Optional[_Literal["all", "com", "no"]] = _Field(
        None, description="Reference coordinate scaling"
    )

    ### Velocity Generation ###
    gen_vel: _Literal["yes", "no"] = _Field("no", description="Generate velocities")
    gen_seed: int = _Field(-1, description="Random seed")
    gen_temp: float = _Field(298.15, description="Generation temperature (K)")

    ### Restraints (for interface compatibility, not used in MDP generation) ###
    use_boresch_restraints: bool = _Field(
        False,
        description="Use Boresch restraints mode (interface compatibility, handled in topology)",
    )
    turn_on_receptor_ligand_restraints: bool = _Field(
        False,
        description="Turn on receptor-ligand restraints mode (interface compatibility, handled in topology)",
    )
    boresch_restraints_dictionary: _Optional[str] = _Field(
        None,
        description="Boresch restraints dictionary content (interface compatibility, handled in topology)",
    )

    ### Free Energy ###
    perturbed_residue_number: int = _Field(
        1,
        alias="perturbed residue number",
        ge=1,
        description="Residue number to perturb. Must be >= 1",
    )

    ligand_charge: int = _Field(
        0,
        description="Net charge of the ligand. If non-zero, must use PME for electrostatics.",
    )

    free_energy: _Literal["yes", "no"] = _Field("yes", description="Enable FEP")
    couple_moltype: str = _Field(
        "LIG", description="Molecule type to couple (e.g., 'LIG')"
    )

    couple_lambda0: str = _Field(
        "vdw-q",
        description="Interactions at lambda=0 (e.g., 'vdw-q', 'vdw', 'q', 'none')",
    )

    couple_lambda1: str = _Field(
        "none",
        description="Interactions at lambda=1 (e.g., 'vdw-q', 'vdw', 'q', 'none')",
    )
    sc_alpha: float = _Field(0.5, description="Soft-core alpha")
    sc_power: int = _Field(1, description="Soft-core power")
    sc_sigma: float = _Field(0.3, description="Soft-core sigma (nm)")
    init_lambda_state: _Optional[int] = _Field(None, description="Initial lambda state")

    lambda_values: _Optional[_List[float]] = _Field(
        None, description="Lambda values for this stage (set by system_prep_config)"
    )

    bonded_lambdas: _Optional[_List[float]] = _Field(
        None, description="Bonded lambda values"
    )
    coul_lambdas: _Optional[_List[float]] = _Field(
        None, description="Coulomb lambda values"
    )
    vdw_lambdas: _Optional[_List[float]] = _Field(None, description="VdW lambda values")

    nstdhdl: int = _Field(100, description="Frequency to write dH/dlambda")
    dhdl_print_energy: _Literal["total", "potential"] = _Field(
        "total", description="dH/dlambda energy output"
    )
    calc_lambda_neighbors: int = _Field(
        -1,
        description="Lambda neighbors to calculate (-1 = all for MBAR, 1 = adjacent for BAR)",
    )
    separate_dhdl_file: _Literal["yes", "no"] = _Field(
        "yes", description="Separate dH/dlambda file"
    )
    couple_intramol: _Literal["yes", "no"] = _Field(
        "yes", description="Couple intramolecular"
    )

    ### Extra options ###
    extra_options: _Dict[str, str] = _Field(
        default_factory=dict, description="Extra options"
    )

    @staticmethod
    def get_file_name() -> str:
        return "gromacs.mdp"

    @property
    def timestep(self) -> float:
        """Return timestep in femtoseconds for compatibility with SomdConfig."""
        return self.dt * 1000.0  # ps to fs

    def setup_lambda_arrays(self, stage_type) -> None:
        """
        Set up GROMACS-specific bonded/coul/vdw lambda arrays.

        Parameters
        ----------
        stage_type : StageType
            The type of stage (RESTRAIN, DISCHARGE, or VANISH)
        """
        if self.lambda_values is None:
            raise ValueError(
                "lambda_values must be set before calling _get_lambda_arrays_for_stage(). "
                "This should be set from GromacsSystemPreparationConfig."
            )

        stage = stage_type.name.lower()

        if stage == "restrain":
            self.bonded_lambdas = self.lambda_values
            self.coul_lambdas = [0.0] * len(self.lambda_values)
            self.vdw_lambdas = [0.0] * len(self.lambda_values)
        elif stage == "discharge":
            self.bonded_lambdas = [1.0] * len(self.lambda_values)
            self.coul_lambdas = self.lambda_values
            self.vdw_lambdas = [0.0] * len(self.lambda_values)
        elif stage == "vanish":
            self.bonded_lambdas = [1.0] * len(self.lambda_values)
            self.coul_lambdas = [1.0] * len(self.lambda_values)
            self.vdw_lambdas = self.lambda_values
        else:
            raise ValueError(f"Unknown stage type: {stage}")

    def _configure_for_mdp_type(self) -> None:
        """
        Configure parameters based on mdp_type.
        Energy minimisation uses a stage-specific copy of the configuration, so
        these changes never leak back into the production settings.
        """
        if self.mdp_type == "em":
            self.integrator = "steep"
            self.constraints = "none"
            self.tcoupl = "no"
            self.pcoupl = "no"
            self.gen_vel = "no"
            self.nsteps = 1000
            self.emtol = 10
            self.emstep = 0.01
            self.nstcomm = 100
            self.nstxout = 250
            self.nstlist = 1
        else:
            # Match the fragment-opt-abfe-benchmark production protocol.
            self.continuation = "yes"
            self.gen_vel = "no"

    def write_config(
        self,
        run_dir: str,
        lambda_val: float,
        runtime: float,
        top_file: str = "",
        coord_file: str = "",
        morph_file: str = "",
    ) -> None:
        """
        Generate GROMACS mdp file matching fragment-opt-abfe-benchmark format.
        """
        # Configure based on type
        self._configure_for_mdp_type()

        # Override nsteps for prod based on runtime. Decimal arithmetic avoids
        # silently truncating a requested runtime because of floating-point noise.
        if self.mdp_type == "prod":
            runtime_ps = _Decimal(str(runtime)) * _Decimal("1000")
            dt_ps = _Decimal(str(self.dt))
            if runtime_ps <= 0:
                raise ValueError("GROMACS runtime must be greater than zero.")
            nsteps = runtime_ps / dt_ps
            if nsteps != nsteps.to_integral_value():
                raise ValueError(
                    f"Runtime {runtime} ns is not an exact multiple of the "
                    f"GROMACS timestep {self.dt} ps."
                )
            self.nsteps = int(nsteps)
            if self.nsteps < self.nstdhdl:
                raise ValueError(
                    f"Runtime {runtime} ns produces only {self.nsteps} steps, fewer "
                    f"than one nstdhdl interval ({self.nstdhdl} steps)."
                )

        # Find lambda state index from the active lambda array
        # Priority: find array containing lambda_val, otherwise use varying array
        lambda_array = None

        # First, try to find array containing lambda_val
        if self.vdw_lambdas and lambda_val in self.vdw_lambdas:
            lambda_array = self.vdw_lambdas
        elif self.coul_lambdas and lambda_val in self.coul_lambdas:
            lambda_array = self.coul_lambdas
        elif self.bonded_lambdas and lambda_val in self.bonded_lambdas:
            lambda_array = self.bonded_lambdas

        # If not found, use varying array (not all values are the same)
        if lambda_array is None:
            if self.vdw_lambdas and len(set(self.vdw_lambdas)) > 1:
                lambda_array = self.vdw_lambdas
            elif self.coul_lambdas and len(set(self.coul_lambdas)) > 1:
                lambda_array = self.coul_lambdas
            elif self.bonded_lambdas and len(set(self.bonded_lambdas)) > 1:
                lambda_array = self.bonded_lambdas

        if lambda_array is None:
            raise ValueError(
                f"Lambda {lambda_val} not found in any lambda array. "
                f"bonded: {self.bonded_lambdas}, "
                f"coul: {self.coul_lambdas}, "
                f"vdw: {self.vdw_lambdas}"
            )

        try:
            self.init_lambda_state = lambda_array.index(lambda_val)
        except ValueError:
            raise ValueError(f"Lambda {lambda_val} not found in {lambda_array}")

        # Build mdp content
        mdp_lines = [
            ";====================================================",
            f"; {self.mdp_type.upper().replace('-', ' ')} {'simulation' if self.mdp_type != 'em' else ''}".strip(),
            ";====================================================",
            "",
        ]

        # Run Control
        if self.mdp_type == "em":
            mdp_lines.extend(
                [
                    ";----------------------------------------------------",
                    "; RUN CONTROL & MINIMIZATION",
                    ";----------------------------------------------------",
                ]
            )
            if self.define:
                mdp_lines.append(f"define                 = {self.define}")
            mdp_lines.extend(
                [
                    f"integrator             = {self.integrator}",
                    f"nsteps                 = {self.nsteps}",
                    f"emtol                  = {self.emtol}",
                    f"emstep                 = {self.emstep}",
                    f"nstcomm                = {self.nstcomm}",
                    f"pbc                    = {self.pbc}",
                ]
            )
        else:
            mdp_lines.extend(
                [
                    "; RUN CONTROL",
                    ";----------------------------------------------------",
                ]
            )
            if self.define:
                mdp_lines.append(f"define       = {self.define}")
            mdp_lines.extend(
                [
                    f"integrator   = {self.integrator:<13} ; langevin integrator",
                    f"nsteps       = {self.nsteps:<13} ; {self.nsteps * self.dt:.3f} ps",
                    f"dt           = {self.dt:<13} ; {self.dt * 1000:.0f} fs",
                    f"comm-mode    = {self.comm_mode:<13} ; remove center of mass translation",
                    f"nstcomm      = {self.nstcomm:<13} ; frequency for center of mass motion removal",
                ]
            )

        # Output Control
        mdp_lines.extend(
            [
                "",
                "; OUTPUT CONTROL"
                if self.mdp_type != "em"
                else ";----------------------------------------------------",
                ""
                if self.mdp_type == "em"
                else ";----------------------------------------------------",
                f"nstxout                = {self.nstxout:<10} ; "
                + (
                    "save coordinates to .trr every " + str(self.nstxout) + " steps"
                    if self.nstxout > 0
                    else "don't save coordinates to .trr"
                ),
                f"nstvout                = {self.nstvout:<10} ; don't save velocities to .trr",
                f"nstfout                = {self.nstfout:<10} ; don't save forces to .trr",
                "",
                f"nstxout-compressed     = {self.nstxout_compressed:<10} ; xtc compressed trajectory output every {self.nstxout_compressed} steps",
                f"compressed-x-precision = {self.compressed_x_precision}",
                f"nstlog                 = {self.nstlog:<10} ; update log file every {self.nstlog} steps",
                f"nstenergy              = {self.nstenergy:<10} ; save energies every {self.nstenergy} steps",
                f"nstcalcenergy          = {self.nstcalcenergy}",
                "",
            ]
        )

        # Neighbor Searching
        if self.mdp_type == "em":
            mdp_lines.extend(
                [
                    ";----------------------------------------------------",
                    "; NEIGHBOR SEARCHING",
                    ";----------------------------------------------------",
                    f"cutoff-scheme          = {self.cutoff_scheme}",
                    f"nstlist                = {self.nstlist}",
                    f"rlist                  = {self.rlist}",
                    "",
                ]
            )
        else:
            mdp_lines.extend(
                [
                    ";----------------------------------------------------",
                    ";----------------------------------------------------",
                    f"cutoff-scheme       = {self.cutoff_scheme}",
                    f"nstlist             = {self.nstlist:<6} ; {self.nstlist * self.dt * 1000:.0f} fs",
                    f"rlist               = {self.rlist:<6} ; short-range neighborlist cutoff (in nm)",
                    f"pbc                 = {self.pbc:<6} ; 3D PBC",
                    "",
                ]
            )

        # Bonds (skip for EM)
        if self.mdp_type != "em":
            constraints_comment = (
                " ; all bonds are constrained (HMR)"
                if self.constraints == "all-bonds"
                else ""
            )

            bonds_section = [
                ";----------------------------------------------------",
                ";----------------------------------------------------",
                f"constraint_algorithm   = {self.constraint_algorithm:<9} ; holonomic constraints",
                f"constraints            = {self.constraints:<9}{constraints_comment}",
            ]

            if self.constraints != "none":
                bonds_section.extend(
                    [
                        f"lincs_iter             = {self.lincs_iter:<9} ; accuracy of LINCS (1 is default)",
                        f"lincs_order            = {self.lincs_order:<9} ; also related to accuracy (4 is default)",
                        f"lincs-warnangle        = {self.lincs_warnangle:<9} ; maximum angle that a bond can rotate before LINCS will complain (30 is default)",
                    ]
                )

            bonds_section.extend([f"continuation           = {self.continuation}", ""])
            mdp_lines.extend(bonds_section)

        # Electrostatics
        if self.mdp_type == "em":
            mdp_lines.extend(
                [
                    ";----------------------------------------------------",
                    "; ELECTROSTATICS",
                    ";----------------------------------------------------",
                    f"coulombtype            = {self.coulombtype}",
                    f"rcoulomb               = {self.rcoulomb}",
                    f"pme-order              = {self.pme_order}",
                    f"fourierspacing         = {self.fourierspacing}",
                    f"ewald-rtol             = {self.ewald_rtol}",
                    "",
                ]
            )
        else:
            mdp_lines.extend(
                [
                    "; ELECTROSTATICS & EWALD",
                    ";----------------------------------------------------",
                    f"coulombtype      = {self.coulombtype:<6} ; Particle Mesh Ewald for long-range electrostatics",
                    f"rcoulomb         = {self.rcoulomb:<6} ; short-range electrostatic cutoff (in nm)",
                    f"ewald_geometry   = {self.ewald_geometry:<6} ; Ewald sum is performed in all three dimensions",
                    f"pme-order        = {self.pme_order:<6} ; interpolation order for PME (default is 4)",
                    f"fourierspacing   = {self.fourierspacing:<6} ; grid spacing for FFT",
                    f"ewald-rtol       = {self.ewald_rtol:<6} ; relative strength of the Ewald-shifted direct potential at rcoulomb",
                    "",
                ]
            )

        # VDW
        vdw_header = [
            ";----------------------------------------------------",
            "; VDW",
            ";----------------------------------------------------",
        ]

        mdp_lines.extend(
            [
                *vdw_header,
                f"vdwtype                 = {self.vdwtype}",
                f"vdw-modifier            = {self.vdw_modifier}",
                f"verlet-buffer-tolerance = {self.verlet_buffer_tolerance}",
                f"rvdw                    = {self.rvdw}          ; short-range van der Waals cutoff (in nm)",
                f"DispCorr                = {self.DispCorr}     ; apply long range dispersion corrections for Energy and Pressure",
                "",
            ]
        )

        # Bonds (for EM only)
        if self.mdp_type == "em":
            mdp_lines.extend(
                [
                    ";----------------------------------------------------",
                    "; BONDS",
                    ";----------------------------------------------------",
                    f"constraints            = {self.constraints}",
                    "",
                ]
            )

        # Temperature & Pressure Coupling
        if self.mdp_type == "em":
            mdp_lines.extend(
                [
                    ";----------------------------------------------------",
                    "; TEMPERATURE & PRESSURE COUPL",
                    ";----------------------------------------------------",
                    f"tcoupl              = {self.tcoupl}",
                    f"pcoupl              = {self.pcoupl}",
                    f"gen-vel             = {self.gen_vel}",
                    "",
                ]
            )
        else:
            mdp_lines.extend(
                [
                    ";----------------------------------------------------",
                    "; TEMPERATURE & PRESSURE COUPL",
                    ";----------------------------------------------------",
                    f"tc-grps          = {self.tc_grps}",
                    f"tau-t            = {self.tau_t}",
                    f"ref-t            = {self.ref_t}",
                    f"pcoupl           = {self.pcoupl}",
                    f"pcoupltype       = {self.pcoupltype}            ; uniform scaling of box vectors",
                    f"tau-p            = {self.tau_p}                  ; time constant (ps)",
                    f"ref-p            = {self.ref_p}              ; reference pressure (bar)",
                    f"compressibility  = {self.compressibility}              ; isothermal compressibility of water (bar^-1)",
                ]
            )
            if self.refcoord_scaling:
                mdp_lines.append(f"refcoord-scaling = {self.refcoord_scaling}")
            mdp_lines.append("")

        # Velocity Generation (only for non-EM stages)
        if self.mdp_type != "em":
            mdp_lines.extend(
                [
                    "; VELOCITY GENERATION",
                    ";----------------------------------------------------",
                    f"gen_vel      = {self.gen_vel}       ; Velocity generation is {'on' if self.gen_vel == 'yes' else 'off'}",
                    f"gen-seed     = {self.gen_seed}       ; Use random seed",
                    f"gen-temp     = {self.gen_temp}",
                    "",
                ]
            )

        # Free Energy
        if self.free_energy == "yes":
            mdp_lines.extend(
                [
                    "; FREE ENERGY",
                    ";----------------------------------------------------",
                    f"free-energy              = {self.free_energy}",
                    f"couple-moltype           = {self.couple_moltype}",
                    f"couple-lambda0           = {self.couple_lambda0}",
                    f"couple-lambda1           = {self.couple_lambda1}",
                    f"sc-alpha                 = {self.sc_alpha}",
                    f"sc-power                 = {self.sc_power}",
                    f"sc-sigma                 = {self.sc_sigma}",
                    f"init-lambda-state        = {'<state>' if self.init_lambda_state is None else self.init_lambda_state}",
                ]
            )

            if self.bonded_lambdas:
                mdp_lines.append(
                    f"bonded-lambdas           = {' '.join(str(x) for x in self.bonded_lambdas)}"
                )

            if self.coul_lambdas:
                mdp_lines.append(
                    f"coul-lambdas             = {' '.join(str(x) for x in self.coul_lambdas)}"
                )

            if self.vdw_lambdas:
                mdp_lines.append(
                    f"vdw-lambdas              = {' '.join(str(x) for x in self.vdw_lambdas)}"
                )

            mdp_lines.extend(
                [
                    f"nstdhdl                  = {self.nstdhdl}",
                    f"dhdl-print-energy        = {self.dhdl_print_energy}",
                    f"calc-lambda-neighbors    = {self.calc_lambda_neighbors}",
                    f"separate-dhdl-file       = {self.separate_dhdl_file}",
                    f"couple-intramol          = {self.couple_intramol}",
                ]
            )

        # Extra options
        if self.extra_options:
            mdp_lines.append("")
            for key, value in self.extra_options.items():
                mdp_lines.append(f"{key} = {value}")

        # Write file
        config_path = _os.path.join(run_dir, self.get_file_name())
        with open(config_path, "w") as f:
            f.write("\n".join(mdp_lines) + "\n")

    def write_all_stage_configs(
        self,
        run_dir: str,
        lambda_val: float,
        runtime: float,
    ) -> None:
        """
        Generate GROMACS energy-minimisation and production MDP files.
        Creates one subdirectory for each stage and writes its MDP file.

        Parameters
        ----------
        run_dir : str
            Base directory (e.g., output/lambda_X.XXX/run_YY/)
        lambda_val : float
            Lambda value for this simulation
        runtime : float
            Runtime for production stage (ns)
        """
        stages = ["em", "prod"]

        for stage in stages:
            stage_dir = _os.path.join(run_dir, stage)
            _os.makedirs(stage_dir, exist_ok=True)

            # Keep EM-specific settings from mutating the production configuration.
            stage_config = self.model_copy(deep=True)
            stage_config.mdp_type = stage

            # Set define based on stage
            if stage == "em":
                stage_config.define = "-DFLEXIBLE"
            else:
                stage_config.define = None

            stage_config.write_config(
                run_dir=stage_dir,
                lambda_val=lambda_val,
                runtime=runtime if stage == "prod" else 0.1,
            )

    def get_run_cmd(self, lam: float) -> str:
        stages = ["em", "prod"]
        commands = []
        executable = _shlex.quote(self.executable)

        for i, stage in enumerate(stages):
            # prepare input coordinates
            if i == 0:
                input_gro = "../gromacs.gro"
            else:
                prev_stage = stages[i - 1]
                input_gro = f"../{prev_stage}/{prev_stage}.gro"

            # grompp + mdrun
            cmd = (
                f"cd {stage} && "
                f"{executable} grompp -f gromacs.mdp -c {input_gro} -p ../gromacs.top -o {stage}.tpr && "
                f"{executable} mdrun -s {stage}.tpr -deffnm {stage} -v && "
                f"cd .."
            )
            commands.append(cmd)

        return " && ".join(commands)
