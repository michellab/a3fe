"""
Configuration classes for system preparation.
"""

__all__ = [
    "SystemPreparationConfig",
]

import yaml as _yaml

from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field
from pydantic import ConfigDict as _ConfigDict

from ..run.enums import StageType as _StageType
from ..run.enums import LegType as _LegType


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

    model_config = _ConfigDict(extra="forbid", validate_assignment=True)

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

    def dump(self, save_dir: str, leg_type: _LegType) -> None:
        """
        Save the configuration to a YAML file.

        Parameters
        ----------
        save_dir : str
            Directory to save the YAML file to.

        leg_type : LegType
            The type of the leg. Used to name the YAML file.
        """
        # First, convert to dict
        model_dict = self.model_dump()

        # Save the dict to a YAML file
        save_path = save_dir + "/" + self.get_file_name(leg_type)
        with open(save_path, "w") as f:
            _yaml.dump(model_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, save_dir: str, leg_type: _LegType) -> "SystemPreparationConfig":
        """
        Load the configuration from a YAML file.

        Parameters
        ----------
        save_dir : str
            Directory to load the YAML file from.

        leg_type : LegType
            The type of the leg. Used to decide the name of the YAML
            file to load.

        Returns
        -------
        SystemPreparationConfig
            Loaded configuration.
        """

        # Load the dict from the YAML file
        load_path = save_dir + "/" + cls.get_file_name(leg_type)
        with open(load_path, "r") as f:
            model_dict = _yaml.load(f, Loader=_yaml.FullLoader)

        # Create the model from the dict
        return cls.model_validate(model_dict)

    @staticmethod
    def get_file_name(leg_type: _LegType) -> str:
        """Get the name of the YAML file for the configuration."""
        return f"system_preparation_config_{leg_type.name.lower()}.yaml"
