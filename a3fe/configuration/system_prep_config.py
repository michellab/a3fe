"""
Configuration classes for system preparation.
"""

__all__ = [
    "SomdSystemPreparationConfig",
]

import yaml as _yaml

from abc import ABC as _ABC

from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field
from pydantic import ConfigDict as _ConfigDict

from .enums import StageType as _StageType
from .enums import LegType as _LegType

from typing import List as _List, Dict as _Dict


class _BaseSystemPreparationConfig(_ABC, _BaseModel):
    """
    Pydantic model for holding system preparation configuration.
    """

    slurm: bool = _Field(True, description="Whether to use SLURM for the preparation.")
    forcefields: dict = _Field(
        default={
            "ligand": "openff_unconstrained-2.0.0",
            "protein": "ff14SB",
            "water": "tip3p",
        },
        description="Forcefields to use for the ligand, protein, and water.",
    )
    water_model: str = _Field("tip3p", description="Water model to use.")
    ion_conc: float = _Field(0.15, ge=0, lt=1, description="Ion concentration in M.")
    steps: int = _Field(
        1000, gt=0, lt=100_000, description="Number of steps for the minimisation."
    )
    runtime_short_nvt: int = _Field(
        5, gt=0, lt=500, description="Runtime for the short NVT equilibration in ps."
    )
    runtime_nvt: int = _Field(
        50, gt=0, lt=5_000, description="Runtime for the NVT equilibration in ps."
    )
    end_temp: float = _Field(
        298.15,
        gt=0,
        lt=350,
        description="End temperature for the NVT equilibration in K.",
    )
    runtime_npt: int = _Field(
        400, gt=0, lt=40_000, description="Runtime for the NPT equilibration in ps."
    )
    runtime_npt_unrestrained: int = _Field(
        1000,
        gt=0,
        lt=100_000,
        description="Runtime for the unrestrained NPT equilibration in ps.",
    )
    ensemble_equilibration_time: int = _Field(
        5000, gt=0, lt=50_000, description="Ensemble equilibration time in ps."
    )
    append_to_ligand_selection: str = _Field(
        "",
        description="If this is a bound leg, this appends the supplied string to the default atom selection which chooses the atoms in the ligand to consider as potential anchor points. The default atom selection is f'resname {ligand_resname} and not name H*'. Uses the mdanalysis atom selection language. For example, 'not name O*' will result in an atom selection of f'resname {ligand_resname} and not name H* and not name O*'.",
    )
    lambda_values: _Dict[_LegType, _Dict[_StageType, _List[float]]] = _Field(
        default={
            _LegType.BOUND: {
                _StageType.RESTRAIN: [0.0, 1.0],
                _StageType.DISCHARGE: [0.0, 0.291, 0.54, 0.776, 1.0],
                _StageType.VANISH: [
                    0.0,
                    0.026,
                    0.054,
                    0.083,
                    0.111,
                    0.14,
                    0.173,
                    0.208,
                    0.247,
                    0.286,
                    0.329,
                    0.373,
                    0.417,
                    0.467,
                    0.514,
                    0.564,
                    0.623,
                    0.696,
                    0.833,
                    1.0,
                ],
            },
            _LegType.FREE: {
                _StageType.DISCHARGE: [0.0, 0.222, 0.447, 0.713, 1.0],
                _StageType.VANISH: [
                    0.0,
                    0.026,
                    0.055,
                    0.09,
                    0.126,
                    0.164,
                    0.202,
                    0.239,
                    0.276,
                    0.314,
                    0.354,
                    0.396,
                    0.437,
                    0.478,
                    0.518,
                    0.559,
                    0.606,
                    0.668,
                    0.762,
                    1.0,
                ],
            },
        },
        description="The lambda values to use for each stage of each leg.",
    )

    @property
    def required_stages(self) -> _Dict[_LegType, _List[_StageType]]:
        """
        Get the required stages for each leg type.

        Returns
        -------
        Dict[LegType, List[StageType]]
            Required stages for each leg type.
        """
        return {
            leg_type: list(self.lambda_values[leg_type].keys())
            for leg_type in self.lambda_values.keys()
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
    def load(cls, save_dir: str, leg_type: _LegType) -> "_BaseSystemPreparationConfig":
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


class SomdSystemPreparationConfig(_BaseSystemPreparationConfig):
    """
    Pydantic model for holding system preparation configuration
    for running simulations with SOMD.

    Currently this doesn't modify the base settings, but it may do
    in the future.
    """
