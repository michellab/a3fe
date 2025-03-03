"""Enums required for Classes in the run package."""

from enum import Enum as _Enum
from typing import List as _List
import yaml as _yaml

from .engine_config import _EngineConfig, SomdConfig as _SomdConfig

from typing import Any as _Any

__all__ = [
    "JobStatus",
    "StageType",
    "LegType",
    "PreparationStage",
    "EngineType",
]


class _YamlSerialisableEnum(_Enum):
    """A base class for enums that can be serialised to and deserialised from YAML."""

    @classmethod
    def to_yaml(cls, dumper: _yaml.Dumper, data: _Any) -> _yaml.nodes.ScalarNode:
        return dumper.represent_scalar(
            f"!{cls.__name__}", f"{cls.__name__}.{data.name}"
        )

    @classmethod
    def from_yaml(cls, loader: _yaml.Loader, node: _yaml.nodes.ScalarNode) -> _Any:
        value = loader.construct_scalar(node)
        enum_name, member_name = value.split(".")
        enum_class = globals()[enum_name]
        return enum_class[member_name]


# Register the custom representers and constructors for _YamlSerialisableEnum
def _yaml_enum_representer(
    dumper: _yaml.Dumper, data: _YamlSerialisableEnum
) -> _yaml.nodes.ScalarNode:
    return dumper.represent_scalar(
        f"!{data.__class__.__name__}", f"{data.__class__.__name__}.{data.name}"
    )


def _yaml_enum_constructor(
    loader: _yaml.Loader, suffix: str, node: _yaml.nodes.ScalarNode
) -> _Any:
    value = loader.construct_scalar(node)
    enum_name, member_name = value.split(".")
    enum_class = globals()[enum_name]
    return enum_class[member_name]


_yaml.add_multi_representer(_YamlSerialisableEnum, _yaml_enum_representer)
_yaml.add_multi_constructor("!", _yaml_enum_constructor)


class JobStatus(_YamlSerialisableEnum):
    """An enumeration of the possible job statuses"""

    NONE = 0
    QUEUED = 1
    FINISHED = 2
    FAILED = 3
    KILLED = 4


class StageType(_YamlSerialisableEnum):
    """Enumeration of the types of stage."""

    RESTRAIN = 1
    DISCHARGE = 2
    VANISH = 3

    @property
    def bss_perturbation_type(self) -> str:
        """Return the corresponding BSS perturbation type."""
        if self == StageType.RESTRAIN:
            return "restraint"
        elif self == StageType.DISCHARGE:
            return "discharge_soft"
        elif self == StageType.VANISH:
            return "vanish_soft"
        else:
            raise ValueError("Unknown stage type.")


class LegType(_YamlSerialisableEnum):
    """The type of leg in the calculation."""

    BOUND = 1
    FREE = 2


class EngineType(_YamlSerialisableEnum):
    SOMD = 1
    # GROMACS = 2

    @property
    def engine_config(self) -> _EngineConfig:
        """Return the configuration class for the engine."""
        engine_configs = {
            EngineType.SOMD: _SomdConfig,
            # EngineType.GROMACS: _GromacsConfig,
        }
        return engine_configs[self]

    @property
    def system_prep_config(self):
        from .system_prep_config import (
            SomdSystemPreparationConfig as _SomdSystemPreparationConfig,
        )

        """Return the system preparation configuration class."""
        system_prep_configs = {
            EngineType.SOMD: _SomdSystemPreparationConfig,
            # EngineType.GROMACS: _GromacsSystemPreparationConfig,
        }
        return system_prep_configs[self]


class PreparationStage(_YamlSerialisableEnum):
    """The stage of preparation of the input files."""

    STRUCTURES_ONLY = 1
    PARAMETERISED = 2
    SOLVATED = 3
    MINIMISED = 4
    PREEQUILIBRATED = 5

    @property
    def file_suffix(self) -> str:
        """Return the suffix to use for the files in this stage."""
        suffixes = {
            PreparationStage.STRUCTURES_ONLY: "",
            PreparationStage.PARAMETERISED: "_param",
            PreparationStage.SOLVATED: "_solv",
            PreparationStage.MINIMISED: "_min",
            PreparationStage.PREEQUILIBRATED: "_preequil",
        }
        return suffixes[self]

    @property
    def prep_fn(self):
        """The function to use to prepare the input files for this stage."""
        from ..run.system_prep import (
            parameterise_input as _parameterise_input,
            solvate_input as _solvate_input,
            minimise_input as _minimise_input,
            heat_and_preequil_input as _heat_and_preequil_input,
        )

        prep_fns = {
            PreparationStage.STRUCTURES_ONLY: None,
            PreparationStage.PARAMETERISED: _parameterise_input,
            PreparationStage.SOLVATED: _solvate_input,
            PreparationStage.MINIMISED: _minimise_input,
            PreparationStage.PREEQUILIBRATED: _heat_and_preequil_input,
        }
        return prep_fns[self]

    def get_simulation_input_files(self, leg_type: LegType) -> _List[str]:
        """Return the input files required for the simulation in this stage."""
        if self == PreparationStage.STRUCTURES_ONLY:
            if leg_type == LegType.BOUND:
                return [
                    "protein.pdb",
                    "ligand.sdf",
                ]  # Need sdf for parameterisation of lig
            elif leg_type == LegType.FREE:
                return ["ligand.sdf"]
        else:
            return [
                f"{leg_type.name.lower()}{self.file_suffix}.{file_type}"
                for file_type in ["prm7", "rst7"]
            ]
