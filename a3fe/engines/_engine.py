"""Abstract interface for simulation engine backends."""

from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from typing import Any as _Any
from typing import List as _List
from typing import Tuple as _Tuple

import numpy as _np

from ..configuration import LegType as _LegType
from ..configuration import StageType as _StageType
from ..configuration import _EngineConfig


class EngineBackend(_ABC):
    """Stateless interface for engine-specific run behaviour."""

    run_engine: str
    """Engine name used by BioSimSpace."""

    supports_adaptive: bool
    """Whether the engine supports adaptive simulations."""

    coordinate_prefix_and_extension: _Tuple[str, str]
    """Prefix and extension of the engine coordinate file."""

    required_input_files: _Tuple[str, ...]
    """Files required to run a simulation."""

    restart_file_pattern: str
    """Glob pattern matching engine restart files."""

    config_file_suffixes: _Tuple[str, ...]
    """Configuration file suffixes generated during setup."""

    clean_file_patterns: _Tuple[str, ...]
    """Glob patterns for engine output removed by `clean`."""

    lighten_file_patterns: _Tuple[str, ...]
    """Glob patterns for engine output removed by `lighten`."""

    @_abstractmethod
    def perturbation_type(self, stage_type: _StageType) -> str:
        """Return the BioSimSpace perturbation type for a stage."""
        pass

    def ignore_setup_warnings(self, ligand_charge: int) -> bool:
        """Whether BioSimSpace setup warnings should be ignored."""
        return False

    @_abstractmethod
    def ensemble_equilibration_output_files(self, leg_type: _LegType) -> _List[str]:
        """Return the files expected from ensemble equilibration."""
        pass

    @_abstractmethod
    def get_equilibrated_data_files(
        self, output_dir: str
    ) -> _Tuple[str, str, _Tuple[str, ...]]:
        """Return file paths and header prefixes for equilibrated data."""
        pass

    def add_alchemical_ions(
        self, system: _Any, ligand: _Any, ligand_charge: int
    ) -> None:
        """Apply engine-specific co-alchemical ion handling."""
        pass

    @_abstractmethod
    def write_run_config(
        self,
        config: _EngineConfig,
        output_dir: str,
        lam: float,
        runtime: float,
    ) -> None:
        """Write the engine configuration for one simulation."""
        pass

    @_abstractmethod
    def get_tot_simtime(self, output_dir: str, config: _EngineConfig) -> float:
        """Return the simulated time in ns."""
        pass

    @_abstractmethod
    def get_tot_gpu_time(self, slurm_output_files: _List[str]) -> float:
        """Return the GPU time in hours."""
        pass

    @_abstractmethod
    def run_completed(self, output_dir: str, slurm_output_files: _List[str]) -> bool:
        """Return whether the simulation completed successfully."""
        pass

    @_abstractmethod
    def read_gradients(
        self,
        output_dir: str,
        config: _EngineConfig,
        equilibrated_only: bool,
        endstate: bool,
    ) -> _Tuple[_np.ndarray, _np.ndarray]:
        """Read simulation times and gradients."""
        pass
