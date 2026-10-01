"""SOMD simulation engine backend."""

import os as _os
import pathlib as _pathlib
import subprocess as _subprocess
from typing import List as _List
from typing import Tuple as _Tuple

import numpy as _np
from sire.units import k_boltz as _k_boltz

from ..configuration import LegType as _LegType
from ..configuration import StageType as _StageType
from ..configuration import _EngineConfig
from ._engine import EngineBackend as _EngineBackend


class SomdBackend(_EngineBackend):
    """Stateless SOMD-specific run behaviour."""

    run_engine = "SOMD"
    supports_adaptive = True
    coordinate_prefix_and_extension = ("somd", "rst7")
    required_input_files = ("somd.prm7", "somd.rst7", "somd.pert")
    restart_file_pattern = "*.s3"
    config_file_suffixes = (".cfg",)
    clean_file_patterns = (
        "*.dcd",
        "moves.dat",
        "simfile.dat",
        "simfile_equilibrated.dat",
        "*.s3",
        "*.s3.previous",
        "latest.pdb",
        "gradients.dat",
        "equilibration_block_gradient.txt",
        "equilibration_shrinking_block_gradient",
    )
    lighten_file_patterns = (
        "*.dcd",
        "*.s3",
        "*.s3.previous",
        "gradients.s3",
        "simfile_equilibrated.dat",
        "latest.pdb",
    )

    def perturbation_type(self, stage_type: _StageType) -> str:
        return stage_type.bss_perturbation_type

    def ensemble_equilibration_output_files(self, leg_type: _LegType) -> _List[str]:
        return ["somd.rst7"]

    def get_equilibrated_data_files(
        self, output_dir: str
    ) -> _Tuple[str, str, _Tuple[str, ...]]:
        return (
            _os.path.join(output_dir, "simfile.dat"),
            _os.path.join(output_dir, "simfile_equilibrated.dat"),
            ("#",),
        )

    def write_run_config(
        self,
        config: _EngineConfig,
        output_dir: str,
        lam: float,
        runtime: float,
    ) -> None:
        config.write_config(
            run_dir=output_dir,
            lambda_val=lam,
            runtime=runtime,
            top_file="somd.prm7",
            coord_file="somd.rst7",
            morph_file="somd.pert",
        )

    def get_tot_simtime(self, output_dir: str, config: _EngineConfig) -> float:
        data_file = _pathlib.Path(self.get_equilibrated_data_files(output_dir)[0])
        if not data_file.is_file() or data_file.stat().st_size == 0:
            return 0

        step = int(
            _subprocess.check_output(["tail", "-1", str(data_file)])
            .decode("utf-8")
            .strip()
            .split()[0]
        )
        return step * (config.timestep / 1_000_000)  # type: ignore

    def get_tot_gpu_time(self, slurm_output_files: _List[str]) -> float:
        total_seconds = 0.0
        for file in slurm_output_files:
            with open(file, "rt") as f:
                for line in f:
                    if line.startswith("Simulation took"):
                        total_seconds += float(line.split()[2])
        return total_seconds / 3600

    def run_completed(self, output_dir: str, slurm_output_files: _List[str]) -> bool:
        for file in slurm_output_files:
            with open(file, "rt") as f:
                if "Simulation took" not in f.read():
                    return False
        return True

    def read_gradients(
        self,
        output_dir: str,
        config: _EngineConfig,
        equilibrated_only: bool,
        endstate: bool,
    ) -> _Tuple[_np.ndarray, _np.ndarray]:
        data_file, equilibrated_file, _ = self.get_equilibrated_data_files(output_dir)
        filename = equilibrated_file if equilibrated_only else data_file
        with open(filename, "r") as ifile:
            lines = ifile.readlines()

        steps = []
        gradients = []
        temperature = None
        for line in lines:
            values = line.split()
            if line.startswith("#Generating temperature is"):
                temperature = values[3]
                try:
                    unit = values[4]
                except IndexError:
                    temperature, unit = temperature.split("°")
                if unit == "C":
                    temperature = float(temperature) + 273.15
                else:
                    temperature = float(temperature)
            if not line.startswith("#"):
                steps.append(int(values[0].strip()))
                if endstate:
                    gradient = float(values[-1]) - float(values[5])
                else:
                    gradient = float(values[2])
                gradients.append(gradient)

        if temperature is None:
            raise ValueError(
                f"Could not find the generating temperature in {filename}."
            )

        # Convert steps to ns using the fs timestep, then reduced gradients to
        # kcal/mol using k_B T (k_B is expressed in kcal mol^-1 K^-1).
        times = _np.array(steps) * (config.timestep / 1_000_000)  # type: ignore
        gradients_array = _np.array(gradients)
        gradients_array *= temperature * _k_boltz.value()
        return times, gradients_array
