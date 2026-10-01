"""GROMACS simulation engine backend."""

import os as _os
import pathlib as _pathlib
import subprocess as _subprocess
from typing import Any as _Any
from typing import List as _List
from typing import Tuple as _Tuple

import BioSimSpace.Sandpit.Exscientia as _BSS
import numpy as _np
from BioSimSpace.Sandpit.Exscientia.Align._alch_ion import (
    _mark_alchemical_ion,
)

from ..configuration import LegType as _LegType
from ..configuration import StageType as _StageType
from ..configuration import _EngineConfig
from ..read._process_gmx_files import read_xvg_dhdl as _read_xvg_dhdl
from ._engine import EngineBackend as _EngineBackend


class GromacsBackend(_EngineBackend):
    """Stateless GROMACS-specific run behaviour."""

    run_engine = "GROMACS"
    supports_adaptive = False
    coordinate_prefix_and_extension = ("gromacs", "gro")
    required_input_files = ("gromacs.top", "gromacs.gro")
    restart_file_pattern = "**/*.cpt"
    config_file_suffixes = (".mdp", ".tpr")
    clean_file_patterns = ("em/*", "prod/*")
    lighten_file_patterns = (
        "**/*.xtc",
        "**/*.trr",
        "**/*.cpt",
        "**/*_equilibrated.xvg",
    )

    def perturbation_type(self, stage_type: _StageType) -> str:
        return "full"

    def ignore_setup_warnings(self, ligand_charge: int) -> bool:
        return ligand_charge != 0

    def ensemble_equilibration_output_files(self, leg_type: _LegType) -> _List[str]:
        files = ["gromacs.gro"]
        if leg_type == _LegType.BOUND:
            files.append("gromacs.xtc")
        return files

    def get_equilibrated_data_files(
        self, output_dir: str
    ) -> _Tuple[str, str, _Tuple[str, ...]]:
        return (
            _os.path.join(output_dir, "prod", "prod.xvg"),
            _os.path.join(output_dir, "prod", "prod_equilibrated.xvg"),
            ("#", "@"),
        )

    def add_alchemical_ions(
        self, system: _Any, ligand: _Any, ligand_charge: int
    ) -> None:
        ion_charge = -1 if ligand_charge > 0 else 1
        ions = [
            molecule
            for molecule in system
            if molecule.nAtoms() == 1 and round(molecule.charge().value()) == ion_charge
        ]
        if len(ions) < abs(ligand_charge):
            raise ValueError(
                f"Could not find {abs(ligand_charge)} monovalent counterion(s) "
                "for the charged ligand."
            )

        space = system._sire_object.property("space")
        ligand_centre = ligand.getAtoms()[ligand.getCOMIdx()]._sire_object.property(
            "coordinates"
        )
        ions.sort(
            key=lambda ion: space.calc_dist(
                ion.getAtoms()[0]._sire_object.property("coordinates"),
                ligand_centre,
            ),
            reverse=True,
        )

        for ion in ions[: abs(ligand_charge)]:
            perturbed_ion = _BSS.Align.merge(ion, ion, mapping={0: 0})
            cursor = perturbed_ion._sire_object.cursor()
            charge = perturbed_ion.getAtoms()[0]._sire_object.property("charge1")
            cursor[0]["charge1"] = 0 * charge
            perturbed_ion._sire_object = cursor.commit()
            system.updateMolecule(
                system.getIndex(ion), _mark_alchemical_ion(perturbed_ion)
            )

    def write_run_config(
        self,
        config: _EngineConfig,
        output_dir: str,
        lam: float,
        runtime: float,
    ) -> None:
        config.write_all_stage_configs(  # type: ignore
            run_dir=output_dir,
            lambda_val=lam,
            runtime=runtime,
        )

    def get_tot_simtime(self, output_dir: str, config: _EngineConfig) -> float:
        data_file = _pathlib.Path(self.get_equilibrated_data_files(output_dir)[0])
        if not data_file.is_file() or data_file.stat().st_size == 0:
            return 0

        last_line = (
            _subprocess.check_output(["tail", "-1", str(data_file)])
            .decode("utf-8")
            .strip()
        )
        if not last_line or last_line.startswith(("#", "@", "&")):
            return 0
        return float(last_line.split()[0]) / 1000

    def get_tot_gpu_time(self, slurm_output_files: _List[str]) -> float:
        total_seconds = 0.0
        for file in slurm_output_files:
            with open(file, "rt") as f:
                for line in f:
                    if line.strip().startswith("Time:"):
                        try:
                            total_seconds += float(line.split()[2])
                        except (IndexError, ValueError):
                            continue
        return total_seconds / 3600

    def run_completed(self, output_dir: str, slurm_output_files: _List[str]) -> bool:
        prod_log = _pathlib.Path(output_dir, "prod", "prod.log")
        if not prod_log.is_file():
            return False
        with open(prod_log, "rt") as f:
            return "Performance:" in f.read()

    def read_gradients(
        self,
        output_dir: str,
        config: _EngineConfig,
        equilibrated_only: bool,
        endstate: bool,
    ) -> _Tuple[_np.ndarray, _np.ndarray]:
        data_file, equilibrated_file, _ = self.get_equilibrated_data_files(output_dir)
        filename = equilibrated_file if equilibrated_only else data_file
        times, data = _read_xvg_dhdl(filename)

        if endstate:
            final_state_column = 3 + len(config.lambda_values)  # type: ignore
            gradients = data[:, final_state_column] - data[:, 4]
        else:
            lambda_arrays = [
                config.coul_lambdas,  # type: ignore
                config.vdw_lambdas,  # type: ignore
                config.bonded_lambdas,  # type: ignore
            ]
            gradient_columns = [
                index
                for index, lambda_array in enumerate(lambda_arrays, start=1)
                if lambda_array is not None and len(set(lambda_array)) > 1
            ]
            if len(gradient_columns) != 1:
                raise ValueError(
                    "Expected exactly one varying GROMACS lambda component."
                )
            gradients = data[:, gradient_columns[0]]

        # Convert time from ps to ns and gradients from kJ/mol to kcal/mol.
        return times / 1000, gradients / 4.184
