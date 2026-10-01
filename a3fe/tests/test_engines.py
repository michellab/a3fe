"""Unit tests for simulation engine backends."""

import numpy as np
import pytest

from a3fe.configuration import EngineType, GromacsConfig, LegType, SomdConfig, StageType
from a3fe.engines import engine_backend_registry


def test_engine_backend_registry_contains_all_engines():
    """Every configured engine has exactly one registered backend."""
    assert set(engine_backend_registry) == set(EngineType)
    assert all(
        backend.run_engine == engine_type.name
        for engine_type, backend in engine_backend_registry.items()
    )


@pytest.mark.parametrize(
    "engine_type, supports_adaptive, suffixes",
    [
        (EngineType.SOMD, True, (".cfg",)),
        (EngineType.GROMACS, False, (".mdp", ".tpr")),
    ],
)
def test_backend_engine_metadata(engine_type, supports_adaptive, suffixes):
    """Backends expose the engine capabilities used by the run layer."""
    backend = engine_backend_registry[engine_type]
    assert backend.supports_adaptive is supports_adaptive
    assert backend.config_file_suffixes == suffixes


def test_perturbation_types():
    """Backends provide the expected perturbation type for each stage."""
    somd_backend = engine_backend_registry[EngineType.SOMD]
    gromacs_backend = engine_backend_registry[EngineType.GROMACS]

    assert [somd_backend.perturbation_type(stage) for stage in StageType] == [
        "restraint",
        "discharge_soft",
        "vanish_soft",
    ]
    assert all(
        gromacs_backend.perturbation_type(stage) == "full" for stage in StageType
    )


def test_gromacs_equilibration_outputs():
    """GROMACS only retains the trajectory for bound-leg equilibration."""
    backend = engine_backend_registry[EngineType.GROMACS]
    assert backend.ensemble_equilibration_output_files(LegType.FREE) == ["gromacs.gro"]
    assert backend.ensemble_equilibration_output_files(LegType.BOUND) == [
        "gromacs.gro",
        "gromacs.xtc",
    ]


def test_gromacs_charge_warning_policy():
    """Charged GROMACS setups suppress BioSimSpace setup warnings."""
    backend = engine_backend_registry[EngineType.GROMACS]
    assert backend.ignore_setup_warnings(0) is False
    assert backend.ignore_setup_warnings(1) is True
    assert backend.ignore_setup_warnings(-1) is True


def test_somd_gradient_unit_conversions(tmp_path):
    """SOMD steps and reduced gradients are converted to ns and kcal/mol."""
    (tmp_path / "simfile.dat").write_text(
        "#Generating temperature is 25 C\n"
        "0 0 1.0 0 0 0.5 2.5\n"
        "1000000 0 2.0 0 0 0.5 3.5\n"
    )

    backend = engine_backend_registry[EngineType.SOMD]
    times, gradients = backend.read_gradients(
        output_dir=str(tmp_path),
        config=SomdConfig(timestep=4.0),
        equilibrated_only=False,
        endstate=False,
    )

    # CODATA molar gas constant in kcal mol^-1 K^-1, independent of Sire.
    gas_constant = 0.00198720425864083
    assert np.allclose(times, [0.0, 4.0])
    assert np.allclose(
        gradients,
        np.array([1.0, 2.0]) * 298.15 * gas_constant,
        rtol=2e-6,
    )

    _, endstate_gradients = backend.read_gradients(
        output_dir=str(tmp_path),
        config=SomdConfig(timestep=4.0),
        equilibrated_only=False,
        endstate=True,
    )
    assert np.allclose(
        endstate_gradients,
        np.array([2.0, 3.0]) * 298.15 * gas_constant,
        rtol=2e-6,
    )


def test_somd_gradients_require_temperature(tmp_path):
    """A malformed SOMD output cannot silently use an unknown temperature."""
    (tmp_path / "simfile.dat").write_text("0 0 1.0\n")

    backend = engine_backend_registry[EngineType.SOMD]
    with pytest.raises(ValueError, match="Could not find the generating temperature"):
        backend.read_gradients(
            output_dir=str(tmp_path),
            config=SomdConfig(),
            equilibrated_only=False,
            endstate=False,
        )


@pytest.mark.parametrize(
    "engine_type, config_type, data_file, contents",
    [
        (EngineType.SOMD, SomdConfig, "simfile.dat", "0 0 0\n50000 0 0\n"),
        (
            EngineType.GROMACS,
            GromacsConfig,
            "prod/prod.xvg",
            '@ title "dhdl"\n0.0 0.0\n200.0 0.0\n',
        ),
    ],
)
def test_backend_simulation_time_ns(
    tmp_path, engine_type, config_type, data_file, contents
):
    """SOMD steps and GROMACS ps both give the same simulation time in ns."""
    backend = engine_backend_registry[engine_type]
    config = config_type()
    assert backend.get_tot_simtime(str(tmp_path), config) == 0

    file = tmp_path / data_file
    file.parent.mkdir(exist_ok=True)
    file.write_text(contents)
    assert backend.get_tot_simtime(str(tmp_path), config) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "engine_type, timing_line",
    [
        (EngineType.SOMD, "Simulation took 1800 s\n"),
        (EngineType.GROMACS, "Time: 3600.000 1800.000 1000.0\n"),
    ],
)
def test_backend_gpu_time_hours(tmp_path, engine_type, timing_line):
    """Engine timing logs report GPU time in hours."""
    log_file = tmp_path / "run.log"
    log_file.write_text(timing_line)

    backend = engine_backend_registry[engine_type]
    assert backend.get_tot_gpu_time([str(log_file)]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "engine_type, log_path, completion_line",
    [
        (EngineType.SOMD, "run.log", "Simulation took 1800 s\n"),
        (EngineType.GROMACS, "prod/prod.log", "Performance: 100 ns/day\n"),
    ],
)
def test_backend_run_completed(tmp_path, engine_type, log_path, completion_line):
    """Engine completion requires its own final log marker."""
    file = tmp_path / log_path
    file.parent.mkdir(exist_ok=True)
    file.write_text("Run started\n")

    backend = engine_backend_registry[engine_type]
    assert not backend.run_completed(str(tmp_path), [str(file)])

    file.write_text(completion_line)
    assert backend.run_completed(str(tmp_path), [str(file)])


def test_gromacs_endstate_gradient(tmp_path):
    """Endstate gradients use the final and initial delta H columns."""
    prod_dir = tmp_path / "prod"
    prod_dir.mkdir()
    (prod_dir / "prod.xvg").write_text(
        '@ title "dhdl"\n200.0 100.0 4.184 8.368 12.552 10.0 18.368\n'
    )

    backend = engine_backend_registry[EngineType.GROMACS]
    times, gradients = backend.read_gradients(
        output_dir=str(tmp_path),
        config=GromacsConfig(lambda_values=[0.0, 1.0]),
        equilibrated_only=False,
        endstate=True,
    )

    assert np.allclose(times, [0.2])
    assert np.allclose(gradients, [2.0])
