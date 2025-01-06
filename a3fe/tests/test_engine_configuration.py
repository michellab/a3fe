"""Unit and regression tests for the SomdConfig class."""

from tempfile import TemporaryDirectory

from a3fe import SomdConfig

import os


def test_create_config():
    """Test that the config can be created."""
    config = SomdConfig()
    assert isinstance(config, SomdConfig)


def test_config_pickle_and_load():
    """Test that the config can be pickled and loaded."""
    with TemporaryDirectory() as dirname:
        config = SomdConfig()
        config.dump(dirname)
        config2 = SomdConfig.load(dirname)
        assert config == config2


def test_get_somd_config():
    """
    Test that the SOMD configuration file is generated correctly
    and that the file is written correctly.
    """
    # Tmpdir to store the config
    with TemporaryDirectory() as dirname:
        config = SomdConfig(
            integrator="langevinmiddle",
            nmoves=25000,
            ncycles=60,
            timestep=4.0,
            cutoff_type="PME",
            cutoff_distance=10.0,
        )
        config_path = config.get_somd_config(
            run_dir=dirname,
            config_name="test"
        )
        assert config_path == os.path.join(dirname, "test.cfg")

        expected_config = (
            "### Integrator ###\n"
            "nmoves = 25000\n"
            "ncycles = 60\n"
            "timestep = 4.0 * femtosecond\n"
            "constraint = hbonds\n"
            "hydrogen mass repartitioning factor = 3.0\n"
            "integrator = langevinmiddle\n"
            "inverse friction = 1.0 * picosecond\n"
            "temperature = 25.0 * celsius\n"
            "thermostat = False\n"
            "\n"
            "### Barostat ###\n"
            "barostat = True\n"
            "pressure = 1.0 * atm\n"
            "\n"
            "### Non-Bonded Interactions ###\n"
            "cutoff type = PME\n"
            "cutoff distance = 10.0 * angstrom\n"
            "\n"
            "### Trajectory ###\n"
            "buffered coordinates frequency = 500\n"
            "center solute = True\n"
            "\n"
            "### Minimisation ###\n"
            "minimise = True\n"
            "\n"
            "### Alchemistry ###\n"
            "perturbed residue number = 1\n"
            "energy frequency = 500\n"
        )

        with open(config_path, "r") as f:
            config_content = f.read()

        assert config_content == expected_config
