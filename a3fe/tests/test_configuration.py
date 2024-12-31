"""Unit and regression tests for Pydantic configuration classes."""

from tempfile import TemporaryDirectory

import pytest

from a3fe import LegType, SystemPreparationConfig


def test_create_default_config():
    """Test that the default config is created correctly."""
    config = SystemPreparationConfig()
    assert config.ensemble_equilibration_time == 5000


def test_incorrect_config():
    """Test that an incorrect config raises an error."""
    with pytest.raises(ValueError):
        SystemPreparationConfig(ensemble_equilibration_time=-1)


def test_incorrect_config_mod():
    """Test that modifying the default config incorrectly raises an error."""
    config = SystemPreparationConfig()
    with pytest.raises(ValueError):
        config.ensemble_equilibration_time = -1


def test_config_pickle_and_load():
    """Test that the config can be pickled and loaded."""
    with TemporaryDirectory() as dirname:
        config = SystemPreparationConfig()
        config.dump(dirname, LegType.FREE)
        config2 = SystemPreparationConfig.load(dirname, LegType.FREE)
        assert config == config2


def test_config_tot_simtime():
    """Test that the total simulation time is calculated correctly."""
    config = SystemPreparationConfig()
    assert config.get_tot_simtime(n_runs=5, leg_type=LegType.FREE) == 26855
    assert config.get_tot_simtime(n_runs=5, leg_type=LegType.BOUND) == 26905
