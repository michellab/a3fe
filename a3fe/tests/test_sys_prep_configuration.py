"""Unit and regression tests for the SystemPreparationConfig class."""

from tempfile import TemporaryDirectory

import pytest

from a3fe import LegType, StageType


def test_create_default_config(system_prep_config):
    """Test that the default config is created correctly."""
    config = system_prep_config()
    assert config.ensemble_equilibration_time == 5000


def test_incorrect_config(system_prep_config):
    """Test that an incorrect config raises an error."""
    with pytest.raises(ValueError):
        system_prep_config(ensemble_equilibration_time=-1)


def test_incorrect_config_mod(system_prep_config):
    """Test that modifying the default config incorrectly raises an error."""
    config = system_prep_config()
    with pytest.raises(ValueError):
        config.ensemble_equilibration_time = -1


def test_config_dump_and_load(system_prep_config):
    """Test that the config can be pickled and loaded."""
    with TemporaryDirectory() as dirname:
        config = system_prep_config()
        config.dump(dirname, LegType.FREE)
        config2 = system_prep_config.load(dirname, LegType.FREE)
        assert config == config2


def test_config_tot_simtime(system_prep_config):
    """Test that the total simulation time is calculated correctly."""
    config = system_prep_config()
    assert config.get_tot_simtime(n_runs=5, leg_type=LegType.FREE) == 26855
    assert config.get_tot_simtime(n_runs=5, leg_type=LegType.BOUND) == 26905


def test_required_stages(system_prep_config):
    """Test that the required stages are calculated correctly."""
    config = system_prep_config()
    assert config.required_stages == {
        LegType.FREE: [StageType.DISCHARGE, StageType.VANISH],
        LegType.BOUND: [StageType.RESTRAIN, StageType.DISCHARGE, StageType.VANISH],
    }
