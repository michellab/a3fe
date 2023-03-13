""" Unit and regression test for the Calculation class. """

from asyncio import subprocess
from glob import glob
import logging
import os
import pytest
import subprocess
from tempfile import TemporaryDirectory

import EnsEquil as ee

# Create a Calculation object for future tests

@pytest.fixture(scope = "module")
def calc():
    """Create a calculation object to use in tests"""
    with TemporaryDirectory() as dirname:
        calc = ee.Calculation(base_dir=dirname, 
                              input_dir="EnsEquil/data/example_run_dir/input",
                              ensemble_size=6,)
        calc._dump()
        # Must use yield so that the temporary directory is deleted after the tests
        # by the context manager and does not persist
        yield calc

# Load calc and check it has all the required stuff
def test_calculation_loading(calc):
    """Check that the calculation loads correctly"""
    # Check that the calculation has the correct attributes
    assert calc.loaded_from_pickle == False
    assert calc.ensemble_size == 6
    assert calc.input_dir == "EnsEquil/data/example_run_dir/input"
    assert calc.output_dir == os.path.join(calc.base_dir, "output")
    assert calc.setup_complete == False
    assert calc.prep_stage == ee.leg.PreparationStage.PARAMETERISED
    assert calc.stream_log_level == logging.INFO
    # Check that pickle file exists
    assert os.path.exists(os.path.join(calc.base_dir, "Calculation.pkl"))

def test_calculation_logging(calc):
    """Check that the calculation logging is set up correctly"""
    assert calc._logger.name == "Calculation_0"
    assert type(calc._logger.handlers[0]) == logging.FileHandler # type: ignore
    assert calc._logger.handlers[0].baseFilename == os.path.join(calc.base_dir, "Calculation.log") # type: ignore
    assert calc._logger.handlers[0].level == logging.DEBUG # type: ignore
    assert type(calc._logger.handlers[1]) == logging.StreamHandler # type: ignore
    assert calc._logger.handlers[1].level == logging.INFO # type: ignore

    # Try writing to the log and check that it's written to the file
    calc._logger.info("Test message")
    with open(os.path.join(calc.base_dir, "Calculation.log"), "r") as f:
        assert "Test message" in f.read()

def test_calculation_reloading(calc):
    """Check that the calculations can be correctly loaded from a pickle."""
    calc2 = ee.Calculation(base_dir=calc.base_dir, input_dir="EnsEquil/data/example_run_dir/input")
    assert calc2.loaded_from_pickle == True
    assert calc2.ensemble_size == 6
    assert calc2.input_dir == "EnsEquil/data/example_run_dir/input"
    assert calc2.output_dir == os.path.join(calc.base_dir, "output")
    assert calc2.setup_complete == False
    assert calc2.prep_stage == ee.leg.PreparationStage.PARAMETERISED
    assert calc2.stream_log_level == logging.INFO

def test_update_paths(calc):
    """Check that the calculation paths can be updated correctly."""
    with TemporaryDirectory() as new_dir:
        for file in glob(os.path.join(calc.base_dir, "*")):
            subprocess.run(["cp", "-r", file, new_dir])
        calc3 = ee.Calculation(base_dir=new_dir, input_dir="EnsEquil/data/example_run_dir/input")
        assert calc3.loaded_from_pickle == True
        current_dir = os.getcwd()
        calc3.update_paths()
        assert calc3.base_dir == current_dir
        assert calc3._logger.handlers[0].baseFilename == os.path.join(current_dir, "Calculation.log") # type: ignore
