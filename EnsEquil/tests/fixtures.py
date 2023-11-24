"""Fixtures for tests"""

import os
import subprocess
from tempfile import TemporaryDirectory
from typing import Optional

import BioSimSpace.Sandpit.Exscientia as BSS
import pytest
import subprocess

import EnsEquil as ee


@pytest.fixture(scope="session")
def restrain_stage():
    """Create a stage object with analysis data to use in tests"""
    with TemporaryDirectory() as dirname:
        # Copy the input files to the temporary directory
        subprocess.run(["cp", "-r", "EnsEquil/data/example_restraint_stage/", dirname])
        stage = ee.Stage(
            base_dir=os.path.join(dirname, "example_restraint_stage"),
            stage_type=ee.enums.StageType.RESTRAIN,
        )
        # Set the relative simuation cost to 1
        stage.recursively_set_attr("relative_simulation_cost", 1, force=True)
        # Must use yield so that the temporary directory is deleted after the tests
        # by the context manager and does not persist
        yield stage


@pytest.fixture(scope="session")
def restrain_stage_iterator(restrain_stage):
    """Create a simulation runner iterator object with analysis data to use in tests"""
    restrain_stage_iterator = ee.run._utils.SimulationRunnerIterator(
        base_dirs=[restrain_stage.base_dir, restrain_stage.base_dir],
        subclass=ee.Stage,
        stage_type=ee.StageType.RESTRAIN,
    )
    yield restrain_stage_iterator


@pytest.fixture(scope="session")
def restrain_stage_grad_data(restrain_stage):
    """Create a gradient data object with analysis data"""
    yield ee.analyse.GradientData(restrain_stage.lam_windows, equilibrated=True)


@pytest.fixture(scope="session")
def calc():
    """Create a calculation object to use in tests"""
    with TemporaryDirectory() as dirname:
        calc = ee.Calculation(
            base_dir=dirname,
            input_dir="EnsEquil/data/example_run_dir/input",
            ensemble_size=6,
        )
        calc._dump()
        # Must use yield so that the temporary directory is deleted after the tests
        # by the context manager and does not persist
        yield calc


@pytest.fixture(scope="session")
def complex_sys():
    """Create a complex system object to use in tests"""
    base_path = os.path.join("EnsEquil", "data", "example_run_dir", "bound", "input")
    complex_sys = BSS.IO.readMolecules(
        [
            os.path.join(base_path, file)
            for file in ["bound_preequil.prm7", "bound_preequil.rst7"]
        ]
    )
    yield complex_sys
