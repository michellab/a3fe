"""Fixtures for tests"""

import os
import pickle as pkl
import subprocess
from tempfile import TemporaryDirectory

import BioSimSpace.Sandpit.Exscientia as BSS
import pytest

import a3fe as a3


@pytest.fixture(scope="session")
def restrain_stage():
    """Create a stage object with analysis data to use in tests"""
    with TemporaryDirectory() as dirname:
        # Copy the input files to the temporary directory
        subprocess.run(
            ["cp", "-r", "a3fe/data/example_restraint_stage/", dirname], check=True
        )
        stage = a3.Stage(
            base_dir=os.path.join(dirname, "example_restraint_stage"),
            stage_type=a3.enums.StageType.RESTRAIN,
        )
        # Set the relative simuation cost to 1
        stage.recursively_set_attr("relative_simulation_cost", 1, force=True)
        # Ensure the tests don't try to use slurm
        stage.recursively_set_attr("slurm_equil_detection", False, force=True)
        # Must use yield so that the temporary directory is deleted after the tests
        # by the context manager and does not persist
        yield stage


@pytest.fixture(scope="session")
def calc_set():
    """Create a calculation set object to use in tests"""
    with TemporaryDirectory() as dirname:
        # Copy input files to the temporary directory
        subprocess.run(["cp", "-r", "a3fe/data/example_calc_set/", dirname], check=True)
        base_dir = os.path.join(dirname, "example_calc_set")
        calc_paths = [
            os.path.join(base_dir, name) for name in ["mdm2_pip2_short", "t4l"]
        ]
        calc_set = a3.CalcSet(
            base_dir=base_dir,
            calc_paths=calc_paths,
        )
        yield calc_set


@pytest.fixture(scope="session")
def restrain_stage_iterator(restrain_stage):
    """Create a simulation runner iterator object with analysis data to use in tests"""
    restrain_stage_iterator = a3.run._utils.SimulationRunnerIterator(
        base_dirs=[restrain_stage.base_dir, restrain_stage.base_dir],
        subclass=a3.Stage,
        stage_type=a3.StageType.RESTRAIN,
    )
    yield restrain_stage_iterator


@pytest.fixture(scope="session")
def restrain_stage_grad_data(restrain_stage):
    """Create a gradient data object with analysis data"""
    yield a3.analyse.GradientData(restrain_stage.lam_windows, equilibrated=True)


@pytest.fixture(scope="session")
def calc():
    """Create a calculation object to use in tests"""
    with TemporaryDirectory() as dirname:
        calc = a3.Calculation(
            base_dir=dirname,
            input_dir="a3fe/data/example_run_dir/input",
            ensemble_size=6,
        )
        calc._dump()
        # Must use yield so that the temporary directory is deleted after the tests
        # by the context manager and does not persist
        yield calc


@pytest.fixture(scope="session")
def t4l_calc():
    """
    Create a calculation using the quickly-parametrised T4L system.
    The preparation stage is STRUCTURES_ONLY, and this is used for
    testing parameterisation.
    """
    with TemporaryDirectory() as dirname:
        # Copy T4L structure files
        subprocess.run(
            ["cp", "-r", "a3fe/data/t4l_input", os.path.join(dirname, "input")],
            check=True,
        )

        # Copy over remaining input files
        for file in ["run_somd.sh", "template_config.cfg"]:
            subprocess.run(
                ["cp", os.path.join("a3fe/data/example_run_dir/input/", file), os.path.join(dirname, "input")],
                check=True,
            )

        calc = a3.Calculation(
            base_dir=dirname,
        )
        calc._dump()

        yield calc



@pytest.fixture(scope="session")
def complex_sys():
    """Create a complex system object to use in tests"""
    base_path = os.path.join("a3fe", "data", "example_run_dir", "bound", "input")
    complex_sys = BSS.IO.readMolecules(
        [
            os.path.join(base_path, file)
            for file in ["bound_preequil.prm7", "bound_preequil.rst7"]
        ]
    )
    yield complex_sys


@pytest.fixture(scope="session")
def a3fe_restraint():
    """Create an A3FE restraint object to use in tests"""
    with open("a3fe/data/example_run_dir/input/restraint.pkl", "rb") as f:
        a3fe_restraint = pkl.load(f)
    yield a3fe_restraint
