"""
Unit and regression test for the run module.
"""

# Import package, test suite, and other packages as needed
import sys
import subprocess
import os

import EnsEquil

from tempfile import TemporaryDirectory

def test_EnsEquil_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "EnsEquil" in sys.modules

def test_dirs_created():
    """Check that all expected directories are created"""
    with TemporaryDirectory() as dirname:
        # Store current working directory to change back to later
        cwd = os.getcwd()
        subprocess.run(["cp", "-r", "EnsEquil/data/example_input", f"{dirname}/input"])
        os.chdir(dirname)
        # This should create output directories
        EnsEquil.Stage(stage_type=EnsEquil.StageType.DISCHARGE)

        lam_dir_names = ["lambda_0.000", "lambda_0.250", "lambda_1.000"]
        run_names = [f"run_0{i}" for i in range(1, 6)]

        for lam_dir in lam_dir_names:
            for run in run_names:
                assert os.path.isdir(os.path.join(dirname, "output", lam_dir, run))

        # Change back to original working directory
        os.chdir(cwd)
