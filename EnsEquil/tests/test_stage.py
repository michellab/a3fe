"""
Unit and regression test for the run module.
"""

# Import package, test suite, and other packages as needed
import logging
import os
import sys
import subprocess

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
        # This should create output directories
        EnsEquil.Stage(stage_type=EnsEquil.StageType.DISCHARGE, 
                       input_dir=f"{dirname}/input",
                       base_dir=dirname,
                       output_dir=f"{dirname}/output",
                       stream_log_level=logging.WARNING)

        lam_dir_names = ["lambda_0.000", "lambda_0.250", "lambda_1.000"]
        run_names = [f"run_0{i}" for i in range(1, 6)]

        for lam_dir in lam_dir_names:
            for run in run_names:
                assert os.path.isdir(os.path.join(dirname, lam_dir, run))

