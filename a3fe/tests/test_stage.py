"""
Unit and regression test for the stage module.
"""

import logging
import os
import subprocess
from tempfile import TemporaryDirectory

import a3fe as a3


def test_dirs_created(engine_config):
    """Check that all expected directories are created"""
    with TemporaryDirectory() as dirname:
        subprocess.run(
            [
                "cp",
                "-r",
                "a3fe/data/example_run_dir/free/discharge/input",
                f"{dirname}/input",
            ]
        )
        config = engine_config(input_dir=f"{dirname}/input")
        config.lambda_values = [0.0, 0.252, 0.593, 1.0]
        # This should create output directories
        a3.Stage(
            stage_type=a3.StageType.DISCHARGE,
            input_dir=f"{dirname}/input",
            base_dir=dirname,
            output_dir=f"{dirname}/output",
            stream_log_level=logging.WARNING,
            engine_config=config,
        )

        lam_dir_names = ["lambda_0.000", "lambda_0.252", "lambda_0.593", "lambda_1.000"]
        run_names = [f"run_0{i}" for i in range(1, 6)]

        for lam_dir in lam_dir_names:
            for run in run_names:
                assert os.path.isdir(os.path.join(dirname, "output", lam_dir, run))
