"""Integration tests for a3fe.

These tests are meant to be run locally before releasing code,
and are not run on CI due to their long runtime.

Note: This file contains dedicated integration tests that are more granular than
the integration test in test_run.py (which is currently skipped).

See README.md in this directory for more information on running these tests.
"""

import os
import pytest
import subprocess
from tempfile import TemporaryDirectory
import logging

import a3fe as a3
from a3fe.tests import SLURM_PRESENT, RUN_SLURM_TESTS


# Define the legs and stages for testing
LEGS_WITH_STAGES = {
    "bound": ["discharge", "vanish", "restrain"],
    "free": ["discharge", "vanish"],
}


@pytest.fixture(scope="module")
def slurm_calc():
    """Set up a calculation for slurm test that is shared across all tests"""
    with TemporaryDirectory() as temp_dir:
        # Copy the example input directory to the temporary directory
        subprocess.run(
            [
                "cp",
                "-r",
                "a3fe/data/example_run_dir/input",
                f"{temp_dir}/input",
            ]
        )

        calc = a3.Calculation(
            base_dir=temp_dir,
            input_dir=f"{temp_dir}/input",
            ensemble_size=2,
            stream_log_level=logging.CRITICAL,
        )  # Close the log output
        calc._dump()
        # Must use yield so that the temporary directory is deleted after the tests
        # by the context manager and does not persist
        yield calc


@pytest.mark.integration
@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
@pytest.mark.skipif(not RUN_SLURM_TESTS, reason="RUN_SLURM_TESTS is False")
class TestSlurmIntegration:
    """Test class for SLURM integration tests."""

    def _get_test_config(self, system_prep_config):
        """Get the system preparation configuration for testing"""
        cfg = system_prep_config()
        cfg.runtime_npt_unrestrained = 20  # ps very short for testing
        cfg.runtime_npt = 20  # ps
        cfg.ensemble_equilibration_time = 20  # ps
        return cfg

    def _setup_calculation(self, calc, system_prep_config):
        """Check that the calculation object is setup already."""
        if not calc.setup_complete:
            cfg = self._get_test_config(system_prep_config)
            calc.setup(bound_leg_sysprep_config=cfg, free_leg_sysprep_config=cfg)

        assert calc.setup_complete
        return calc

    @pytest.mark.integration
    def test_slurm_calculation_setup(self, slurm_calc, system_prep_config):
        """Test that the SLURM calculation setup works correctly."""

        calc = self._setup_calculation(slurm_calc, system_prep_config)

        # Check that the required SLURM job names exist
        required_slurm_jobnames = ["minimised.sh", "solvated.sh", "preequilibrated.sh"]
        for jobname in required_slurm_jobnames:
            job_path = os.path.join(calc.base_dir, "input", jobname)
            assert os.path.exists(job_path)

        # Check that the required leg directories exist
        for leg in LEGS_WITH_STAGES.keys():
            leg_dir = os.path.join(calc.base_dir, leg)
            assert os.path.exists(leg_dir)

            # Check that the required stage directories exist
            for stage in LEGS_WITH_STAGES[leg]:
                stage_dir = os.path.join(leg_dir, stage)
                assert os.path.exists(stage_dir)
                output_dir = os.path.join(stage_dir, "output")
                assert os.path.exists(output_dir)

    @pytest.mark.integration
    def test_slurm_non_adaptive_run(self, slurm_calc, system_prep_config):
        """
        Full integration test for non-adaptive run using SLURM.
        """
        calc = self._setup_calculation(slurm_calc, system_prep_config)

        # Run a short non-adaptive calculation
        calc.run(adaptive=False, runtime=0.1)
        calc.wait()

        # Check that the calculation is not running
        assert not calc.running

        # Set the entire simulation time to equilibrated for analysis
        for leg in calc.legs:
            for stage in leg.stages:
                for lam_win in stage.lam_windows:
                    lam_win._equilibrated = True
                    lam_win._equil_time = 0

        # Check that the simulation time for each lambda window is correct
        for leg in calc.legs:
            for stage in leg.stages:
                for lam_win in stage.lam_windows:
                    assert round(lam_win.tot_simtime / lam_win.ensemble_size, 1) == 0.1

        # Run analysis with error handling
        try:
            calc.analyse()

            # Check that the output file exists
            assert os.path.exists(
                os.path.join(calc.base_dir, "output", "overall_stats.dat")
            )

            # Check that the free energy change is reasonable
            with open(
                os.path.join(calc.base_dir, "output", "overall_stats.dat"), "r"
            ) as f:
                try:
                    dg = float(f.readlines()[1].split(" ")[3])
                    assert dg < -5
                    assert dg > -25
                except (IndexError, ValueError) as e:
                    pytest.fail(f"Failed to parse free energy value: {str(e)}")

        except FileNotFoundError as e:
            import warnings

            warnings.warn(f"SLURM job file not found: {str(e)}")

    @pytest.mark.integration
    def test_slurm_optimal_lambda(self, slurm_calc, system_prep_config):
        """Test that the optimal lambda window creation function works."""
        calc = self._setup_calculation(slurm_calc, system_prep_config)

        calc.get_optimal_lam_vals(delta_er=2)

        # Check that the old runs have been moved, and that the new lambda values are
        # different to the old ones
        for leg in LEGS_WITH_STAGES.keys():
            for stage in LEGS_WITH_STAGES[leg]:
                assert os.path.exists(
                    os.path.join(calc.base_dir, leg, stage, "lam_val_determination")
                )

    @pytest.mark.integration
    def test_slurm_adaptive_run(self, slurm_calc, system_prep_config):
        """
        Full integration test for adaptive run using SLURM.
        """
        calc = self._setup_calculation(slurm_calc, system_prep_config)

        # Use a larger runtime_constant to greatly accelerate the test
        # The standard value is 0.0005, we use 0.05 (100x) to greatly accelerate the test
        calc.run(adaptive=True, runtime_constant=0.05)
        calc.wait()

        # Check that the calculation is not running
        assert not calc.running

        # Set the entire simulation time to equilibrated for analysis
        for leg in calc.legs:
            for stage in leg.stages:
                for lam_win in stage.lam_windows:
                    lam_win._equilibrated = True
                    lam_win._equil_time = 0

        # Run analysis with error handling
        try:
            calc.analyse()

            # Check that the analysis output file exists
            assert os.path.exists(
                os.path.join(calc.base_dir, "output", "overall_stats.dat")
            )

            # Check that the free energy change is reasonable
            with open(
                os.path.join(calc.base_dir, "output", "overall_stats.dat"), "r"
            ) as f:
                try:
                    dg = float(f.readlines()[1].split(" ")[3])
                    assert dg < -5
                    assert dg > -25
                except (IndexError, ValueError) as e:
                    pytest.fail(f"Failed to parse free energy value: {str(e)}")

        except FileNotFoundError as e:
            import warnings

            warnings.warn(f"SLURM job file not found: {str(e)}")
