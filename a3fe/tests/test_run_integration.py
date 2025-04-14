"""Integration tests for a3fe.

These tests are meant to be run locally,
and are not run on CI due to their long runtime.

This file contains dedicated integration tests that are more granular than
the integration test in test_run.py (which is currently skipped).

To run integration tests, use the following command:

cd your/path/a3fe
RUN_SLURM_TESTS=1 pytest a3fe/tests --run-integration -v

To run a specific integration test:

RUN_SLURM_TESTS=1 pytest a3fe/tests/test_run_integration.py::TestSlurmIntegration::test_slurm_calculation_setup --run-integration -v

See README.md in this directory for more information on running these tests.
"""

import os
import pytest
import subprocess
import glob
from tempfile import TemporaryDirectory
import logging

import a3fe as a3
from a3fe.tests import SLURM_PRESENT, RUN_SLURM_TESTS


# Define the legs and stages for testing
LEGS_WITH_STAGES = {
    "bound": ["discharge", "vanish", "restrain"],
    "free": ["discharge", "vanish"],
}


def _create_example_input_dir(engine_type):
    """Create the example input directory in the temporary directory"""
    with TemporaryDirectory() as temp_dir:
        subprocess.run(
            [
                "cp",
                "-r",
                "a3fe/data/t4l_input",
                f"{temp_dir}/input",
            ]
        )
        calc = a3.Calculation(
            base_dir=temp_dir,
            input_dir=f"{temp_dir}/input",
            ensemble_size=2,
            stream_log_level=logging.CRITICAL,
            engine_type=engine_type,
        )
        calc._dump()
        yield calc


@pytest.fixture(scope="class")
def slurm_calc_non_adaptive(engine_type):
    """Set up a calculation for non-adaptive slurm tests"""
    yield from _create_example_input_dir(engine_type)


@pytest.fixture(scope="class")
def slurm_calc_adaptive(engine_type):
    """Set up a calculation for adaptive slurm tests"""
    yield from _create_example_input_dir(engine_type)


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
            calc.setup(sysprep_config=cfg)

        assert calc.setup_complete
        return calc

    @pytest.mark.integration
    def test_slurm_calculation_setup(self, slurm_calc_non_adaptive, system_prep_config):
        """Test that the SLURM calculation setup works correctly."""

        calc = self._setup_calculation(slurm_calc_non_adaptive, system_prep_config)

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
    def test_slurm_non_adaptive_run(self, slurm_calc_non_adaptive, system_prep_config):
        """
        Full integration test for non-adaptive run using SLURM.
        """
        calc = self._setup_calculation(slurm_calc_non_adaptive, system_prep_config)

        # Run a short non-adaptive calculation
        calc.run(adaptive=False, runtime=0.1)
        calc.wait()

        # Check that the calculation is not running
        assert not calc.running

        # Set the equilibration time for analysis using the proper API
        calc.set_equilibration_time(0)

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
                    assert dg < 0
                    assert dg > -25
                except (IndexError, ValueError) as e:
                    pytest.fail(f"Failed to parse free energy value: {str(e)}")

        except FileNotFoundError as e:
            pytest.fail(f"SLURM job file not found: {str(e)}")

    @pytest.mark.integration
    def test_slurm_optimal_lambda(self, slurm_calc_adaptive, system_prep_config):
        """Test that the optimal lambda window creation function works."""
        calc = self._setup_calculation(slurm_calc_adaptive, system_prep_config)

        calc.get_optimal_lam_vals(delta_er=2)

        # Check that the old runs have been moved, and that the new lambda values are
        # different to the old ones
        for leg in LEGS_WITH_STAGES.keys():
            for stage in LEGS_WITH_STAGES[leg]:
                assert os.path.exists(
                    os.path.join(calc.base_dir, leg, stage, "lam_val_determination")
                )
                lam_dirs_old = glob.glob(
                    os.path.join(
                        calc.base_dir, leg, stage, "lam_val_determination", "lambda_*"
                    )
                )
                lam_dirs_new = glob.glob(
                    os.path.join(calc.base_dir, leg, stage, "output", "lambda_*")
                )
                assert lam_dirs_new != lam_dirs_old

    @pytest.mark.integration
    def test_slurm_adaptive_run(self, slurm_calc_adaptive, system_prep_config):
        """
        Full integration test for adaptive run using SLURM.
        """
        calc = self._setup_calculation(slurm_calc_adaptive, system_prep_config)

        # Use a larger runtime_constant to greatly accelerate the test
        # The standard value is 0.0005
        calc.run(adaptive=True, runtime_constant=0.0005)
        calc.wait()

        # Check that the calculation is not running
        assert not calc.running

        # Check that the equilibration time has successfully been set and matches expected value
        for leg_name in LEGS_WITH_STAGES.keys():
            for stage_name in LEGS_WITH_STAGES[leg_name]:
                # Check that the check_equil_multiwindow_paired_t.txt file exists
                leg_path = os.path.join(calc.base_dir, leg_name)
                stage_path = os.path.join(leg_path, stage_name)
                output_path = os.path.join(stage_path, "output")
                equil_file = os.path.join(
                    output_path, "check_equil_multiwindow_paired_t.txt"
                )

                # Read the fractional equilibration time from the file
                fractional_equil_time = None
                with open(equil_file, "r") as f:
                    for line in f:
                        if "Fractional equilibration time:" in line:
                            fractional_equil_time = float(line.split(":")[1].strip())
                            break

                # Verify all lambda windows
                for leg in calc.legs:
                    if leg_name == os.path.basename(leg.base_dir):
                        for stage in leg.stages:
                            if stage_name == os.path.basename(stage.base_dir):
                                for lam_win in stage.lam_windows:
                                    assert lam_win.equil_time is not None
                                    expected_time = (
                                        fractional_equil_time
                                        * lam_win.get_tot_simtime(run_nos=[1])
                                    )
                                    assert (
                                        abs(lam_win.equil_time - expected_time) < 1e-6
                                    )

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
                    assert dg < 0
                    assert dg > -25
                except (IndexError, ValueError) as e:
                    pytest.fail(f"Failed to parse free energy value: {str(e)}")

        except FileNotFoundError as e:
            pytest.fail(f"SLURM job file not found: {str(e)}")
