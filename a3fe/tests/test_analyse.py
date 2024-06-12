"""Testing the analysis module."""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from ..analyse.compare import get_comparitive_convergence_data
from ..analyse.detect_equil import (
    check_equil_block_gradient,
    check_equil_chodera,
    check_equil_multiwindow_gelman_rubin,
    check_equil_multiwindow_kpss,
    check_equil_multiwindow_modified_geweke,
    check_equil_multiwindow_paired_t,
)
from ..analyse.process_grads import get_time_series_multiwindow
from . import RUN_SLURM_TESTS, SLURM_PRESENT

EXPECTED_CONVERGENCE_RESULTS = np.array(
    [
        [
            1.811328,
            1.79284,
            1.686816,
            1.645066,
            1.603964,
            1.560607,
            1.560784,
            1.591202,
            1.579906,
            1.595907,
            1.590091,
            1.597802,
            1.621957,
            1.625856,
            1.62638,
            1.626285,
            1.627279,
            1.636536,
            1.631825,
            1.624827,
        ],
        [
            1.80024,
            1.720471,
            1.669383,
            1.679446,
            1.651117,
            1.667022,
            1.695923,
            1.717015,
            1.742829,
            1.745102,
            1.775038,
            1.769987,
            1.764087,
            1.776267,
            1.787469,
            1.802154,
            1.802192,
            1.805146,
            1.805154,
            1.803477,
        ],
        [
            1.465609,
            1.347802,
            1.370879,
            1.393137,
            1.361814,
            1.361366,
            1.370421,
            1.385677,
            1.391987,
            1.386363,
            1.404679,
            1.407778,
            1.416555,
            1.412339,
            1.416453,
            1.411892,
            1.4214,
            1.426488,
            1.429623,
            1.426475,
        ],
        [
            1.342867,
            1.367449,
            1.358504,
            1.407188,
            1.431588,
            1.435713,
            1.437941,
            1.428834,
            1.424351,
            1.412054,
            1.405822,
            1.403497,
            1.402927,
            1.411375,
            1.414117,
            1.40972,
            1.414707,
            1.413852,
            1.413605,
            1.41811,
        ],
        [
            1.561257,
            1.618,
            1.640034,
            1.620545,
            1.625897,
            1.608938,
            1.632432,
            1.679825,
            1.697756,
            1.714111,
            1.715355,
            1.712736,
            1.722261,
            1.715033,
            1.700635,
            1.712187,
            1.711371,
            1.713132,
            1.717778,
            1.716112,
        ],
    ]
)


def test_analysis_all_runs(restrain_stage):
    """Check that the analysis works on all runs."""
    res, err = restrain_stage.analyse()
    assert res.mean() == pytest.approx(1.5978, abs=1e-2)
    assert err.mean() == pytest.approx(0.0254, abs=1e-3)


def test_analysis_all_runs_fraction(restrain_stage):
    """Check that the analysis works on all runs."""
    res, err = restrain_stage.analyse(fraction=0.5)
    assert res.mean() == pytest.approx(1.5707, abs=1e-2)
    assert err.mean() == pytest.approx(0.0351, abs=1e-3)


def test_get_results_df(restrain_stage):
    """Check that the results dataframe is correctly generated."""
    # Re-analyse to ensure that the order of the tests doesn't matter
    res, err = restrain_stage.analyse()
    df = restrain_stage.get_results_df()
    # Check that the csv has been output
    assert os.path.exists(os.path.join(restrain_stage.output_dir, "results.csv"))
    # Check that the results are correct
    assert df.loc["restrain_stage"]["dg / kcal mol-1"] == pytest.approx(1.6, abs=1e-1)
    assert df.loc["restrain_stage"]["dg_95_ci / kcal mol-1"] == pytest.approx(
        0.21, abs=1e-2
    )
    assert df.loc["restrain_stage"]["tot_simtime / ns"] == pytest.approx(6.0, abs=1e-1)
    assert df.loc["restrain_stage"]["tot_gpu_time / GPU hours"] == pytest.approx(
        1, abs=1e-0
    )


def test_analysis_subselection_runs(restrain_stage):
    """Check that the analysis works on a subselection of runs."""
    res, err = restrain_stage.analyse(run_nos=[1, 2, 4])
    assert res.mean() == pytest.approx(1.6154, abs=1e-2)
    assert err.mean() == pytest.approx(0.0257, abs=1e-3)


def test_convergence_analysis(restrain_stage):
    """Test the convergence analysis."""
    stage = restrain_stage
    _, free_energies = stage.analyse_convergence()
    assert np.allclose(free_energies, EXPECTED_CONVERGENCE_RESULTS, atol=1e-2)


def test_get_time_series_multiwindow(restrain_stage):
    """Check that the time series are correctly extracted/ combined."""
    # Check that this fails if we haven't set equil times
    overall_dgs, overall_times = get_time_series_multiwindow(
        lambda_windows=restrain_stage.lam_windows,
        equilibrated=True,
        run_nos=[1, 2],
    )

    # Check that the output has the correct shape
    assert overall_dgs.shape == (2, 100)
    assert overall_times.shape == (2, 100)

    # Check that the total time is what we expect
    tot_simtime = restrain_stage.get_tot_simtime(run_nos=[1])
    assert overall_times[0][-1] == pytest.approx(tot_simtime, abs=1e-2)

    # Check that the output values are correct
    assert overall_dgs.mean(axis=0)[-1] == pytest.approx(1.7751, abs=1e-2)
    assert overall_times.sum(axis=0)[-1] == pytest.approx(2.4, abs=1e-2)


def test_get_time_series_multiwindow_mbar(restrain_stage):
    """Check that the time series are correctly extracted/ combined."""
    # Check that this fails if we haven't set equil times
    overall_dgs, overall_times = get_time_series_multiwindow(
        lambda_windows=restrain_stage.lam_windows,
        equilibrated=True,
        run_nos=[1, 2],
    )

    # Check that the output has the correct shape
    assert overall_dgs.shape == (2, 100)
    assert overall_times.shape == (2, 100)

    # Check that the total time is what we expect
    tot_simtime = restrain_stage.get_tot_simtime(run_nos=[1])
    assert overall_times[0][-1] == pytest.approx(tot_simtime, abs=1e-2)

    # Check that the output values are correct
    assert overall_dgs.mean(axis=0)[-1] == pytest.approx(1.7751, abs=1e-2)
    assert overall_times.sum(axis=0)[-1] == pytest.approx(2.4, abs=1e-2)


# Parameterise with a dictionary of equilibration functions and expected results
@pytest.mark.parametrize(
    "equil_func, expected",
    [
        (check_equil_block_gradient, 0.0024),
        (check_equil_chodera, 0.0567999),
    ],
)
def test_per_window_equilibration_detection(restrain_stage, equil_func, expected):
    """Regression tests for equilibration detection methods working on a single window."""
    with TemporaryDirectory() as tmpdir:
        lam_win = restrain_stage.lam_windows[3]
        lam_win.output_dir = tmpdir
        lam_win.block_size = 0.05
        equilibrated, fractional_equil_time = equil_func(lam_win=lam_win, run_nos=[1])
        assert equilibrated
        assert fractional_equil_time == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize(
    "equil_func, expected, args",
    [
        (check_equil_multiwindow_kpss, 0.3, {}),
        (
            check_equil_multiwindow_modified_geweke,
            0.0048,
            {"intervals": 10, "p_cutoff": 0.4},
        ),
    ],
)
def test_equil_multiwindow(restrain_stage, equil_func, expected, args):
    """Test the multiwindow equilibration analysis."""
    with TemporaryDirectory() as tmpdir:
        equilibrated, fractional_equil_time = equil_func(
            lambda_windows=restrain_stage.lam_windows,
            output_dir=tmpdir,
            **args,
        )

        assert equilibrated
        assert fractional_equil_time == pytest.approx(expected, abs=1e-2)


def test_paired_t(restrain_stage):
    """Test the paired t-test equilibration analysis."""
    (
        equilibrated,
        fractional_equil_time,
    ) = check_equil_multiwindow_paired_t(
        lambda_windows=restrain_stage.lam_windows,
        output_dir=restrain_stage.output_dir,
        intervals=10,
        p_cutoff=0.05,
    )

    assert equilibrated
    assert fractional_equil_time == pytest.approx(0.0048, abs=1e-2)


def test_gelman_rubin(restrain_stage):
    """Test the Gelman-Rubin convergence analysis."""
    with TemporaryDirectory() as tmpdir:
        rhat_dict = check_equil_multiwindow_gelman_rubin(
            lambda_windows=restrain_stage.lam_windows,
            output_dir=tmpdir,
        )

        expected_rhat_dict = {
            0.0: 1.0496660104040842,
            0.125: 1.0122689789813877,
            0.25: 1.0129155249894615,
            0.375: 1.0088598498180925,
            0.5: 1.020819039702674,
            1.0: 1.0095474751197715,
        }
        assert rhat_dict == expected_rhat_dict


def test_get_comparitive_convergence_data_cumulative(restrain_stage_iterator):
    """Test the get_comparitive_convergence_data function."""
    results = get_comparitive_convergence_data(restrain_stage_iterator)
    times1 = results[0][0]
    times2 = results[1][0]
    assert times1.shape == (20,)
    # Check the arrays are the same with all
    assert np.array_equal(times1, times2)
    assert times1[-1] == pytest.approx(6.0, abs=1e-2)
    dgs1 = results[0][1]
    dgs2 = results[1][1]
    assert dgs1.shape == (5, 20)
    assert np.array_equal(dgs1, dgs2)
    assert dgs1[-1][-1] == pytest.approx(1.716112, abs=1e-2)


def test_get_comparitive_convergence_data_block(restrain_stage_iterator):
    """Test the get_comparitive_convergence_data function."""
    results = get_comparitive_convergence_data(restrain_stage_iterator, mode="block")
    times1 = results[0][0]
    times2 = results[1][0]
    assert times1.shape == (20,)
    # Check the arrays are the same with all
    assert np.array_equal(times1, times2)
    assert times1[-1] == pytest.approx(6.0, abs=1e-2)
    dgs1 = results[0][1]
    dgs2 = results[1][1]
    assert dgs1.shape == (5, 20)
    assert np.array_equal(dgs1, dgs2)
    assert dgs1[-1][-1] == pytest.approx(1.674398, abs=1e-2)


def test_predicted_improvement_factor(restrain_stage_grad_data):
    """
    Test the predicted improvement factor calculation,
    and also check that the optimal lam vals calulcation
    satisfies this.
    """
    grad_data = restrain_stage_grad_data
    # Test with SEM
    improvement_sem_20 = grad_data.get_predicted_improvement_factor(
        initial_lam_vals=grad_data.calculate_optimal_lam_vals(
            n_lam_vals=20, er_type="sem", smoothen_sems=False, round_lams=False
        ),
        er_type="sem",
    )
    assert improvement_sem_20 > 0.98
    improvement_sem_100 = grad_data.get_predicted_improvement_factor(
        initial_lam_vals=grad_data.calculate_optimal_lam_vals(
            n_lam_vals=100, er_type="sem", smoothen_sems=False, round_lams=False
        ),
        er_type="sem",
    )
    assert improvement_sem_100 == pytest.approx(1.0, abs=1e-2)
    # Test with SD
    improvement_sd_20 = grad_data.get_predicted_improvement_factor(
        initial_lam_vals=grad_data.calculate_optimal_lam_vals(
            n_lam_vals=20, er_type="root_var", round_lams=False
        ),
        er_type="root_var",
    )
    assert improvement_sd_20 > 0.98
    improvement_sd_100 = grad_data.get_predicted_improvement_factor(
        initial_lam_vals=grad_data.calculate_optimal_lam_vals(
            n_lam_vals=100, er_type="root_var", round_lams=False
        ),
        er_type="root_var",
    )
    assert improvement_sd_100 == pytest.approx(1.0, abs=1e-2)


##################### Tests Requiring Slurm #####################


@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
@pytest.mark.skipif(not RUN_SLURM_TESTS, reason="RUN_SLURM_TESTS is False")
def test_analysis_all_runs_slurm(restrain_stage):
    """Check that the analysis works on all runs."""
    res, err = restrain_stage.analyse(slurm=True)
    assert res.mean() == pytest.approx(1.5978, abs=1e-2)
    assert err.mean() == pytest.approx(0.0254, abs=1e-3)


@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
@pytest.mark.skipif(not RUN_SLURM_TESTS, reason="RUN_SLURM_TESTS is False")
def test_analysis_all_runs_fraction_slurm(restrain_stage):
    """Check that the analysis works on all runs."""
    res, err = restrain_stage.analyse(fraction=0.5, slurm=True)
    assert res.mean() == pytest.approx(1.5707, abs=1e-2)
    assert err.mean() == pytest.approx(0.0351, abs=1e-3)


@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
@pytest.mark.skipif(not RUN_SLURM_TESTS, reason="RUN_SLURM_TESTS is False")
def test_analysis_subselection_runs_slurm(restrain_stage):
    """Check that the analysis works on a subselection of runs."""
    res, err = restrain_stage.analyse(run_nos=[1, 2, 4], slurm=True)
    assert res.mean() == pytest.approx(1.6154, abs=1e-2)
    assert err.mean() == pytest.approx(0.0257, abs=1e-3)


@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
@pytest.mark.skipif(not RUN_SLURM_TESTS, reason="RUN_SLURM_TESTS is False")
def test_convergence_analysis_slurm(restrain_stage):
    """Test the convergence analysis."""
    stage = restrain_stage
    _, free_energies = stage.analyse_convergence(slurm=True)
    assert np.allclose(free_energies, EXPECTED_CONVERGENCE_RESULTS, atol=1e-2)


@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
@pytest.mark.skipif(not RUN_SLURM_TESTS, reason="RUN_SLURM_TESTS is False")
def test_get_time_series_multiwindow_mbar_slurm(restrain_stage):
    """Check that the time series are correctly extracted/ combined."""
    try:
        # Set the "slurm_equil_detection" attribute on the lambda windows
        # so that slurm is used for the MBAR analysis
        restrain_stage.recursively_set_attr("slurm_equil_detection", True, force=True)

        # Check that this fails if we haven't set equil times
        overall_dgs, overall_times = get_time_series_multiwindow(
            lambda_windows=restrain_stage.lam_windows,
            equilibrated=True,
            run_nos=[1, 2],
        )

        # Check that the output has the correct shape
        assert overall_dgs.shape == (2, 100)
        assert overall_times.shape == (2, 100)

        # Check that the total time is what we expect
        tot_simtime = restrain_stage.get_tot_simtime(run_nos=[1])
        assert overall_times[0][-1] == pytest.approx(tot_simtime, abs=1e-2)

        # Check that the output values are correct
        assert overall_dgs.mean(axis=0)[-1] == pytest.approx(1.7751, abs=1e-2)
        assert overall_times.sum(axis=0)[-1] == pytest.approx(2.4, abs=1e-2)

    except Exception as e:
        raise e

    finally:
        # Reset the "slurm_equil_detection" attribute on the lambda windows
        # so that slurm is not used for the MBAR analysis
        restrain_stage.recursively_set_attr("slurm_equil_detection", False, force=True)
