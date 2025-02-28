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
        [1.30255,  1.263541, 1.256123, 1.304293, 1.333761, 1.314648, 1.320447, 1.313836,
         1.325728, 1.333878, 1.33893,  1.34218,  1.329786, 1.328869, 1.338726, 1.337869,
         1.332531, 1.329312, 1.326862, 1.327306],
        [1.568074, 1.626966, 1.505023, 1.421106, 1.406772, 1.388232, 1.358939, 1.352856,
         1.362695, 1.366573, 1.365685, 1.36921,  1.359138, 1.358404, 1.360066, 1.357095,
         1.363388, 1.360922, 1.352165, 1.343858],
        [1.370002, 1.348377, 1.386502, 1.315088, 1.383008, 1.427251, 1.458077, 1.445036,
         1.43719,  1.432348, 1.410855, 1.392283, 1.378152, 1.366362, 1.363622, 1.366651,
         1.358232, 1.347265, 1.346105, 1.339764],
        [1.329163, 1.375653, 1.393664, 1.379855, 1.349581, 1.342975, 1.342577, 1.335664,
         1.332043, 1.349121, 1.350339, 1.366634, 1.383481, 1.397546, 1.411033, 1.420306,
         1.437954, 1.432909, 1.429654, 1.421102],
        [1.304829, 1.362232, 1.342601, 1.328957, 1.318289, 1.29414,  1.304097, 1.305963,
         1.313026, 1.363487, 1.409961, 1.462056, 1.521547, 1.561595, 1.600894, 1.640916,
         1.684132, 1.71397,  1.740793, 1.765496],
    ]
)


def test_analysis_all_runs(restrain_stage):
    """Check that the analysis works on all runs."""
    res, err = restrain_stage.analyse()
    assert res.mean() == pytest.approx(1.4395, abs=1e-2)
    assert err.mean() == pytest.approx(0.0267, abs=1e-3)


def test_analysis_all_runs_fraction(restrain_stage):
    """Check that the analysis works on all runs."""
    res, err = restrain_stage.analyse(fraction=0.5)
    assert res.mean() == pytest.approx(1.369, abs=1e-2)
    assert err.mean() == pytest.approx(0.0319, abs=1e-3)


def test_get_results_df(restrain_stage):
    """Check that the results dataframe is correctly generated."""
    # Re-analyse to ensure that the order of the tests doesn't matter
    res, err = restrain_stage.analyse()
    df = restrain_stage.get_results_df()
    # Check that the csv has been output
    assert os.path.exists(os.path.join(restrain_stage.output_dir, "results.csv"))
    # Check that the results are correct
    assert df.loc["restrain_stage"]["dg / kcal mol-1"] == pytest.approx(1.44, abs=1e-1)
    assert df.loc["restrain_stage"]["dg_95_ci / kcal mol-1"] == pytest.approx(
        0.23, abs=1e-2
    )
    assert df.loc["restrain_stage"]["tot_simtime / ns"] == pytest.approx(6.0, abs=1e-1)
    assert df.loc["restrain_stage"]["tot_gpu_time / GPU hours"] == pytest.approx(
        1, abs=1e-0
    )


def test_analysis_subselection_runs(restrain_stage):
    """Check that the analysis works on a subselection of runs."""
    res, err = restrain_stage.analyse(run_nos=[1, 2, 4])
    assert res.mean() == pytest.approx(1.3641, abs=1e-2)
    assert err.mean() == pytest.approx(0.0218, abs=1e-3)


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
    assert overall_dgs.mean(axis=0)[-1] == pytest.approx(1.6446, abs=1e-2)
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
    assert overall_dgs.mean(axis=0)[-1] == pytest.approx(1.6446, abs=1e-2)
    assert overall_times.sum(axis=0)[-1] == pytest.approx(2.4, abs=1e-2)


# Parameterise with a dictionary of equilibration functions and expected results
@pytest.mark.parametrize(
    "equil_func, expected",
    [
        (check_equil_block_gradient, 0.0024),
        (check_equil_chodera, 0.0007999),
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
        (check_equil_multiwindow_kpss, 0.5, {}),
        (
            check_equil_multiwindow_modified_geweke,
            0.0555,
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
            0.0: 1.2701819053567023,
            0.125: 1.0000427598019113,
            0.25: 1.0875242491052906,
            0.375: 1.005262713994976,
            0.5: 1.0002510744669293,
            1.0: 0.9994696617711509,
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
    assert dgs1[-1][-1] == pytest.approx(1.765496, abs=1e-2)


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
    assert dgs1[-1][-1] == pytest.approx(3.42643, abs=1e-2)


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
    assert improvement_sem_20 > 0.97
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
    assert res.mean() == pytest.approx(1.4395, abs=1e-2)
    assert err.mean() == pytest.approx(0.0267, abs=1e-3)


@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
@pytest.mark.skipif(not RUN_SLURM_TESTS, reason="RUN_SLURM_TESTS is False")
def test_analysis_all_runs_fraction_slurm(restrain_stage):
    """Check that the analysis works on all runs."""
    res, err = restrain_stage.analyse(fraction=0.5, slurm=True)
    assert res.mean() == pytest.approx(1.369, abs=1e-2)
    assert err.mean() == pytest.approx(0.0319, abs=1e-3)


@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
@pytest.mark.skipif(not RUN_SLURM_TESTS, reason="RUN_SLURM_TESTS is False")
def test_analysis_subselection_runs_slurm(restrain_stage):
    """Check that the analysis works on a subselection of runs."""
    res, err = restrain_stage.analyse(run_nos=[1, 2, 4], slurm=True)
    assert res.mean() == pytest.approx(1.3641, abs=1e-2)
    assert err.mean() == pytest.approx(0.0218, abs=1e-3)


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
        assert overall_dgs.mean(axis=0)[-1] == pytest.approx(1.6446, abs=1e-2)
        assert overall_times.sum(axis=0)[-1] == pytest.approx(2.4, abs=1e-2)

    except Exception as e:
        raise e

    finally:
        # Reset the "slurm_equil_detection" attribute on the lambda windows
        # so that slurm is not used for the MBAR analysis
        restrain_stage.recursively_set_attr("slurm_equil_detection", False, force=True)
