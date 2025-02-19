"""Unit and regression tests for the CalcSet class."""

import os

import pytest

import pandas as pd


def test_calc_set_analysis(calc_set):
    """Test the analysis method of the CalcSet class."""
    # Make sure an error is raised if the exp dgs file is not supplied
    with pytest.raises(ValueError):
        calc_set.analyse(exp_dgs_path=None)

    # Run the analysis without the experimental data
    calc_set.analyse(compare_to_exp=False)

    # First, check that we have expected outputs in the output directory
    output_dir = os.path.join(calc_set.base_dir, "output")
    results_no_exp_path = os.path.join(output_dir, "results_summary.txt")
    assert os.path.exists(results_no_exp_path)
    results_no_exp = pd.read_csv(results_no_exp_path, index_col=0)

    # Repeat the analysis with the experimental data and check we get
    # the same results, and that the results are as expected
    exp_dgs_path = os.path.join(calc_set.base_dir, "input/exp_dgs.csv")
    calc_set.analyse(exp_dgs_path=exp_dgs_path, offset=False)

    # First, check that we have expected outputs in the output directory
    output_dir = os.path.join(calc_set.base_dir, "output")
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "overall_results.png"))
    assert os.path.exists(os.path.join(output_dir, "overall_stats.txt"))
    assert os.path.exists(os.path.join(output_dir, "results_summary.txt"))

    # Then, check that the results in results_summary.txt are as expected
    results_exp = pd.read_csv(
        os.path.join(output_dir, "results_summary.txt"), index_col=0
    )
    # Check that the results are the same once the experimental data is dropped
    assert results_no_exp.drop(columns=["exp_dg", "exp_er"]).equals(
        results_exp.drop(columns=["exp_dg", "exp_er"])
    )

    # Regression test for the results
    assert results_exp.loc["t4l", "calc_dg"] == pytest.approx(4.6929, abs=1e-2)
    assert results_exp.loc["t4l", "calc_er"] == pytest.approx(0.1120, abs=1e-2)
    assert results_exp.loc["mdm2_short", "calc_dg"] == pytest.approx(
        7.9100, abs=1e-2
    )
    assert results_exp.loc["mdm2_short", "calc_er"] == pytest.approx(
        0.1870, abs=1e-2
    )
