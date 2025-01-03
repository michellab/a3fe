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
    assert results_exp.loc["t4l", "calc_dg"] == pytest.approx(5.0378, abs=1e-2)
    assert results_exp.loc["t4l", "calc_er"] == pytest.approx(0.1501, abs=1e-2)
    assert results_exp.loc["mdm2_pip2_short", "calc_dg"] == pytest.approx(
        8.4956, abs=1e-2
    )
    assert results_exp.loc["mdm2_pip2_short", "calc_er"] == pytest.approx(
        0.0935, abs=1e-2
    )


def test_calc_set_sub_sim_attrs(calc_set):
    """Test that the attributes of the sub-simulation runners are correctly saved."""

    def check_attr(sim_runner, attr_name="test_attr", value=42, force=True):
        """Recursively check that the attribute is set for all sub-simulation runners."""
        if force:
            assert hasattr(sim_runner, attr_name)
            assert getattr(sim_runner, attr_name) == value
        else:
            if hasattr(sim_runner, attr_name):
                assert getattr(sim_runner, attr_name) == value

        for sub_sim_runner in sim_runner._sub_sim_runners:
            check_attr(sub_sim_runner)

    # Try forcing an attribute to be set
    calc_set.recursively_set_attr("test_attr", force=True, value=42)
    check_attr(calc_set, attr_name="test_attr", value=42, force=True)

    # Also check that we can set the equilibration time attribute
    calc_set.set_equilibration_time(0.77)
    check_attr(calc_set, attr_name="_equilibration_time", value=0.77, force=False)
    check_attr(calc_set, attr_name="_equilibrated", value=True, force=False)
