"""Unit and regression tests for the CalcSet class."""

import os

import pytest


def test_calc_set_analysis(calc_set):
    """Test the analysis method of the CalcSet class."""
    exp_dgs_path = os.path.join(calc_set.base_dir, "input/exp_dgs.csv")
    calc_set.analyse(exp_dgs_path=exp_dgs_path, offset=False)

    # First, check that we have expected outputs in the output directory
    output_dir = os.path.join(calc_set.base_dir, "output")
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "overall_results.png"))
    assert os.path.exists(os.path.join(output_dir, "overall_stats.txt"))
    assert os.path.exists(os.path.join(output_dir, "results_summary.txt"))

    # Then, check that the results in results_summary.txt are as expected
    with open(os.path.join(output_dir, "results_summary.txt")) as f:
        lines = f.readlines()
    assert float(lines[1].split(",")[-2]) == pytest.approx(5.0378, abs=1e-2)
    assert float(lines[1].split(",")[-1]) == pytest.approx(0.1501, abs=1e-2)
    assert float(lines[2].split(",")[-2]) == pytest.approx(8.4956, abs=1e-2)
    assert float(lines[2].split(",")[-1]) == pytest.approx(0.0935, abs=1e-2)
