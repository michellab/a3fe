"""
Unit and regression test for functionality in the read module.
"""
import os
from tempfile import TemporaryDirectory

import pytest

from ..read._process_somd_files import (read_mbar_gradients, read_mbar_pmf,
                                        read_mbar_result, read_overlap_mat,
                                        write_truncated_sim_datafile)


def test_read_mbar_result():
    """Test that the read_mbar_result function works correctly"""
    free_energy, free_energy_err = read_mbar_result(
        "EnsEquil/data/example_output/freenrg-MBAR-run_01.dat"
    )
    assert free_energy == 2.613516
    assert free_energy_err == 0.037911


def test_read_overlap_mat():
    """Test that the read_overlap_mat function works correctly"""
    overlap_mat = read_overlap_mat(
        "EnsEquil/data/example_output/freenrg-MBAR-run_01.dat"
    )
    assert overlap_mat[0, 0] == 0.3659
    assert overlap_mat[0, 1] == 0.2730
    assert overlap_mat[-1, -1] == 0.1770


def test_read_pmf():
    """Test that the PMF is read correctly"""
    lam_vals, pmf, pmf_err = read_mbar_pmf(
        "EnsEquil/data/example_output/freenrg-MBAR-run_01.dat"
    )
    assert lam_vals[0] == 0.0
    assert lam_vals[-1] == 1.0
    assert lam_vals[20] == 0.3150
    assert len(lam_vals) == 44
    assert pmf[0] == 0.0
    assert pmf[-1] == 2.6135
    assert pmf[20] == -5.4155
    assert len(pmf) == 44
    assert pmf_err[0] == 0.0
    assert pmf_err[-1] == 0.0379
    assert pmf_err[20] == 0.0269
    assert len(pmf_err) == 44


def test_mbar_gradients():
    """Test that the MBAR gradients are read correctly"""
    av_lams, grads, grads_errs = read_mbar_gradients(
        "EnsEquil/data/example_output/freenrg-MBAR-run_01.dat"
    )
    assert av_lams[0] == 0.0045
    assert av_lams[-1] == 0.9900
    assert av_lams[20] == 0.326
    assert len(av_lams) == 43
    assert pytest.approx(grads[0], 1e-3) == -293.088
    assert pytest.approx(grads[-1], 1e-3) == 27.635
    assert pytest.approx(grads[20], 1e-3) == 25.691
    assert len(grads) == 43
    assert pytest.approx(grads_errs[0], 1e-3) == 0.311
    assert pytest.approx(grads_errs[-1], 1e-3) == 0.020
    assert pytest.approx(grads_errs[20], 1e-3) == 0.1727
    assert len(grads_errs) == 43


def test_write_truncated_sim_datafile_end():
    """
    Test that the write_truncated_sim_datafile function works correctly
    when truncating the end of the data
    """
    with TemporaryDirectory() as tmpdir:
        write_truncated_sim_datafile(
            "EnsEquil/data/example_output/simfile.dat",
            os.path.join(tmpdir, "simfile.dat"),
            0.1,
        )
        with open(os.path.join(tmpdir, "simfile.dat"), "r") as f:
            lines = f.readlines()
        assert lines[13].split()[0] == "100"
        assert lines[-2].split()[0] == "1000"
        write_truncated_sim_datafile(
            "EnsEquil/data/example_output/simfile.dat",
            os.path.join(tmpdir, "simfile_2.dat"),
            1,
        )
        with open(os.path.join(tmpdir, "simfile_2.dat"), "r") as f:
            lines = f.readlines()
        assert lines[-2].split()[0] == "10000"


def test_write_truncated_sim_datafile_end_and_start():
    """
    Test that the write_truncated_sim_datafile function works correctly
    when truncating the end of the data and the start of the data.
    """
    with TemporaryDirectory() as tmpdir:
        write_truncated_sim_datafile(
            "EnsEquil/data/example_output/simfile.dat",
            os.path.join(tmpdir, "simfile_start_trunc.dat"),
            fraction_final=0.9,
            fraction_initial=0.5,
        )
        with open(os.path.join(tmpdir, "simfile_start_trunc.dat"), "r") as f:
            lines = f.readlines()
        assert lines[13].split()[0] == "5000"
        assert lines[-2].split()[0] == "9000"
