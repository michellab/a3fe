"""
Unit and regression test for functionality in the read module.
"""
import os
from tempfile import TemporaryDirectory

from ..read._process_somd_files import read_mbar_result, read_overlap_mat, write_truncated_sim_datafile

def test_read_mbar_result():
    """Test that the read_mbar_result function works correctly"""
    free_energy, free_energy_err = read_mbar_result("EnsEquil/data/example_output/freenrg-MBAR-run_01.dat")
    assert free_energy == 2.613516
    assert free_energy_err == 0.037911

def test_read_overlap_mat():
    """Test that the read_overlap_mat function works correctly"""
    overlap_mat = read_overlap_mat("EnsEquil/data/example_output/freenrg-MBAR-run_01.dat")
    assert overlap_mat[0, 0] == 0.3659
    assert overlap_mat[0, 1] == 0.2730
    assert overlap_mat[-1, -1] == 0.1770

def test_write_truncated_sim_datafile():
    """Test that the write_truncated_sim_datafile function works correctly"""
    with TemporaryDirectory() as tmpdir: 
        write_truncated_sim_datafile("EnsEquil/data/example_output/simfile.dat", os.path.join(tmpdir, "simfile.dat") , 0.1)
        with open(os.path.join(tmpdir, "simfile.dat"), 'r') as f:
            lines = f.readlines()
        assert lines[-2].split()[0] == "1000"
        write_truncated_sim_datafile("EnsEquil/data/example_output/simfile.dat", os.path.join(tmpdir, "simfile_2.dat") , 1)
        with open(os.path.join(tmpdir, "simfile_2.dat"), 'r') as f:
            lines = f.readlines()
        assert lines[-2].split()[0] == "10000"
    