""" Unit and regression test for the Calculation class. """

from asyncio import subprocess
from glob import glob
import logging
import os
import pytest
import subprocess
from tempfile import TemporaryDirectory

import EnsEquil as ee

LEGS_WITH_STAGES = {"bound" : ["discharge", "vanish"], "free": ["discharge", "vanish"]}

# Create a Calculation object for future tests

@pytest.fixture(scope = "module")
def calc():
    """Create a calculation object to use in tests"""
    with TemporaryDirectory() as dirname:
        calc = ee.Calculation(base_dir=dirname, 
                              input_dir="EnsEquil/data/example_run_dir/input",
                              ensemble_size=6,)
        calc._dump()
        # Must use yield so that the temporary directory is deleted after the tests
        # by the context manager and does not persist
        yield calc

# Load calc and check it has all the required stuff
def test_calculation_loading(calc):
    """Check that the calculation loads correctly"""
    # Check that the calculation has the correct attributes
    assert calc.loaded_from_pickle == False
    assert calc.ensemble_size == 6
    assert calc.input_dir == "EnsEquil/data/example_run_dir/input"
    assert calc.output_dir == os.path.join(calc.base_dir, "output")
    assert calc.setup_complete == False
    assert calc.prep_stage == ee.run.enums.PreparationStage.PARAMETERISED
    assert calc.stream_log_level == logging.INFO
    # Check that pickle file exists
    assert os.path.exists(os.path.join(calc.base_dir, "Calculation.pkl"))

def test_calculation_logging(calc):
    """Check that the calculation logging is set up correctly"""
    assert calc._logger.name == "Calculation_0"
    assert type(calc._logger.handlers[0]) == logging.FileHandler # type: ignore
    assert calc._logger.handlers[0].baseFilename == os.path.join(calc.base_dir, "Calculation.log") # type: ignore
    assert calc._logger.handlers[0].level == logging.DEBUG # type: ignore
    assert type(calc._logger.handlers[1]) == logging.StreamHandler # type: ignore
    assert calc._logger.handlers[1].level == logging.INFO # type: ignore

    # Try writing to the log and check that it's written to the file
    calc._logger.info("Test message")
    with open(os.path.join(calc.base_dir, "Calculation.log"), "r") as f:
        assert "Test message" in f.read()

def test_calculation_reloading(calc):
    """Check that the calculations can be correctly loaded from a pickle."""
    calc2 = ee.Calculation(base_dir=calc.base_dir, input_dir="EnsEquil/data/example_run_dir/input")
    assert calc2.loaded_from_pickle == True
    assert calc2.ensemble_size == 6
    assert calc2.input_dir == "EnsEquil/data/example_run_dir/input"
    assert calc2.output_dir == os.path.join(calc.base_dir, "output")
    assert calc2.setup_complete == False
    assert calc2.prep_stage == ee.run.enums.PreparationStage.PARAMETERISED
    assert calc2.stream_log_level == logging.INFO

def test_update_paths(calc):
    """Check that the calculation paths can be updated correctly."""
    with TemporaryDirectory() as new_dir:
        for file in glob(os.path.join(calc.base_dir, "*")):
            subprocess.run(["cp", "-r", file, new_dir])
        calc3 = ee.Calculation(base_dir=new_dir, input_dir="EnsEquil/data/example_run_dir/input")
        assert calc3.loaded_from_pickle == True
        current_dir = os.getcwd()
        calc3.update_paths(old_sub_path = calc3.base_dir, new_sub_path = current_dir)
        assert calc3.base_dir == current_dir
        assert calc3._logger.handlers[0].baseFilename == os.path.join(current_dir, "Calculation.log") # type: ignore

######################## Testings Requiring SLURM ########################

# Check if slurm is present and only run following tests if so
SLURM_PRESENT = subprocess.run(["sinfo"], capture_output=True).returncode == 0

@pytest.fixture(scope = "module")
def calc_slurm():
    """Create a calculation object to use in tests"""
    with TemporaryDirectory() as dirname:
        local_dir = "EnsEquil/test/temp_test_output"
        # Copy the example input directory to the temporary directory
        # as we'll create some new files there
        subprocess.run(["mkdir", "-p", local_dir])
        subprocess.run(["cp", "-r", "EnsEquil/data/example_run_dir/input", f"{local_dir}/input"])
        calc = ee.Calculation(base_dir=local_dir, 
                              input_dir=f"{local_dir}/input",
                              ensemble_size=2,
                              stream_log_level=logging.CRITICAL,) # Shut up the logging
        calc._dump()
        # Must use yield so that the temporary directory is deleted after the tests
        # by the context manager and does not persist
        yield calc

# Test that the preparation stages work 
@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
def test_integration_calculation(calc_slurm):
    """Integration test to check that all major stages of the calculation work."""
    
    # Check that the preparation stages work
    assert calc_slurm.prep_stage == ee.run.enums.PreparationStage.PARAMETERISED
    calc_slurm.setup(short_ensemble_equil=True)
    assert calc_slurm.setup_complete == True
    # Check that all required slurm bash jobs have been created
    required_slurm_jobnames = ["minimise", "solvate", "heat_preequil"]
    for leg in LEGS_WITH_STAGES.keys():
        for jobname in [f"{job}_{leg}.sh" for job in required_slurm_jobnames]:
            assert os.path.exists(os.path.join(calc_slurm.base_dir, "input", jobname))
    # Check that all required slurm output files have been created
    assert len(glob(os.path.join(calc_slurm.base_dir, "input", "*.out"))) == 6
    # Check that all required stage dirs have been created for each leg
    for leg in LEGS_WITH_STAGES.keys():
        for stage in LEGS_WITH_STAGES[leg]:
            assert os.path.exists(os.path.join(calc_slurm.base_dir, leg, stage))
    # Check that all required input files have been created 
    for leg in LEGS_WITH_STAGES.keys():
        for stage in LEGS_WITH_STAGES[leg]:
            required_files =  ["somd.cfg", 
                               "somd.rst7", 
                               "somd.prm7", 
                               "somd.pert", 
                               "run_somd.sh",
                               "somd_1.rst7",
                               "somd_2.rst7",]
            if leg == "bound":
                required_files.extend(["restraint_1.txt", "restraint_2.txt"])
            for file in required_files:
                assert os.path.exists(os.path.join(calc_slurm.base_dir, leg, stage, "input", file))
    # Check that the optimal lambda window creation function works
    calc_slurm.get_optimal_lam_vals()
    # Check that the old runs have been moved, and that the new lambda values are
    # different to the old ones
    for leg in LEGS_WITH_STAGES.keys():
        for stage in LEGS_WITH_STAGES[leg]:
            assert os.path.exists(os.path.join(calc_slurm.base_dir, leg, stage, "output_saved"))
            lam_dirs_old = glob(os.path.join(calc_slurm.base_dir, leg, stage, "output_saved", "lam*"))
            lam_dirs_new = glob(os.path.join(calc_slurm.base_dir, leg, stage, "output", "lam*"))
            assert lam_dirs_old != lam_dirs_new

    # Check that a short, non-adaptive run works
    calc_slurm.run(adaptive=False, runtime=0.1)
    calc_slurm.wait()
    assert calc_slurm.running == False
    # Set the entire simulation time to equilibrated
    for leg in calc_slurm.legs:
        for stage in leg.stages:
            for lam_win in stage.lam_windows:
                lam_win._equilibrated = True
                lam_win._equil_time = 0
    for leg in calc_slurm.legs:
        for stage in leg.stages:
            for lam_win in stage.lam_windows:
                assert round(lam_win.tot_simtime / lam_win.ensemble_size, 1) == 0.1

    # Check that analysis runs and gives a sane result
    calc_slurm.analyse()
    # Check that the output file exists
    assert os.path.exists(os.path.join(calc_slurm.base_dir, "output", "overall_stats.dat"))
    # Check that the free energy change is sane
    with open(os.path.join(calc_slurm.base_dir, "output", "overall_stats.dat"), "r") as f:
        dg = float(f.readlines()[1].split(" ")[3])
        print(dg)
        assert dg < -5
        assert dg > -25

    # Check that all calculations can be killed
    calc_slurm.run(adaptive=False, runtime=0.1)
    calc_slurm.kill()
    assert not calc_slurm.running





# Test that finding optimal lambda windows works
#@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
#def test_get_optimal_lam_vals(calc_slurm):
    #"""Check that the optimal lambda window creation function works"""
    #calc_slurm.get_optimal_lam_vals()
    ## Check that the old runs have been moved, and that the new lambda values are
    ## different to the old ones
    #for leg in LEGS_WITH_STAGES.keys():
        #for stage in LEGS_WITH_STAGES[leg]:
            #assert os.path.exists(os.path.join(calc_slurm.base_dir, leg, stage, "output_saved"))
            #lam_dirs_old = glob(os.path.join(calc_slurm.base_dir, leg, stage, "output_saved", "lam*"))
            #lam_dirs_new = glob(os.path.join(calc_slurm.base_dir, leg, stage, "output", "lam*"))
            #assert lam_dirs_old != lam_dirs_new

## Test that the simulation runs and
#@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
#def test_run_calc_non_adaptive(calc_slurm):
    #"""Check that a short, non-adaptive run works"""
    #calc_slurm.run(adaptive=False, runtime=0.1)
    #calc_slurm.wait()
    #assert calc_slurm.running == False
    ## Set the entire simulation time to equilibrated
    #for leg in calc_slurm.legs:
        #for stage in leg.stages:
            #for lam_win in stage.lam_windows:
                #lam_win._equilibrated = True
                #lam_win._equil_time = 0
    #for leg in calc_slurm.legs:
        #for stage in leg.stages:
            #for lam_win in stage.lam_windows:
                #assert round(lam_win.tot_simtime / lam_win.ensemble_size, 1) == 0.1

## Test that the analysis runs and gives a reasonable result
#@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
#def test_analysis(calc_slurm):
    #"""Check that analysis runs and gives a sane result"""
    #calc_slurm.analyse()
    ## Check that the output file exists
    #assert os.path.exists(os.path.join(calc_slurm.base_dir, "output", "overall_stats.dat"))
    ## Check that the free energy change is sane
    #with open(os.path.join(calc_slurm.base_dir, "output", "overall_stats.dat"), "r") as f:
        #dg = float(f.readlines()[1].split(" ")[3])
        #print(dg)
        #assert dg < -5
        #assert dg > -25

#@pytest.mark.skipif(not SLURM_PRESENT, reason="SLURM not present")
#def test_kill_calc(calc_slurm):
    #"""Check that all calculations can be killed"""
    #calc_slurm.run(adaptive=False, runtime=0.1)
    #calc_slurm.kill()
    #assert not calc_slurm.running
