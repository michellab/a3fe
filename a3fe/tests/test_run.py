"""Unit and regression tests for the run module."""

import logging
import os
import pathlib
import subprocess
from glob import glob
from tempfile import TemporaryDirectory
from typing import Optional
from unittest.mock import patch

import BioSimSpace.Sandpit.Exscientia as BSS
import numpy as np
import pytest

import a3fe as a3
from a3fe.analyse.detect_equil import dummy_check_equil_multiwindow


LEGS_WITH_STAGES = {"bound": ["discharge", "vanish"], "free": ["discharge", "vanish"]}


# Load calc and check it has all the required stuff
def test_calculation_loading(calc):
    """Check that the calculation loads correctly"""
    # Check that the calculation has the correct attributes
    assert not calc.loaded_from_pickle
    assert calc.ensemble_size == 6
    assert calc.input_dir == str(
        pathlib.Path("a3fe/data/example_run_dir/input").resolve()
    )
    assert calc.output_dir == os.path.join(calc.base_dir, "output")
    assert not calc.setup_complete
    assert (
        calc.prep_stage.name
        == a3.configuration.enums.PreparationStage.PARAMETERISED.name
    )
    assert calc.stream_log_level == logging.INFO
    # Check that pickle file exists
    assert os.path.exists(os.path.join(calc.base_dir, "Calculation.pkl"))


def test_calculation_logging(calc):
    """Check that the calculation logging is set up correctly"""
    assert type(calc._logger.handlers[0]) is logging.FileHandler
    assert calc._logger.handlers[0].baseFilename == os.path.join(
        calc.base_dir, "Calculation.log"
    )
    assert calc._logger.handlers[0].level == logging.DEBUG
    assert type(calc._logger.handlers[1]) is logging.StreamHandler
    assert calc._logger.handlers[1].level == logging.INFO

    # Try writing to the log and check that it's written to the file
    calc._logger.info("Test message")
    with open(os.path.join(calc.base_dir, "Calculation.log"), "r") as f:
        assert "Test message" in f.read()


def test_calculation_reloading(calc):
    """Check that the calculations can be correctly loaded from a pickle."""
    calc2 = a3.Calculation(
        base_dir=calc.base_dir, input_dir="a3fe/data/example_run_dir/input"
    )
    assert calc2.loaded_from_pickle
    assert calc2.ensemble_size == 6
    assert calc2.input_dir == str(
        pathlib.Path("a3fe/data/example_run_dir/input").resolve()
    )
    assert calc2.output_dir == os.path.join(calc.base_dir, "output")
    assert not calc2.setup_complete
    assert (
        calc2.prep_stage.name
        == a3.configuration.enums.PreparationStage.PARAMETERISED.name
    )
    assert calc2.stream_log_level == logging.INFO


def test_logging_level(calc):
    """Check that changing the logging level works as expected."""
    calc3 = a3.Calculation(
        base_dir=calc.base_dir, input_dir="a3fe/data/example_run_dir/input"
    )
    calc3.stream_log_level = logging.WARNING
    assert calc3._logger.handlers[1].level == logging.WARNING
    assert calc3._logger.handlers[0].level == logging.DEBUG


def test_update_paths(calc):
    """Check that the calculation paths can be updated correctly."""
    with TemporaryDirectory() as new_dir:
        for file in glob(os.path.join(calc.base_dir, "*")):
            subprocess.run(["cp", "-r", file, new_dir])
        calc4 = a3.Calculation(
            base_dir=new_dir, input_dir="a3fe/data/example_run_dir/input"
        )
        assert calc4.loaded_from_pickle
        current_dir = os.getcwd()
        calc4.update_paths(old_sub_path=calc4.base_dir, new_sub_path=current_dir)
        assert calc4.base_dir == current_dir
        assert calc4._logger.handlers[0].baseFilename == os.path.join(
            current_dir, "Calculation.log"
        )  # type: ignore


def test_set_and_get_attributes(restrain_stage):
    """Check that the calculation attributes can be set and obtained correctly."""
    attr_dict = restrain_stage.recursively_get_attr("ensemble_size")
    assert attr_dict["ensemble_size"] == 5
    assert (
        attr_dict["sub_sim_runners"][list(attr_dict["sub_sim_runners"].keys())[0]][
            "ensemble_size"
        ]
        == 5
    )
    # Check it fails if the attribute doesn't exist
    restrain_stage.recursively_set_attr("ensemble_sizee", 7)
    attr_dict = restrain_stage.recursively_get_attr("ensemble_sizee")
    assert attr_dict["ensemble_sizee"] is None
    # Check that we can force it to set the attribute
    restrain_stage.recursively_set_attr("ensemble_sizee", 7, force=True)
    attr_dict = restrain_stage.recursively_get_attr("ensemble_sizee")
    assert attr_dict["ensemble_sizee"] == 7
    # Change the ensemble size attribute
    restrain_stage.recursively_set_attr("ensemble_sizee", 7)
    attr_dict = restrain_stage.recursively_get_attr("ensemble_sizee")
    assert attr_dict["ensemble_sizee"] == 7


def test_reset(restrain_stage):
    """Test that runtime attributes are reset correctly"""
    # First, check that they're consistent with having run the stage
    equilibrated = all([lam._equilibrated for lam in restrain_stage.lam_windows])
    equil_times = [lam._equil_time for lam in restrain_stage.lam_windows]
    assert equilibrated
    assert None not in equil_times
    # Now reset the stage and recheck
    restrain_stage.reset()
    print([lam._equilibrated for lam in restrain_stage.lam_windows])
    equilibrated = any([lam._equilibrated for lam in restrain_stage.lam_windows])
    equil_times = [lam._equil_time for lam in restrain_stage.lam_windows]
    equil_times_none = all([time is None for time in equil_times])
    assert not equilibrated
    assert equil_times_none


def test_set_equilibration_time(restrain_stage):
    """Test that the set_equilibration_time method works"""
    # First, set to unequilibrated
    restrain_stage.reset()
    restrain_stage.set_equilibration_time(0.1)
    assert all([lam._equil_time == 0.1 for lam in restrain_stage.lam_windows])
    assert all([lam._equilibrated for lam in restrain_stage.lam_windows])
    assert restrain_stage.equil_time == pytest.approx(
        0.1 * len(restrain_stage.lam_windows), abs=1e-6
    )


def test_update(restrain_stage):
    """Check that the stage update method works"""
    # Change the positions of the lambda windows and ensemble size
    # and ensure that this is reflected in the lambda windows
    # after updating
    new_lam_vals = list(np.arange(0.0, 1.1, 0.1))
    restrain_stage.engine_config.lambda_values = new_lam_vals
    restrain_stage.ensemble_size = 2
    restrain_stage.update()
    assert len(restrain_stage.lam_windows) == 11
    for lam, lam_val in zip(restrain_stage.lam_windows, new_lam_vals):
        assert lam.lam == lam_val
        assert lam.ensemble_size == 2


def test_simulation_runner_iterator(restrain_stage):
    """Test that the simulation runner iterator works as expected."""
    # Take the first 3 lambda windows
    base_dirs = [window.base_dir for window in restrain_stage.lam_windows[:3]]
    sim_runner_iterator = a3.run._utils.SimulationRunnerIterator(
        base_dirs=base_dirs,
        subclass=a3.LamWindow,
        lam=0,  # Overwritten once pickle loaded
        virtual_queue=restrain_stage.virtual_queue,
    )
    # Check that the iterator works as expected
    for i, sim_runner in enumerate(sim_runner_iterator):
        assert sim_runner.base_dir == base_dirs[i]

    # Cycle through the iterator again and check that it still works
    for i, sim_runner in enumerate(sim_runner_iterator):
        assert sim_runner.base_dir == base_dirs[i]


def test_parameterisation_free(t4l_calc, system_prep_config, engine_type):
    """Test that the parameterisation of benzene works as expected."""

    leg_type = a3.LegType.FREE
    free_leg = a3.Leg(leg_type=leg_type, base_dir=t4l_calc.base_dir)

    try:
        # We need to save the config to the input directory
        system_prep_config(
            slurm=False,
            forcefields={"ligand": "gaff2", "protein": "ff14SB", "water": "tip3p"},
        ).dump(t4l_calc.input_dir, leg_type)
        # Parameterise benzene
        a3.PreparationStage.PARAMETERISED.prep_fn(
            leg_type=leg_type,
            engine_type=engine_type,
            input_dir=t4l_calc.input_dir,
            output_dir=free_leg.input_dir,
        )

        # Check that the expected files are produced
        expected_files = ["free_param.rst7", "free_param.prm7"]
        assert all([file in os.listdir(free_leg.input_dir) for file in expected_files])

        # Check that the ligand is uncharged
        bss_lig = BSS.IO.readMolecules(f"{free_leg.input_dir}/free_param.*")[0]
        assert bss_lig.charge().value() == 0

        # Check that they have properties that only charged molecules should have
        expected_properties = ["charge", "LJ"]
        assert all(
            [
                prop in bss_lig._sire_object.property_keys()
                for prop in expected_properties
            ]
        )

        # Check that the charge on the first atom is as expected
        assert (
            pytest.approx(bss_lig.getAtoms()[0].charge().value(), abs=1e-6) == -0.1300
        )

    # Always delete Leg.pkl
    finally:
        os.remove(f"{free_leg.base_dir}/Leg.pkl")


def test_parameterisation_bound(t4l_calc, system_prep_config, engine_type):
    """Test that the parameterisation of benzene and T4L works as expected."""
    leg_type = a3.LegType.BOUND
    bound_leg = a3.Leg(leg_type=leg_type, base_dir=t4l_calc.base_dir)

    try:
        # We need to save the config to the input directory
        system_prep_config(
            slurm=False,
            forcefields={"ligand": "gaff2", "protein": "ff14SB", "water": "tip3p"},
        ).dump(t4l_calc.input_dir, leg_type)
        # Parameterise benzene
        assert leg_type == a3.LegType.BOUND
        assert bound_leg.leg_type == leg_type
        a3.PreparationStage.PARAMETERISED.prep_fn(
            leg_type=leg_type,
            engine_type=engine_type,
            input_dir=t4l_calc.input_dir,
            output_dir=bound_leg.input_dir,
        )

        # Check that the expected files are produced
        expected_files = ["bound_param.rst7", "bound_param.prm7"]
        assert all([file in os.listdir(bound_leg.input_dir) for file in expected_files])

        # Check that the ligand is uncharged
        bss_sys = BSS.IO.readMolecules(f"{bound_leg.input_dir}/bound_param.*")
        bss_lig = bss_sys[0]
        assert bss_lig.charge().value() == 0

        # Check that they have properties that only charged molecules should have
        expected_properties = ["charge", "LJ"]
        assert all(
            [
                prop in bss_lig._sire_object.property_keys()
                for prop in expected_properties
            ]
        )

        # Check that the charge on the first atom is as expected
        assert (
            pytest.approx(bss_lig.getAtoms()[0].charge().value(), abs=1e-6) == -0.1300
        )

        # Check that the protein has +9 charge
        bss_prot = bss_sys[1]
        assert round(bss_prot.charge().value()) == 9

    # Always delete Leg.pkl
    finally:
        os.remove(f"{bound_leg.base_dir}/Leg.pkl")


class TestCalcSetup:
    """
    Test the setup of a calculation and all sub-simulation runners.
    The function 'run_process' is patched so that the setup stages can be tested
    without actually running the simulations.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def mock_run_process(complex_sys):
        """
        Mock the run_process function so that we can test the setup stages without
        actually running.
        """
        # Store the original run_process function
        # original_run_process = a3.run.system_prep.run_process

        def _mock_run_process_inner(
            system: BSS._SireWrappers._system.System,
            protocol: BSS.Protocol._protocol.Protocol,
            work_dir: Optional[str] = None,
        ) -> BSS._SireWrappers._system.System:
            """Mock the run_process function so that we can test the setup stages without
            actually running"""
            # If the protocol is production, this must be the Ensemble Equilibration stage.
            # If so, make sure that there is a gromacs.xtc file in the work_dir
            if isinstance(protocol, BSS.Protocol.Production) and work_dir is not None:
                traj_path = os.path.join(
                    "a3fe",
                    "data",
                    "example_run_dir",
                    "bound",
                    "ensemble_equilibration_1",
                    "gromacs.xtc",
                )
                subprocess.run(["cp", traj_path, work_dir])

            return complex_sys

        # Patch the run_process function with the mock
        # We need to patch in a3fe and EnsEquil
        with patch("a3fe.run.system_prep.run_process", new=_mock_run_process_inner):
            yield

    @staticmethod
    @pytest.fixture(scope="class")
    def setup_calc(mock_run_process, system_prep_config, engine_type):
        """Set up a calculation object completely from input files."""
        with TemporaryDirectory() as dirname:
            # Copy the example input directory to the temporary directory
            # as we'll create some new files there
            subprocess.run(
                [
                    "cp",
                    "-r",
                    "a3fe/data/example_run_dir/input",
                    f"{dirname}/input",
                ]
            )

            setup_calc = a3.Calculation(
                base_dir=dirname,
                input_dir=f"{dirname}/input",
                ensemble_size=1,
                stream_log_level=logging.CRITICAL,  # Silence the logging
                engine_type=engine_type,
            )
            assert (
                setup_calc.prep_stage.name
                == a3.configuration.enums.PreparationStage.PARAMETERISED.name
            )
            cfg = system_prep_config()
            cfg.slurm = False
            setup_calc.setup(sysprep_config=cfg)
            yield setup_calc

    def test_setup_calc_overall(self, setup_calc, mock_run_process):
        """Test that setting up the calculation was successful at a high level."""
        assert setup_calc.setup_complete
        assert (
            setup_calc.prep_stage.name
            == a3.configuration.enums.PreparationStage.PREEQUILIBRATED.name
        )
        assert len(setup_calc.legs) == 2
        legs = [leg.leg_type for leg in setup_calc.legs]
        assert a3.LegType.BOUND in legs
        assert a3.LegType.FREE in legs

    def test_correct_equil_algorithm(self, setup_calc, mock_run_process):
        """Test that the expected equilibration detection algorithms have been set."""
        assert setup_calc.equil_detection == "multiwindow"
        for leg in setup_calc.legs:
            assert leg.equil_detection == "multiwindow"
            for stage in leg.stages:
                assert stage.equil_detection == "multiwindow"
                for lam_win in stage.lam_windows:
                    assert lam_win.check_equil == dummy_check_equil_multiwindow

    def test_setup_calc_legs(self, setup_calc, mock_run_process):
        """Test that setting up the calculation produced the correct legs."""
        for leg in setup_calc.legs:
            expected_files = [
                "Leg.log",
                "vanish",
                "Leg.pkl",
                "discharge",
                "ensemble_equilibration_1",
                "virtual_queue.log",
            ]
            expected_stage_types = [a3.StageType.DISCHARGE, a3.StageType.VANISH]

            if leg.leg_type == a3.LegType.BOUND:
                expected_stage_types.append(a3.StageType.RESTRAIN)
                expected_files.append("restrain")

            stage_types = [stage.stage_type for stage in leg.stages]
            output_files = os.listdir(leg.base_dir)
            for stage_type in expected_stage_types:
                assert stage_type in stage_types
            for file in expected_files:
                assert file in output_files

    def test_setup_calc_stages(self, setup_calc, system_prep_config):
        """Test that setting up the calculation produced the correct stages."""
        for leg in setup_calc.legs:
            expected_input_files = {
                "somd_1.rst7",
                "somd.prm7",
                "somd.rst7",
                "somd.pert",
                "somd.err",
                "somd.out",
            }
            expected_base_files = {
                "input",
                "Stage.pkl",
                "output",
                "virtual_queue.log",
                "Stage.log",
            }

            for stage in leg.stages:
                input_files = set(os.listdir(stage.input_dir))
                assert expected_input_files == input_files
                base_files = set(os.listdir(stage.base_dir))
                assert base_files == expected_base_files

                lam_vals = {
                    float(lam.split("_")[1]) for lam in os.listdir(stage.output_dir)
                }
                cfg = system_prep_config()
                expected_lam_vals = set(
                    cfg.lambda_values[leg.leg_type][stage.stage_type]
                )
                assert lam_vals == expected_lam_vals

                if leg.leg_type == a3.LegType.BOUND:
                    assert stage.engine_config.boresch_restraints_dictionary is not None

    def test_setup_calc_lam(self, setup_calc):
        """Test that setting up the calculation produced the correct lambda windows."""
        expected_base_files = {"LamWindow.pkl", "LamWindow.log", "run_01"}
        for leg in setup_calc.legs:
            for stage in leg.stages:
                for lam_win in stage.lam_windows:
                    assert lam_win.ensemble_size == 1
                    assert set(os.listdir(lam_win.base_dir)) == expected_base_files

    def test_setup_calc_sims(self, setup_calc):
        """Test that setting up the calculation produced the correct simulations."""
        for leg in setup_calc.legs:
            expected_base_files = {
                "Simulation.pkl",
                "Simulation.log",
                "somd.prm7",
                "somd.rst7",
                "somd.pert",
                "somd.err",
                "somd.out",
            }
            for stage in leg.stages:
                for lam_win in stage.lam_windows:
                    for sim in lam_win.sims:
                        base_dir_files = set(os.listdir(sim.base_dir))
                        assert base_dir_files == expected_base_files

                        if leg.leg_type == a3.LegType.BOUND:
                            assert (
                                sim.engine_config.boresch_restraints_dictionary
                                is not None
                            )
