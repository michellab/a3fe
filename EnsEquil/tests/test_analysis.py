"""Testing the analysis module."""

import os
import numpy as np
import pytest
import subprocess
from tempfile import TemporaryDirectory

import EnsEquil as ee


@pytest.fixture(scope="module")
def restrain_stage():
    """Create a stage object with analysis data to use in tests"""
    with TemporaryDirectory() as dirname:
        # Copy the input files to the temporary directory
        subprocess.run(["cp", "-r", "EnsEquil/data/example_restraint_stage/", dirname])
        stage = ee.Stage(
            base_dir=os.path.join(dirname, "example_restraint_stage"),
            stage_type=ee.enums.StageType.RESTRAIN,
        )
        # Must use yield so that the temporary directory is deleted after the tests
        # by the context manager and does not persist
        yield stage


def test_analysis_all_runs(restrain_stage):
    """Check that the analysis works on all runs."""
    stage = restrain_stage
    res, err = stage.analyse()
    assert res.mean() == pytest.approx(1.5978, abs=1e-2)
    assert err.mean() == pytest.approx(0.0254, abs=1e-3)


def test_analysis_subselection_runs(restrain_stage):
    """Check that the analysis works on a subselection of runs."""
    stage = restrain_stage
    res, err = stage.analyse(run_nos=[1, 2, 4])
    assert res.mean() == pytest.approx(1.6154, abs=1e-2)
    assert err.mean() == pytest.approx(0.0257, abs=1e-3)


def test_convergence_analysis(restrain_stage):
    """Test the convergence analysis."""
    expected_results = np.array(
        [
            [
                1.820821,
                1.772137,
                1.693184,
                1.646514,
                1.597176,
                1.564588,
                1.561149,
                1.587887,
                1.581941,
                1.59686,
                1.591971,
                1.596187,
                1.621205,
                1.625246,
                1.626665,
                1.626075,
                1.629184,
                1.639542,
                1.629151,
                1.624827,
            ],
            [
                1.767312,
                1.721075,
                1.672817,
                1.668589,
                1.64544,
                1.67515,
                1.706788,
                1.721461,
                1.742795,
                1.74792,
                1.773763,
                1.769854,
                1.762897,
                1.779876,
                1.78781,
                1.802843,
                1.804173,
                1.805945,
                1.806565,
                1.803477,
            ],
            [
                1.453422,
                1.343316,
                1.375671,
                1.393149,
                1.358901,
                1.364104,
                1.371496,
                1.383695,
                1.395921,
                1.382649,
                1.406549,
                1.412048,
                1.413439,
                1.413896,
                1.416637,
                1.413021,
                1.424004,
                1.426646,
                1.430358,
                1.426475,
            ],
            [
                1.359394,
                1.367384,
                1.370174,
                1.415366,
                1.429112,
                1.432025,
                1.437935,
                1.427675,
                1.419545,
                1.413919,
                1.401691,
                1.406581,
                1.402699,
                1.411552,
                1.412378,
                1.41328,
                1.414688,
                1.412149,
                1.412207,
                1.41811,
            ],
            [
                1.537087,
                1.639381,
                1.656298,
                1.610339,
                1.622517,
                1.624194,
                1.639976,
                1.684027,
                1.696781,
                1.717429,
                1.720163,
                1.711127,
                1.717504,
                1.712591,
                1.704241,
                1.711488,
                1.709887,
                1.710316,
                1.716876,
                1.716112,
            ],
        ]
    )
    stage = restrain_stage
    _, free_energies = stage.analyse_convergence()
    assert np.allclose(free_energies, expected_results, atol=1e-2)
