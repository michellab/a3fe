import os
import shutil
import subprocess
from typing import Optional

import BioSimSpace.Sandpit.Exscientia as BSS

import a3fe

# Check if slurm is present
SLURM_PRESENT = False if shutil.which("sbatch") is None else True

# Check if gromacs is present
GROMACS_PRESENT = False if shutil.which("gmx") is None else True

# See if the user wants to run the slurm tests
RUN_SLURM_TESTS = os.environ.get("RUN_SLURM_TESTS") not in [None, "False", "0"]

# Make sure that we're in the correct directory and raise an error if not
if not os.path.exists("a3fe"):
    raise RuntimeError(
        "Please run the tests from the root directory of the repository."
    )

# Globally mock the run_process function so that we can test the setup stages without
# actually running. This is a temporary fix until we figure out how to do the mocking
# properly with fixtures.
original_run_process = a3fe.run.system_prep.run_process

# Create a complex system object to return from the mock run_process function
base_path = os.path.join("a3fe", "data", "example_run_dir", "bound", "input")
complex_sys = BSS.IO.readMolecules(
    [
        os.path.join(base_path, file)
        for file in ["bound_preequil.prm7", "bound_preequil.rst7"]
    ]
)


def mock_run_process(
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


# Now patch globally
a3fe.run.system_prep.run_process = mock_run_process
