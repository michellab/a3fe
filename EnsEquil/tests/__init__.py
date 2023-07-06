import os
import shutil

# Check if slurm is present
SLURM_PRESENT = False if shutil.which("sbatch") is None else True

# Check if gromacs is present
GROMACS_PRESENT = False if shutil.which("gmx") is None else True

# See if the user wants to skip the slurm tests
RUN_SLURM_TESTS = True
if os.environ.get("RUN_SLURM_TESTS") == "False":
    RUN_SLURM_TESTS = False

# Make sure that we're in the correct directory and raise an error if not
if not os.path.exists("EnsEquil"):
    raise RuntimeError(
        "Please run the tests from the root directory of the repository."
    )
