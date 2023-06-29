import os
import shutil

# Check if slurm is present
SLURM_PRESENT = False if shutil.which("sbatch") is None else True

# See if the user wants to skip the slurm tests
RUN_SLURM_TESTS = True
if os.environ.get("RUN_SLURM_TESTS") == "False":
    RUN_SLURM_TESTS = False
