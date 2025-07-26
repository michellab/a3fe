"""
Script used to run an example calculation on Graham HPC cluster.
"""

import os
import shutil
import subprocess
from a3fe.run.enums import LegType as _LegType
import a3fe as a3
from a3fe.run._virtual_queue import VirtualQueue
from a3fe.run.system_prep import SystemPreparationConfig

# sysprep_cfg = SystemPreparationConfig(
#     mdrun_options="-ntmpi 1 -ntomp 1",
#     runtime_short_nvt=5,
#     runtime_nvt=10,
#     runtime_npt=10,  # added for local test run on mac; unit - ps
#     runtime_npt_unrestrained=10,  # added for local test run on mac; unit - ps
#     ensemble_equilibration_time=10,
# )  # added for local test run on mac; unit - ps

a3.Calculation.required_legs = [_LegType.BOUND]

print('step-1: Initializing calculation...')
# This is needed to run the preparation steps via SLURM
for step in ["parameterise", "solvate", "minimise", "heat_preequil", "ensemble_equil"]:
    a3.Leg.update_default_slurm_config(
        step_type=step,
        pre_commands=['export PATH="$CONDA_PREFIX/bin:$PATH"']
    )

# initialize the calculation
calc = a3.Calculation(
    ensemble_size=3,
    base_dir="/home/jjhuang/project/jjhuang/fep_workflows/test_run_full/",
    input_dir="/home/jjhuang/project/jjhuang/fep_workflows/test_run_full/input",
)
print("step-2: Setting up calculation...")
calc.setup()
# we could update the slurm script for steps here
calc.bound_leg.update_slurm_script(
    "somd_production",
    mem="2G",             
    time="00:20:20"       
)
print('step-3: Get optimal lambda values...')
calc.get_optimal_lam_vals()
print('step-4...')
calc.run(
    parallel=False,
    adaptive=False,
) 
print('step-5...')
calc.wait()
print('step-6...')
calc.set_equilibration_time(1)  # Discard the first ns of simulation time
print('step-7...')
calc.analyse()
calc.save()
