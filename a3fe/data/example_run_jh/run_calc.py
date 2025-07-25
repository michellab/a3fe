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

print('step-1...')
# initialize the calculation
calc = a3.Calculation(
    ensemble_size=3,
    base_dir="/home/jjhuang/project/jjhuang/fep_workflows/test_run_full/",
    input_dir="/home/jjhuang/project/jjhuang/fep_workflows/test_run_full/input",
)
print("step-2...")
calc.setup()
# we can update the slurm script for steps here
calc.bound_leg.update_slurm_script(
    "parameterise", 
    time="00:12:12",     
    mem="1G",           
    cpus_per_task=1       
)
calc.bound_leg.update_slurm_script(
    "solvate",
    mem="8G",             
    time="00:11:11"       
)
calc.bound_leg.update_slurm_script(
    "somd_production",
    mem="2G",             
    time="00:13:13"       
)
print('step-3...')
# get optimal lambda spacing
calc.get_optimal_lam_vals()
print('step-4...')
calc.run(
    parallel=False,
)  # run things sequentially
print('step-5...')
calc.wait()
print('step-6...')
calc.set_equilibration_time(1)  # Discard the first ns of simulation time
print('step-7...')
calc.analyse()
calc.save()
