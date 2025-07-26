import shutil
import subprocess
import a3fe as a3
import os
from a3fe.run.system_prep import SystemPreparationConfig
from a3fe.run._virtual_queue import VirtualQueue
from a3fe.run.enums import LegType as _LegType

print("mdrun_options" in SystemPreparationConfig.model_fields)

# monkey-patch for local run without using slurm 
if shutil.which("squeue") is None:

    VirtualQueue._read_slurm_queue = lambda self: []

    # replace sbatch + script with "direct somd-freenrg" invocation
    def _submit_locally(self, job_command_list):
        print('job_command_list-----------------------', job_command_list)
        # pick off --chdir so we can cwd into it
        cwd = None
        if "--chdir" in job_command_list:   # "sbatch --chdir a/b/c/d" -> cd into a specific dir 
            i = job_command_list.index("--chdir")
            cwd = job_command_list[i + 1]   # a/b/c/d

        # find the .sh script in the args 
        # this is for submitting all *.sh scripts in the folder, even though it seems there should be only
        # one "run_somd.sh" 
        script_idx = next(
            (j for j, tok in enumerate(job_command_list) if tok.endswith(".sh")),
            None 
        )
        if script_idx is None:
            raise RuntimeError(f"No .sh launcher in {job_command_list!r}")

        script   = job_command_list[script_idx]   # or we can simply try script = run_somd.sh
        lam_arg  = job_command_list[-1]  # e.g. "0.0", defined in Simulation.run() 
        script_path = os.path.join(cwd or os.getcwd(), script)

        # scan the script for the srun line
        with open(script_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("srun "):
                    # drop the "srun " prefix
                    cmdline = line.split(None, 1)[1]
                    break
            else:
                raise RuntimeError(f"No srun line in {script_path}")

        # split into args, substitute $lam
        parts = cmdline.split()
        # change "-p CUDA" to "-p " 
        if "-p" in parts:
            p = parts.index("-p")
            # ensure there’s an argument after "-p"
            if p + 1 < len(parts):
                # normalize case and check for "cuda"
                if parts[p+1].lower() == "cuda":
                    parts[p+1] = "CPU"

        parts = [tok.replace("$lam", lam_arg) for tok in parts]

        print(f"[VirtualQueue→LOCAL] cwd={cwd or os.getcwd()}  →  {' '.join(parts)}")
        subprocess.run(parts, cwd=cwd, check=True)
        return 0  # pretend we got job ID 0

    VirtualQueue._submit_job = _submit_locally


# Set global defaults before creating any Leg instances
for step in ["parameterise", "solvate", "minimise", "heat_preequil", "ensemble_equil"]:
    a3.Leg.update_default_slurm_config(
        step_type=step,
        pre_commands=['export PATH="$CONDA_PREFIX/bin:$PATH"']
    )

a3.Calculation.required_legs = [_LegType.BOUND]
                                
sysprep_cfg = SystemPreparationConfig(slurm=False,
                                      mdrun_options="-ntmpi 1 -ntomp 1",
                                      runtime_short_nvt=5,
                                      runtime_nvt=10,
                                      runtime_npt=10,                   # added for local test run on mac; unit - ps
                                      runtime_npt_unrestrained=10,      # added for local test run on mac; unit - ps
                                      ensemble_equilibration_time=10,)  # added for local test run on mac; unit - ps

print('step-1...')
calc = a3.Calculation(ensemble_size=1, 
                      base_dir="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2",
                      input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2/input")
print("step-2:...")

calc.setup(
    bound_leg_sysprep_config=sysprep_cfg,
    free_leg_sysprep_config=sysprep_cfg,
    # skip_preparation=True,  # skip system preparation
)

calc.bound_leg.update_slurm_script(
    "somd_production",
    mem="2G",             
    time="00:13:13"       
)

print('step-3...')
# calc.get_optimal_lam_vals()
print('step-4...')
# calc.run(adaptive=False, runtime=5,   # run non-adaptively for 5 ns per replicate
#          parallel=False)              # run things sequentially
print('step-5...')
# calc.wait()
print('step-6...')
# calc.set_equilibration_time(1)        # Discard the first ns of simulation time
print('step-7...')
# calc.analyse()
# calc.save()
