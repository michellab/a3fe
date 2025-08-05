import os
import shutil
import subprocess
from a3fe.run._virtual_queue import VirtualQueue
import a3fe as a3
from a3fe.run.enums import LegType as _LegType
from a3fe.run.system_prep import SystemPreparationConfig
import re
import logging
import time
from datetime import datetime
from a3fe.run.simulation import Simulation
from a3fe.run._virtual_queue import Job
from a3fe.run.enums import JobStatus as _JobStatus

# Configuration options
FORCE_LOCAL_EXECUTION = True  # Set to False for normal SLURM execution
FORCE_CPU_PLATFORM = False   # Set to True to force CPU even on GPU systems
FAST_UPDATE_INTERVAL = 3  # seconds between updates for local execution


class DedupStatusFilter(logging.Filter):

    """
    we may want to de-duplicate massive logging info like:

    INFO - 2025-08-02 22:28:36,711 - Simulation (stage=discharge, lam=0.0, run_no=1)_70 - Not running
    INFO - 2025-08-02 22:28:36,719 - Simulation (stage=discharge, lam=0.0, run_no=1)_70 - Job 
     (virtual_job_id = 1, slurm_job_id= 999999), status = JobStatus.FINISHED finished successfully
    INFO - 2025-08-02 22:28:36,719 - Simulation (stage=discharge, lam=1.0, run_no=1)_74 - Not running
    INFO - 2025-08-02 22:28:36,743 - Simulation (stage=discharge, lam=1.0, run_no=1)_74 - Job 
     (virtual_job_id = 2, slurm_job_id= 999999), status = JobStatus.FINISHED finished successfully

    so that we only log out info when the status changes.
    """
    JOBID_RE = re.compile(r"slurm_job_id=\s*(\d+)")
    STATUS_RE = re.compile(r"status\s*=\s*([^,)+]+)")

    def __init__(self):
        super().__init__()
        self._last_status_by_jobid: dict[str, str] = {}  # jobid -> last seen status
        self._last_not_running_by_logger: dict[str, bool] = {}  # logger name -> saw "Not running"

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        name = record.name

        if "Not running" in msg:
            if self._last_not_running_by_logger.get(name, False):
                return False
            self._last_not_running_by_logger[name] = True
            return True

        if "status =" in msg:
            jobid_m = self.JOBID_RE.search(msg)
            status_m = self.STATUS_RE.search(msg)
            if jobid_m and status_m:
                jobid = jobid_m.group(1)
                status = status_m.group(1).strip()
                prev = self._last_status_by_jobid.get(jobid)
                if prev == status:
                    return False
                self._last_status_by_jobid[jobid] = status
        return True
    

def add_filter_recursively(sim_runner):
    if hasattr(sim_runner, "_logger"):
        sim_runner._logger.addFilter(DedupStatusFilter())
    if hasattr(sim_runner, "_sub_sim_runners"):
        for sub in sim_runner._sub_sim_runners:
            add_filter_recursively(sub)
            

def _parse_sim_info_from_job(job) -> str:
    """
    Job.command_list is like:
      ['--chdir', '/Users/jingjinghuang/Documents/fep_workflow/
        test_somd_run_again2_copy1/bound/vanish/output/lambda_0.000/run_01', 
        '/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2_copy1/
        bound/vanish/output/lambda_0.000/run_01/run_somd.sh', '0.0']
    """
    stage = "?"
    lam = "?"
    run_no = "?"

    # Lambda is last element if numeric
    try:
        potential_lam = job.command_list[-1]
        float(potential_lam)
        lam = potential_lam
    except Exception:
        pass

    # Get cwd from --chdir
    cwd = None
    if "--chdir" in job.command_list:
        idx = job.command_list.index("--chdir")
        if idx + 1 < len(job.command_list):
            cwd = job.command_list[idx + 1]

    if isinstance(cwd, str):
        # stage: e.g., .../bound/vanish/output/...
        m_stage = re.search(r"/(?:bound|free)/([^/]+)/output/", cwd)
        if m_stage:
            stage = m_stage.group(1)

        # run number: e.g., .../run_01
        m_run = re.search(r"run_(\d+)", cwd)
        if m_run:
            run_no = m_run.group(1)

    return f"stage={stage}, lam={lam}, run_no={run_no}"


def patch_virtual_queue_for_local_execution(): 
    """
    Patch VirtualQueue to run jobs locally instead of through SLURM.
    Works on both local machines and HPC systems.
    """
    # Check if we should use local execution
    use_local = FORCE_LOCAL_EXECUTION or (shutil.which("squeue") is None)
    
    if not use_local:
        print("SLURM detected and local execution not forced. Using normal SLURM submission.")
        return
    
    # Detect GPU availability
    print("Patching VirtualQueue for local execution...")
    print(f"Force CPU: {FORCE_CPU_PLATFORM}")

    # Silence subprocess calls (for ln commands and other system calls)
    original_call = subprocess.call
    def quiet_call(*args, **kwargs):
        if args and isinstance(args[0], list) and len(args[0]) > 0:
            if args[0][0] == "ln":
                kwargs.setdefault('stdout', subprocess.DEVNULL)
                kwargs.setdefault('stderr', subprocess.DEVNULL)
            elif args[0][0] in ["mkdir", "cp", "mv", "rm"]:
                kwargs.setdefault('stdout', subprocess.DEVNULL)
                kwargs.setdefault('stderr', subprocess.DEVNULL)
        return original_call(*args, **kwargs)
    
    # Apply the subprocess patch
    subprocess.call = quiet_call
    
    # Mock SLURM queue reading (always return empty queue)
    VirtualQueue._read_slurm_queue = lambda self: []
    
    # Replace job submission with local execution
    def _submit_locally(self, job_command_list):
        """Submit job locally instead of through SLURM."""
        # Get the working directory from the command list
        cwd = None
        if "--chdir" in job_command_list:
            idx = job_command_list.index("--chdir")
            cwd = job_command_list[idx + 1]
        real_cwd = cwd or os.getcwd()

        # Find the script to execute
        script_idx = next(
            (j for j, tok in enumerate(job_command_list) if tok.endswith(".sh")),
            None 
        )
        
        if script_idx is None:
            raise RuntimeError(f"No .sh script found in command: {job_command_list!r}")
        
        script_file = job_command_list[script_idx]
        script_path = os.path.join(cwd or os.getcwd(), script_file)
        
        # Check if this is a SOMD simulation (has lambda argument)
        if len(job_command_list) > script_idx + 1:
            lam_arg = job_command_list[-1]  # Lambda value

            # ——— EARLY SKIP if Simulation.log shows FINISHED ———
            sim_log = os.path.join(real_cwd, "Simulation.log")
            exe_log = os.path.join(real_cwd, "local_execution.log")
            if os.path.exists(sim_log) and os.path.exists(exe_log):
                with open(sim_log, "rb") as f:
                    # only read the last few KB for speed
                    f.seek(max(0, os.path.getsize(sim_log) - 4096))
                    sim_tail = f.read().decode(errors="ignore").splitlines()
                sim_ok = any("status = JobStatus.FINISHED finished successfully" in L
                             for L in sim_tail)

                with open(exe_log, "r") as f:
                    local_content = f.read()
                exe_ok = "Job completed successfully" in local_content

                if sim_ok and exe_ok:
                    with open(exe_log, "a") as f:
                        f.write(f"[LOCAL SOMD] SKIPPED lambda={lam_arg} at "
                                f"{datetime.now():%Y-%m-%d %H:%M:%S}\n")
                    print(f"[LOCAL SOMD] Already finished in {real_cwd}; SKIPPING")
                    # Return a fake job ID that will immediately be marked as finished
                    return 999999
                
            return _run_somd_locally(script_path, lam_arg, cwd)
        else:
            # This is a preparation step
            return _run_prep_locally(script_path, cwd)
    
    def _run_somd_locally(script_path, lam_arg, cwd) -> int:
        """Run SOMD simulation locally.
           note somd.cfg may be easily corrupated by the original a3fe code. 
        """
        real_cwd = cwd or os.path.dirname(script_path)
        cfg_path = os.path.join(real_cwd, "somd.cfg")

        if not os.path.exists(cfg_path) or os.path.getsize(cfg_path) == 0:
            print(f"[LOCAL SOMD] ❌ {cfg_path} is missing or empty in {real_cwd}; cannot run SOMD")
            with open(os.path.join(real_cwd, "local_execution.log"), "a") as flog:
                flog.write(f"[LOCAL SOMD] ❌ JOB FAILED {cfg_path} is missing or empty in {real_cwd}; cannot run SOMD\n")
            raise RuntimeError(f"somd.cfg is missing or empty in {real_cwd}; cannot run SOMD")

        # Record start time
        start_time = time.time()
        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[LOCAL SOMD] {start_timestamp} Running lambda={lam_arg} in {cwd or os.getcwd()}")
        
        # Read the script to find the somd command
        somd_command = None
        try:
            with open(script_path) as f:
                for line in f:
                    line = line.strip()
                    if "somd-freenrg" in line:
                        # Extract the somd command (remove srun prefix if present)
                        if line.startswith("srun "):
                            somd_command = line.split(None, 1)[1]
                        else:
                            somd_command = line
                        break
        except FileNotFoundError:
            log_path = os.path.join(real_cwd, "local_execution.log")
            with open(log_path, "a") as flog:
                flog.write(f"[LOCAL SOMD] ❌ JOB FAILED missing script {script_path}\n")
            # now propagate up so you still surface the error
            raise 

        if not somd_command:
            raise RuntimeError(f"No somd-freenrg command found in {script_path}")
        
        # Parse and modify the command
        parts = somd_command.split()
        
        # Smart platform selection
        if "-p" in parts:
            p_idx = parts.index("-p")
            if p_idx + 1 < len(parts):
                current_platform = parts[p_idx + 1].upper()
                
                if FORCE_CPU_PLATFORM:
                    # Force CPU regardless of hardware
                    if current_platform != "CPU":
                        parts[p_idx + 1] = "CPU"
                        print("[LOCAL SOMD] Forced platform to CPU")
                else:
                    # Keep whatever platform was specified
                    print(f"[LOCAL SOMD] Using {current_platform} platform")
        
        # Substitute lambda value
        parts = [tok.replace("$lam", lam_arg).replace("${lam}", lam_arg) for tok in parts]
        
        print(f"[LOCAL SOMD] Executing: {' '.join(parts)} at {real_cwd}")
        
        try:
            subprocess.run(parts, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)

            end_time = time.time()
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration_seconds = end_time - start_time

            print(f"[LOCAL SOMD] ✅ {end_timestamp} Completed successfully for lambda={lam_arg}")
            print(f"[LOCAL SOMD] Simulation took {duration_seconds:.2f} seconds")

            # Create a local_execution.log file that mimics SLURM output
            local_execution_log_path = os.path.join(real_cwd, "local_execution.log")
            with open(local_execution_log_path, 'a') as f:
                f.write(f"[LOCAL SOMD] Starting lambda={lam_arg} at {start_timestamp}\n")
                f.write(f"[LOCAL SOMD] Completed lambda={lam_arg} at {end_timestamp}\n")
                f.write(f"Simulation took {duration_seconds:.2f} seconds\n")
                f.write(f"✅ Job completed successfully\n")
            return 888888  # Return fake job ID on success
        
        except subprocess.CalledProcessError as e:
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[LOCAL SOMD] ❌ JOB FAILED with return code {e.returncode}")
            print(f"[LOCAL SOMD] ❌ STDERR:\n{e.stderr}")

            # --- write a “failed” marker into the same execution log ---
            local_execution_log_path = os.path.join(real_cwd, "local_execution.log")
            with open(local_execution_log_path, "a") as f:
                f.write(f"[LOCAL SOMD] {end_timestamp} ❌ JOB FAILED with return code {e.returncode}\n")
                f.write(f"[LOCAL SOMD] ❌ STDERR: {e.stderr}\n")

            raise RuntimeError(f"SOMD simulation failed: {e}")
    

    def _run_prep_locally(script_path, cwd) -> int:
        """Run preparation step locally."""
        print(f"[LOCAL PREP] Running preparation script in {cwd or os.getcwd()}")
        
        # Read the script to find the python command
        python_command = None
        with open(script_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("python -c") and "a3fe.run.system_prep" in line:
                    python_command = line
                    break
        
        if not python_command:
            raise RuntimeError(f"No A3FE preparation command found in {script_path}")
        
        print(f"[LOCAL PREP] Executing: {python_command} at {cwd or os.getcwd()}")
        
        try:
            subprocess.run(python_command, shell=True, cwd=cwd, check=True)
            print(f"[LOCAL PREP] ✅ Completed successfully")
            return 888888 # Return fake job ID
        except subprocess.CalledProcessError as e:
            print(f"[LOCAL PREP] ❌ Failed with return code {e.returncode}")
            raise RuntimeError(f"Preparation step failed: {e}")
    

    def timing_based_get_tot_gpu_time(self) -> float:
        """need to get tot_gpu_time to set relative_simulation_cost which is 
           a must for adaptive runtime mode

           however, we set set_relative_sim_cost=False for get_optimal_lam_vals()
              so this method will not be called in the first place.
        """
        # First check for local execution log
        timing_log_path = os.path.join(self.base_dir, "local_execution.log")
        if os.path.exists(timing_log_path):
            try:
                with open(timing_log_path, 'r') as f:
                    content = f.read()
                    # Look for "Simulation took X.XX seconds"
                    match = re.search(r"Simulation took ([\d.]+) seconds", content)
                    if match:
                        seconds = float(match.group(1))
                        hours = seconds / 3600
                        return hours
            except Exception as e:
                print(f"[ERROR] ❌ get tot_gpu_time - Failed to read local execution: {e}")
        else:
            print(f"[ERROR] ❌ get tot_gpu_time - Local execution log not found at {timing_log_path}")


    def local_slurm_outfile(self):
        """Mock slurm outfile property for local execution."""
        if hasattr(self, '_local_outfile'):
            return self._local_outfile
    
        cwd = None
        if "--chdir" in self.command_list:
            idx = self.command_list.index("--chdir")
            cwd = self.command_list[idx+1]
        log_dir = cwd or os.getcwd()
        log_file = os.path.join(log_dir, "local_execution.log")

        os.makedirs(log_dir, exist_ok=True)
        open(log_file, "a").close()
        
        self._local_outfile = log_file
        return log_file
    
    def local_has_failed(self):
        """Mock has_failed method for local execution."""
        try:
            log_file = self.slurm_outfile   # points at local_execution.log
            with open(log_file, "r") as f:
                return "JOB FAILED" in f.read()
        except Exception:
            return False  # no log yet → not failed

    # Override the update method to handle fake job IDs
    original_update = VirtualQueue.update
    def local_update(self) -> None:
        """Updated update method that handles local execution fake job IDs."""
        # Define fake job IDs used for already-completed or local jobs
        fake_job_ids = [888888, 999999]  # 888888 from actual execution, 999999 from early skip
        fake_fail_ids = [777777]
        
        # Mark any jobs with fake IDs as finished and remove from queue
        jobs_to_remove = []
        for job in self._slurm_queue:
            # grab more info like command list, etc.
            job_sim_info = _parse_sim_info_from_job(job)

            if job.slurm_job_id in fake_job_ids:
                if getattr(job, "_already_marked_finished", False):
                    continue
                job.status = _JobStatus.FINISHED
                job._already_marked_finished = True  # flag to prevent repeated downstream prints
                jobs_to_remove.append(job)
                print(f"[LOCAL UPDATE] ✅ Marking job slurm_job_id={job.slurm_job_id}, {job_sim_info} as finished")
            elif job.slurm_job_id in fake_fail_ids:
                # leave it in the queue so downstream sees it as FAILED
                # (local_has_failed will now return True)
                job.status = _JobStatus.FAILED
                continue

        # Remove the completed local jobs
        for job in jobs_to_remove:
            self._slurm_queue.remove(job)
        # Call original update for any real SLURM jobs (if any)
        original_update(self)

    
    # Reduce VirtualQueue logging verbosity by patching the submit method
    def quiet_submit(self, command_list, slurm_file_base):
        """Submit method without the 'submitted' logging message."""
        virtual_job_id = self._available_virt_job_id
        self._available_virt_job_id += 1
        job = Job(virtual_job_id, command_list, slurm_file_base=slurm_file_base)
        job.status = _JobStatus.QUEUED
        self._pre_queue.append(job)
        # Remove this line: self._logger.info(f"{job} submitted")
        self.update()
        return job
    

    # APPLY THE PATCHES NOW
    Simulation.get_tot_gpu_time = timing_based_get_tot_gpu_time

    VirtualQueue.update = local_update
    VirtualQueue._submit_job = _submit_locally

    Job.slurm_outfile = property(local_slurm_outfile)
    Job.has_failed = local_has_failed

    VirtualQueue.submit = quiet_submit

    print("VirtualQueue successfully patched for local execution!")



if __name__ == "__main__":
    # Configure via environment variables
    FORCE_LOCAL_EXECUTION = True
    FORCE_CPU_PLATFORM = True
    
    patch_virtual_queue_for_local_execution()
    
    # # Set global defaults before creating any Leg instances
    # for step in ["parameterise", "solvate", "minimise", "heat_preequil", "ensemble_equil"]:
    #     a3.Leg.update_default_slurm_config(
    #         step_type=step,
    #         time="12:00:00",
    #         gres="",  # switch to CPU-only
    #         pre_commands=['export PATH="$CONDA_PREFIX/bin:$PATH"']
    #     )

    a3.Calculation.required_legs = [_LegType.BOUND]
                                    
    sysprep_cfg = SystemPreparationConfig(slurm=True)
                                        #runtime_short_nvt=5,
                                        #runtime_nvt=10,
                                        #runtime_npt=10,                   # added for local test run on mac; unit - ps
                                        #runtime_npt_unrestrained=10,      # added for local test run on mac; unit - ps
                                        #ensemble_equilibration_time=10,)  # added for local test run on mac; unit - ps

    calc = a3.Calculation(ensemble_size=3, 
                      base_dir="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again5",
                      input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again5/input")

    calc.setup(
        bound_leg_sysprep_config=sysprep_cfg,
        free_leg_sysprep_config=sysprep_cfg,
        skip_preparation=True,  # skip system preparation
    )

    add_filter_recursively(calc)

    # calc.bound_leg.update_slurm_script(
    #     "somd_production",
    #     mem="2G",             
    #     time="00:13:13",
    #     gres="", # switch to CPU-only
    #     setup_cuda_env=False,       # Disable CUDA environment setup
    #     somd_platform="CPU"         # Use CPU instead of CUDA 
    # )
    # by default use simtime=0.1 ns
    # we might need to reduce delta_er to get more lambda windows
    calc.get_optimal_lam_vals() 
    calc.run(adaptive=False, 
             runtime=25,                  # run non-adaptively for 25 ns per replicate
             parallel=False)              # run things sequentially
    calc.wait()
    calc.set_equilibration_time(1)        # Discard the first ns of simulation time
    calc.analyse()
    calc.save()

