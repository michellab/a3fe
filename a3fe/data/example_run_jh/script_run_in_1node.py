import os
os.environ["MPLBACKEND"] = "Agg"  # prevents macOS GUI backend from opening windows
try:
    import matplotlib as _mpl
    _mpl.use("Agg", force=True)
except Exception:
    pass

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
from a3fe.run.stage import Stage
from a3fe.run._simulation_runner import SimulationRunner
from a3fe.run._virtual_queue import Job
from a3fe.run.enums import JobStatus as _JobStatus
from time import sleep
from functools import lru_cache
import concurrent.futures
import itertools
import subprocess
from tqdm import tqdm
import sys
import shlex


# Configuration options
FORCE_LOCAL_EXECUTION = True  # Set to False for normal SLURM execution
FORCE_CPU_PLATFORM = False   # Set to True to force CPU even on GPU systems
FAST_UPDATE_INTERVAL = 3  # seconds between updates for local execution
SKIP_ADAPTIVE_EFFICIENCY = False  # Set to True to skip adaptive efficiency checks


# --- set up colored logging ---
class ColorFormatter(logging.Formatter):
    ORANGE = "\033[33m"  # ANSI “yellow” as an orange stand‐in
    GREEN  = "\033[32m"
    RED    = "\033[31m"
    RESET  = "\033[0m"

    FORMATS = {
        logging.DEBUG: ORANGE + "%(levelname)s: %(message)s" + RESET,
        logging.INFO:  GREEN  + "%(levelname)s: %(message)s" + RESET,
        logging.ERROR: RED    + "%(levelname)s: %(message)s" + RESET,
        logging.WARNING: RED    + "%(levelname)s: %(message)s" + RESET,

    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, "%(levelname)s: %(message)s")
        return logging.Formatter(fmt).format(record)


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
    STATUS_RE = re.compile(r"status\s*=\s*(JobStatus\.\w+)")
    SIM_DETAILS_RE = re.compile(r"Simulation \(stage=([^,]+), lam=([^,]+), run_no=([^)]+)\)")

    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.debug_mode = debug_mode
        self.suppress_mbar_noise: bool = False
        self._mbar_noise = re.compile(
            r"(?:Submitted MBAR job \d+:|\[LOCAL UPDATE\].*MBAR job \d+.*|MBAR job \d+ (?:running|completed successfully|failed))"
        )
        self._last_status_by_job: dict[str, str] = {}  # unique_job_key -> last seen status
        self._not_running_jobs: set[str] = set()  # Track which jobs have logged "Not running"
    
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        name = record.name

        if self.debug_mode:
            print(f"DEBUG: Processing message from {name}: {msg[:100]}...")
        
        # For "Not running" messages, try to identify which job this is about
        if "Not running" in msg:
            job_key = self._get_job_key(msg, name)  # No jobid for "Not running"
            
            if job_key and job_key in self._not_running_jobs:
                if self.debug_mode:
                    print(f"DEBUG: Suppressing duplicate 'Not running' for job {job_key}")
                return False            
            if job_key:
                self._not_running_jobs.add(job_key)
                if self.debug_mode:
                    print(f"DEBUG: First 'Not running' for job {job_key}, allowing")
            else:
                if self.debug_mode:
                    print(f"DEBUG: Allowing 'Not running' (couldn't identify job)")
            return True

        # For status messages
        if "status =" in msg:
            jobid_m = self.JOBID_RE.search(msg)
            status_m = self.STATUS_RE.search(msg)
            
            if jobid_m and status_m:
                jobid = jobid_m.group(1)
                status = status_m.group(1).strip()
                
                unique_job_key = self._get_job_key(msg, name, jobid)
                prev_status = self._last_status_by_job.get(unique_job_key)

                if self.debug_mode:
                    print(f"DEBUG: Job key: '{unique_job_key}', Status: '{status}' (was: '{prev_status}')")
                
                if prev_status == status:
                    if self.debug_mode:
                        print(f"DEBUG: Suppressing duplicate status for {unique_job_key}")
                    return False
                
                self._last_status_by_job[unique_job_key] = status
                # If this job is now running/finished, allow "Not running" to be logged again later
                if status in ["JobStatus.FINISHED", "JobStatus.FAILED", "JobStatus.KILLED"]:
                    self._not_running_jobs.discard(unique_job_key)
                    if self.debug_mode:
                        print(f"DEBUG: Job {unique_job_key} finished, allowing future 'Not running'")

        if self.suppress_mbar_noise and self._mbar_noise.search(msg):
            return False

        if self.debug_mode:
            print(f"DEBUG: Allowing message through")
        return True
    
    @lru_cache(maxsize=1000) 
    def _get_job_key(self, msg: str, logger_name: str, jobid: str = None) -> str | None:
        sim_details_m = self.SIM_DETAILS_RE.search(msg)
        if not sim_details_m:
            sim_details_m = self.SIM_DETAILS_RE.search(logger_name)
        if sim_details_m:
            stage = sim_details_m.group(1)
            lam = sim_details_m.group(2)
            run_no = sim_details_m.group(3)            
            if jobid:
                return f"{jobid}:{stage}:{lam}:{run_no}"
            else:
                return f"current:{stage}:{lam}:{run_no}"
        else:
            if jobid:
                return f"{jobid}:{logger_name}"
            else:
                return None


# Create ONE instance and reuse it
shared_filter = DedupStatusFilter(debug_mode=False) # set to True for debugging


def setup_global_logging():
    """Set up global logging configuration with deduplication filter."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    
    # Create handler with color formatting
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    handler.addFilter(shared_filter)
    
    root_logger.addHandler(handler)
    
    # Patch SimulationRunner._set_up_logging
    if hasattr(SimulationRunner, '_set_up_logging'):
        original_sr_setup = SimulationRunner._set_up_logging
        def patched_sr_setup(self, null: bool = False):
            original_sr_setup(self, null)
            if hasattr(self, '_logger'):
                self._logger.addFilter(shared_filter)
                for handler in self._logger.handlers:
                    handler.addFilter(shared_filter)
        SimulationRunner._set_up_logging = patched_sr_setup
    
    # Patch VirtualQueue._set_up_logging
    if hasattr(VirtualQueue, '_set_up_logging'):
        original_vq_setup = VirtualQueue._set_up_logging
        def patched_vq_setup(self):
            original_vq_setup(self)
            if hasattr(self, '_logger'):
                self._logger.addFilter(shared_filter)
                for handler in self._logger.handlers:
                    handler.addFilter(shared_filter)
        VirtualQueue._set_up_logging = patched_vq_setup


def add_filter_recursively(obj, filter_instance=shared_filter):
    """Recursively add filter to all loggers in an object hierarchy."""
    if hasattr(obj, "_logger"):
        obj._logger.addFilter(filter_instance)
        for handler in obj._logger.handlers:
            handler.addFilter(filter_instance)
    
    # Handle different types of sub-objects
    sub_objects = []
    if hasattr(obj, "_sub_sim_runners") and obj._sub_sim_runners:
        sub_objects.extend(obj._sub_sim_runners)
    if hasattr(obj, "stages") and obj.stages:
        sub_objects.extend(obj.stages)
    if hasattr(obj, "lam_windows") and obj.lam_windows:
        sub_objects.extend(obj.lam_windows)
    if hasattr(obj, "sims") and obj.sims:
        sub_objects.extend(obj.sims)
    if hasattr(obj, "legs") and obj.legs:
        sub_objects.extend(obj.legs)
    if hasattr(obj, "virtual_queue"):
        sub_objects.append(obj.virtual_queue)
    
    # Recursively apply to sub-objects
    for sub_obj in sub_objects:
        add_filter_recursively(sub_obj, filter_instance)

    
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


def _format_sim_info(cwd: str, lam_arg: str = None) -> str:
    """
    Given a working directory (the --chdir path), an optional lambda string,
    and an optional restraint type, returns:
       "stage=<stage>, lam=<lam>, run_no=<run_no>[, restraint=<type>]"
    """
    # parse lam from argument or from cwd
    if lam_arg is None:
        m_lam = re.search(r"/lambda_([0-9.]+)/", cwd)
        lam_arg = m_lam.group(1) if m_lam else "?"
    # parse stage
    m_stage = re.search(r"/(?:bound|free)/([^/]+)/output/", cwd)
    stage = m_stage.group(1) if m_stage else "?"
    # parse run number
    m_run = re.search(r"/run_(\d+)", cwd)
    run_no = m_run.group(1) if m_run else "?"
    parts = [f"stage={stage}", f"lam={lam_arg}", f"run_no={run_no}"]
    return ", ".join(parts)


def _mbar_worker(cwd: str, mbar_command: str) -> tuple[int, str, str, float]:
    """
    Run MBAR command in a separate process with timing.
    Returns (returncode, stdout, stderr, duration). Never raises.
    """
    start_time = time.time()
    
    env = os.environ.copy()
    # prevent BLAS oversubscription when using multiprocessing
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    
    proc = subprocess.run(
        mbar_command, shell=True, cwd=cwd, check=False,
        capture_output=True, text=True, env=env
    )
    
    duration = time.time() - start_time
    return proc.returncode, proc.stdout, proc.stderr, duration


def _is_mbar_script(script_path) -> bool:
    """Check if a script is an MBAR analysis script."""
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        return "analyse_freenrg" in content or "freenrg-MBAR" in content
    except:
        return False


def _create_dummy_mbar_output(output_path: str, cwd: str) -> None:
    """Create a realistic dummy MBAR output file that matches the expected format."""
    lambda_files = []
    try:
        import glob
        lambda_dirs = glob.glob(os.path.join(cwd, "../lambda_*"))
        lambda_files = [f"'{os.path.join(dir, 'run_01/simfile_truncated_1.0_end_0.0_start.dat')}'" 
                        for dir in sorted(lambda_dirs)]
    except:
        lambda_files = ["'/path/to/lambda_0.000/run_01/simfile_truncated_1.0_end_0.0_start.dat'",
                        "'/path/to/lambda_1.000/run_01/simfile_truncated_1.0_end_0.0_start.dat'"]
    
    # Create dummy content that matches the real MBAR output format
    dummy_content = f"""# Analysing data contained in file(s) [{', '.join(lambda_files)}]
        # WARNING: This is a dummy MBAR output created due to insufficient simulation data
        # This is expected during early adaptive equilibration phases
        #Overlap matrix
        0.0000 0.0000
        0.0000 0.0000
        #DG from neighbouring lambda in kcal/mol
        0.0000 0.0000 0.0000 0.0000
        #PMF from MBAR in kcal/mol
        0.0000 0.0000 0.0000
        0.0000 0.0000 0.0000
        #TI average gradients and standard deviation in kcal/mol
        0.0000 0.0000 0.0000
        0.0000 0.0000 0.0000
        #PMF from TI in kcal/mol
        0.0000 0.0000
        0.0000 0.0000
        #MBAR free energy difference in kcal/mol: 
        0.000000, 0.000000  #WARNING DUMMY OUTPUT - INSUFFICIENT DATA FOR REAL MBAR ANALYSIS
        #TI free energy difference in kcal/mol: 
        0.000000  #WARNING DUMMY OUTPUT - INSUFFICIENT DATA FOR REAL MBAR ANALYSIS
        """
    with open(output_path, 'w') as f:
        f.write(dummy_content)


_FALLBACK_RE = re.compile(r"(freenrg-MBAR[^\s/]*?\.dat)\b")
def _extract_mbar_output_file(command: str) -> str | None:
    """
    Extract the MBAR output file basename from a CLI command string.

    a command may look like:
    analyse_freenrg mbar -i /project/6097686/jjhuang/fep_workflows/new_run_final_1/\
        bound/restrain/output/lambda*/run_02/simfile_truncated_66.0_end_65.0_start.dat\
            -p 100 --overlap -o \
        /project/6097686/jjhuang/fep_workflows/new_run_final_1/bound/restrain/output/\
            freenrg-MBAR-run_02_66.0_end_65.0_start.dat"
    """
    try:
        parts = shlex.split(command)  # safer than .split()
    except Exception:
        parts = command.split()

    output_value = None
    for i, tok in enumerate(parts):
        if tok in ("--output", "-o"):
            if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                output_value = parts[i + 1]
        elif tok.startswith("--output="):
            output_value = tok.split("=", 1)[1]

    if not output_value:
        m = _FALLBACK_RE.search(command)
        if m:
            output_value = m.group(1)

    return os.path.basename(output_value) if output_value else None


# ==================================================
# Global MBAR Manager for Parallel Execution
# ==================================================
class GlobalMBARManager:
    """Global manager for parallel MBAR execution with proper synchronization."""
    
    def __init__(self, max_workers: int = None, use_progress: bool = True):
        self.use_progress = use_progress
        if max_workers is None:
            # Use half the available cores to avoid oversubscription
            max_workers = max(1, (os.cpu_count() or 4) // 2)
        
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}  # job_id -> Future
        self.job_metadata = {}  # job_id -> {"cwd": str, "cmd": str, "script": str}
        self.job_counter = itertools.count(600000)
        self.expected_outputs = set() 
        self.logger = logging.getLogger(__name__ + ".MBAR_MANAGER")
        
    def submit_mbar_job(self, script_path: str, cwd: str) -> int:
        """Submit an MBAR job for parallel execution."""
        if hasattr(shared_filter, "suppress_mbar_noise"):
            shared_filter.suppress_mbar_noise = True
        # Parse the MBAR command from script
        try:
            with open(script_path) as f:
                script_content = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"MBAR script not found: {script_path}")

        mbar_command = None
        for line in script_content.splitlines():
            line = line.strip()
            if (("analyse_freenrg" in line) and 
                not line.startswith("#") and
                not line.startswith("export")):
                mbar_command = line
                break

        if not mbar_command:
            self.logger.warning(f"No explicit MBAR command found, running script directly")
            mbar_command = f"bash {script_path}"

        ofile = _extract_mbar_output_file(mbar_command)  # ofile can be None somehow
        if ofile:
            self.expected_outputs.add(os.path.join(cwd, ofile))

        # Submit to executor
        job_id = next(self.job_counter)
        future = self.executor.submit(_mbar_worker, cwd, mbar_command)
        
        self.futures[job_id] = future
        self.job_metadata[job_id] = {
            "cwd": cwd, 
            "cmd": mbar_command, 
            "script": script_path
        }
        self._log_mbar_start(cwd=cwd, command=mbar_command, job_id=job_id)
        
        self.logger.info(f"Submitted MBAR job {job_id}: {mbar_command}")
        return job_id
    
    def _log_mbar_start(self, cwd: str, command: str, job_id: int):
        """Log MBAR job start to local_execution.log"""
        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mbar_info = self._format_mbar_info(cwd, command)
        log_path = os.path.join(cwd, "local_execution.log")
        os.makedirs(cwd, exist_ok=True)  # although should already exist
        with open(log_path, "a") as f:
            f.write(f"[LOCAL MBAR] {start_timestamp} Starting MBAR job {job_id}: {mbar_info}\n")
            f.write(f"[LOCAL MBAR] Command: {command}\n")

    def _log_mbar_completion(self, cwd: str, job_id: int, success: bool, duration: float, error_msg: str = None):
        """Log MBAR job completion to local_execution.log"""
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = os.path.join(cwd, "local_execution.log")
        with open(log_path, "a") as f:
            if success:
                f.write(f"[LOCAL MBAR] {end_timestamp} ✅ MBAR job {job_id} completed in {duration:.2f} seconds\n")
            else:
                f.write(f"[LOCAL MBAR] {end_timestamp} ❌ MBAR job {job_id} failed (dummy output created)\n")
                f.write(f"[LOCAL MBAR] Error: {error_msg}\n")


    def _format_mbar_info(self, cwd: str, command: str) -> str:
        """Format MBAR job info for logging"""
        # Extract stage info from path
        stage_match = re.search(r"/(?:bound|free)/([^/]+)/output/", cwd)
        stage = stage_match.group(1) if stage_match else "unknown"
        
        # Extract output file
        output_file = _extract_mbar_output_file(command)
        
        return f"stage={stage}, output={output_file or 'unknown'}"

    def wait_for_completion(self):
        """Wait for all submitted MBAR jobs to complete (robust)."""
        if not self.futures:
            self.logger.info("No MBAR jobs to wait for")
            return

        self.logger.info(f"Waiting for {len(self.futures)} MBAR jobs to complete...")

        total = len(self.futures)
        future_to_id = {fut: jid for jid, fut in self.futures.items()}
        # Use a TTY-aware progress bar if available
        use_pb = self.use_progress and (tqdm is not None) and sys.stderr.isatty()
        ok = 0
        fail = 0
        if use_pb:
            bar = tqdm(total=total, desc="MBAR analyses", unit="job", leave=True, smoothing=0.1)
            # (optional) hide noisy “running/submitted” logs while bar is active
            if hasattr(shared_filter, "suppress_mbar_noise"):
                shared_filter.suppress_mbar_noise = True
        else:
            self.logger.info(f"Waiting for {total} MBAR jobs to complete...")
        
        for future in concurrent.futures.as_completed(self.futures.values()):
            job_id = future_to_id.get(future)
            if job_id is None:
                continue

            meta = self.job_metadata.get(job_id, {})
            cwd = meta.get("cwd", "")
            cmd = meta.get("cmd", "")
            # script = meta.get("script", "")

            try:
                rc, stdout, stderr, duration = future.result()
            except Exception as e:
                rc, stdout, stderr, duration = -1, "", str(e)

            # NOTE: Success must also produce a .dat file
            ofile = _extract_mbar_output_file(cmd)
            success = False
            if ofile is None:
                self.logger.error(f"MBAR job {job_id} did not produce an output file")
                rc = -1 
            else:
                # when ofile is not None
                ofile_path = os.path.join(cwd, ofile)
                if rc == 0 and os.path.exists(ofile_path) and os.path.getsize(ofile_path) > 0:
                    success = True
                    ok += 1
                    if not use_pb:
                        self.logger.info(f"MBAR job {job_id} completed successfully")
                else:
                    fail += 1
                    error_msg = stderr if stderr else "Output file missing or empty"
                    if not use_pb:
                        self.logger.warning(f"MBAR job {job_id} failed: {error_msg}")
            

            self._log_mbar_completion(cwd=cwd, job_id=job_id, success=success, duration=duration, 
                                      error_msg=stderr if not success else None)
            if use_pb:
                bar.update(1)
                bar.set_postfix_str(f"ok={ok} fail={fail}")


        if use_pb:
            bar.close()
            if hasattr(shared_filter, "suppress_mbar_noise"):
                shared_filter.suppress_mbar_noise = False

        self.futures.clear()
        self.job_metadata.clear()
        self.logger.info(f"All MBAR jobs completed (ok={ok}, fail={fail})")
       
    def has_pending_jobs(self) -> bool:
        """Check if there are any pending MBAR jobs."""
        return bool(self.futures)
    
    def get_job_status(self, job_id: int) -> str:
        """Get the status of a specific job."""
        if job_id not in self.futures:
            return "UNKNOWN"
        
        future = self.futures[job_id]
        if future.done():
            try:
                result = future.result()
                return "FINISHED"
            except:
                return "FAILED"
        else:
            return "RUNNING"

# Global instance
_GLOBAL_MBAR_MANAGER = None

def _install_mbar_barrier_wrapper(logger):
    import a3fe.analyse.mbar as mbar
    import a3fe.analyse.process_grads as process_grads
    import a3fe.analyse.detect_equil as detect_equil  # need this for equil analysis
    import a3fe.run.stage as stage # need this for calc.analyse()

    if not hasattr(mbar, "_original_collect_mbar_slurm"):
        mbar._original_collect_mbar_slurm = mbar.collect_mbar_slurm

    mbar_sync_in_progress = False
    def _collect_mbar_wrapper(*args, **kwargs):
        nonlocal mbar_sync_in_progress
        # Only say anything if there are outstanding MBAR futures
        has_pending = _GLOBAL_MBAR_MANAGER and _GLOBAL_MBAR_MANAGER.has_pending_jobs()
        if has_pending and not mbar_sync_in_progress:
            logger.info("[MBAR SYNC] collect_mbar_slurm called - waiting for all MBAR jobs to complete")
            mbar_sync_in_progress = True

        if has_pending:
            _GLOBAL_MBAR_MANAGER.wait_for_completion()

        # Safety net: ensure all expected outputs exist (create dummies if not)
        if _GLOBAL_MBAR_MANAGER:
            missing = []
            for ofile in list(_GLOBAL_MBAR_MANAGER.expected_outputs):
                if not os.path.exists(ofile):
                    # raise FileNotFoundError(f"Expected MBAR output missing: {ofile}")
                    missing.append(ofile)
                    _create_dummy_mbar_output(ofile, os.path.dirname(ofile))
                    logger.warning(f"[MBAR SYNC] Missing MBAR output; created dummy: {ofile}")

            if missing:
                logger.warning(f"[MBAR SYNC] {len(missing)} MBAR outputs were missing and replaced with dummies.")

        # Only print the completion line once per wave
        if has_pending and mbar_sync_in_progress:
            logger.info("[MBAR SYNC] All MBAR jobs completed - proceeding to collect results")
            mbar_sync_in_progress = False

        kwargs_modified = kwargs.copy()
        kwargs_modified['delete_outfiles'] = False
        return mbar._original_collect_mbar_slurm(*args, **kwargs_modified)

    # Replace on module
    mbar.collect_mbar_slurm = _collect_mbar_wrapper
    # Rebind any cached aliases
    if hasattr(process_grads, "_collect_mbar_slurm"):
        process_grads._collect_mbar_slurm = _collect_mbar_wrapper
    if hasattr(detect_equil, "_collect_mbar_slurm"):
        detect_equil._collect_mbar_slurm = _collect_mbar_wrapper
    if hasattr(stage, "_collect_mbar_slurm"):
        stage._collect_mbar_slurm = _collect_mbar_wrapper



def patch_virtual_queue_for_local_execution(use_faster_wait: bool = False): 
    """
    Patch VirtualQueue to run jobs locally instead of through SLURM.
    Works on both local machines and HPC systems.

    turn on use_faster_wait to speed up local testing by reducing wait time
    """
    global _GLOBAL_MBAR_MANAGER

    # Check if we should use local execution
    use_local = FORCE_LOCAL_EXECUTION or (shutil.which("squeue") is None)

    # Set up colored logger for this function
    logger = logging.getLogger(__name__ + ".LOCAL_SOMD")
    logger.handlers.clear()  # Clear any existing handlers
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't propagate to avoid duplicate messages    
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    handler.addFilter(shared_filter)
    logger.addHandler(handler)

    if not use_local:
        logger.info("SLURM detected and local execution not forced. Using normal SLURM submission.")
        return
    
    # Detect GPU availability
    logger.info("Patching VirtualQueue for local execution...")
    logger.info(f"Force CPU: {FORCE_CPU_PLATFORM}")

    # Initialize global MBAR manager
    _GLOBAL_MBAR_MANAGER = GlobalMBARManager()
    # APPLY THE MBAR PATCHES HERE
    _install_mbar_barrier_wrapper(logger)
    logger.info(f"MBAR parallel workers: {_GLOBAL_MBAR_MANAGER.max_workers}")

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
        """Submit job locally instead of through SLURM.
           note that mbar analysis is also submitted as a slurm job in A3FE
        """
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
                with open(sim_log, "r") as f:
                    sim_lines = f.read().splitlines()
                sim_ok = any("JobStatus.FINISHED finished successfully" in L
                             for L in sim_lines)

                with open(exe_log, "r") as f:
                    local_content = f.read()
                exe_ok = "Job completed successfully" in local_content

                if sim_ok and exe_ok:
                    with open(exe_log, "a") as f:
                        f.write(f"[LOCAL SOMD] SKIPPED lambda={lam_arg} at "
                                f"{datetime.now():%Y-%m-%d %H:%M:%S}\n")
                    logger.info(f"[LOCAL SOMD] Already finished in {real_cwd}; SKIPPING")
                    # Return a fake job ID that will immediately be marked as finished
                    return 999999
                
            return _run_somd_locally(script_path, lam_arg, cwd)
        else:
            # Check if this is an MBAR analysis script
            if _is_mbar_script(script_path):
                return _submit_mbar_parallel(script_path, cwd)
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
            logger.error(f"[LOCAL SOMD] ❌ {cfg_path} is missing or empty in {real_cwd}; cannot run SOMD")
            with open(os.path.join(real_cwd, "local_execution.log"), "a") as flog:
                flog.write(f"[LOCAL SOMD] ❌ JOB FAILED {cfg_path} is missing or empty in {real_cwd}; cannot run SOMD\n")
            raise RuntimeError(f"somd.cfg is missing or empty in {real_cwd}; cannot run SOMD")

        # Record start time
        start_time = time.time()
        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sim_info = _format_sim_info(real_cwd, lam_arg)
        logger.info(f"[LOCAL SOMD] {start_timestamp} Running {sim_info} in {cwd or os.getcwd()}")
        
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
                        logger.warning("[LOCAL SOMD] Forced platform to CPU")
                else:
                    # Keep whatever platform was specified
                    logger.info(f"[LOCAL SOMD] Using {current_platform} platform")
        
        # Substitute lambda value
        parts = [tok.replace("$lam", lam_arg).replace("${lam}", lam_arg) for tok in parts]
        
        logger.info(f"[LOCAL SOMD] Executing: {' '.join(parts)} at {real_cwd}")
        
        try:
            subprocess.run(parts, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)

            end_time = time.time()
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration_seconds = end_time - start_time

            logger.info(f"[LOCAL SOMD] ✅ {end_timestamp} Completed successfully for {sim_info}")
            logger.info(f"[LOCAL SOMD] Simulation took {duration_seconds:.2f} seconds")

            # Create a local_execution.log file that mimics SLURM output
            local_execution_log_path = os.path.join(real_cwd, "local_execution.log")
            with open(local_execution_log_path, 'a') as f:
                f.write(f"[LOCAL SOMD] Starting {sim_info} at {start_timestamp}\n")
                f.write(f"[LOCAL SOMD] Completed {sim_info} at {end_timestamp}\n")
                f.write(f"Simulation took {duration_seconds:.2f} seconds\n")
                f.write(f"✅ Job completed successfully\n")
            return 888888  # Return fake job ID on success
        
        except subprocess.CalledProcessError as e:
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.error(f"[LOCAL SOMD] ❌ JOB FAILED with return code {e.returncode}")
            logger.error(f"[LOCAL SOMD] ❌ STDERR:\n{e.stderr}")

            # --- write a “failed” marker into the same execution log ---
            local_execution_log_path = os.path.join(real_cwd, "local_execution.log")
            with open(local_execution_log_path, "a") as f:
                f.write(f"[LOCAL SOMD] {end_timestamp} ❌ JOB FAILED with return code {e.returncode}\n")
                f.write(f"[LOCAL SOMD] ❌ STDERR: {e.stderr}\n")

            raise RuntimeError(f"SOMD simulation failed: {e}")
    

    def _run_prep_locally(script_path, cwd) -> int:
        """Run preparation step locally."""
        logger.info(f"[LOCAL PREP] Running preparation script in {cwd or os.getcwd()}")
        
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
        
        logger.info(f"[LOCAL PREP] Executing: {python_command} at {cwd or os.getcwd()}")
        
        try:
            subprocess.run(python_command, shell=True, cwd=cwd, check=True)
            logger.info(f"[LOCAL PREP] ✅ Completed successfully")
            return 888888 # Return fake job ID
        except subprocess.CalledProcessError as e:
            logger.error(f"[LOCAL PREP] ❌ Failed with return code {e.returncode}")
            raise RuntimeError(f"Preparation step failed: {e}")
    

    def _run_mbar_locally(self, script_path, cwd) -> int: 
        """Submit MBAR to the process pool and return a fake job id immediately."""
        logger.info(f"[LOCAL MBAR] Queuing MBAR analysis in {cwd or os.getcwd()}")

        try:
            with open(script_path) as f:
                script_content = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"MBAR script not found: {script_path}")
        
        # Find the MBAR command
        mbar_command = None
        for line in script_content.splitlines():
            line = line.strip()
            if (("analyse_freenrg" in line) and 
                not line.startswith("#") and
                not line.startswith("export")):
                mbar_command = line
                break
        
        if not mbar_command:
            logger.warning(f"[LOCAL MBAR] No MBAR command found, trying to run script directly")
            mbar_command = f"bash {script_path}"
        

        logger.info(f"[LOCAL MBAR] Executing MBAR command: {mbar_command}")
        
        try:
            subprocess.run(mbar_command, shell=True, cwd=cwd, check=True,
                                  capture_output=True, text=True)
            
            logger.info(f"[LOCAL MBAR] ✅ MBAR analysis completed successfully")
            return 666666  # Return fake job ID
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[LOCAL MBAR] ❌ MBAR analysis failed with return code {e.returncode}")
            logger.error(f"[LOCAL MBAR] ❌ STDERR: {e.stderr}")
            
            # MBAR often fails during early adaptive phases due to insufficient data
            logger.warning(f"[LOCAL MBAR] ⚠️ MBAR failure is common during early adaptive phases")
            logger.warning(f"[LOCAL MBAR] ⚠️ Creating dummy output file to allow simulation to continue")
            
            # Create a realistic dummy output file that matches the expected MBAR format
            dummy_output = _extract_mbar_output_file(mbar_command)
            if dummy_output:
                dummy_path = os.path.join(cwd, dummy_output)
                _create_dummy_mbar_output(dummy_path, cwd)
                logger.warning(f"[LOCAL MBAR] Created dummy MBAR output file: {dummy_path}")
            
            return 666666  # Return success to continue execution
        

    def _submit_mbar_parallel(script_path, cwd) -> int:
        """Submit MBAR job to global manager for parallel execution."""
        return _GLOBAL_MBAR_MANAGER.submit_mbar_job(script_path, cwd)


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
                logger.error(f"[ERROR] ❌ get tot_gpu_time - Failed to read local execution: {e}")
        else:
            logger.error(f"[ERROR] ❌ get tot_gpu_time - Local execution log not found at {timing_log_path}")

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
        # 888888 from actual execution, 999999 from early skip, 666666 from mbar output
        fake_job_ids = [888888, 999999]  
        fake_fail_ids = [777777]
        
        # Mark any jobs with fake IDs as finished and remove from queue
        jobs_to_remove = []
        for job in self._slurm_queue:
            # grab more info like command list, etc.
            job_sim_info = _parse_sim_info_from_job(job)

            sid = getattr(job, "slurm_job_id", None)
            if not isinstance(sid, int):
                # not yet assigned, skip this cycle
                continue

            # Handle MBAR jobs - check status in global manager
            if job.slurm_job_id >= 600000:  # MBAR job IDs start at 600000
                status = _GLOBAL_MBAR_MANAGER.get_job_status(job.slurm_job_id)
                
                if status == "FINISHED":
                    if not getattr(job, "_already_marked_finished", False):
                        job.status = _JobStatus.FINISHED
                        job._already_marked_finished = True
                        jobs_to_remove.append(job)
                        logger.info(f"[LOCAL UPDATE] ✅ MBAR job {job.slurm_job_id} finished, {job_sim_info}")
                elif status == "FAILED":
                    if not getattr(job, "_already_marked_finished", False):
                        job.status = _JobStatus.FINISHED  # Still mark as finished to continue pipeline
                        job._already_marked_finished = True
                        jobs_to_remove.append(job)
                        logger.warning(f"[LOCAL UPDATE] ⚠️ MBAR job {job.slurm_job_id} failed but continuing, {job_sim_info}")
                elif status == "RUNNING":
                    # Keep job in queue, mark as running if not already done
                    if not hasattr(job, '_logged_running'):
                        logger.info(f"[LOCAL UPDATE] 🟡 MBAR job {job.slurm_job_id} running, {job_sim_info}")
                        job._logged_running = True
                continue
            
            if job.slurm_job_id in fake_job_ids:
                if getattr(job, "_already_marked_finished", False):
                    continue
                job.status = _JobStatus.FINISHED
                job._already_marked_finished = True  # flag to prevent repeated downstream logging
                jobs_to_remove.append(job)
                logger.info(f"[LOCAL UPDATE] ✅ Marking job slurm_job_id={job.slurm_job_id}, {job_sim_info} as finished")
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
    
    def _local_wait(self) -> None:
        """Wait for all jobs to finish with faster polling for local execution."""
        while len(self.queue) > 0:
            self.update()
            sleep(3) 
    
    def _local_stage_wait(self) -> None:
        """Wait for the stage to finish with faster polling."""
        sleep(3)  
        self.virtual_queue.update()
        while self.running:
            sleep(3) 
            self.virtual_queue.update()

    def _local_sim_runner_wait(self) -> None:
        """Wait for the simulation runner to finish with faster polling for local execution."""
        sleep(3) 
        while self.running:
            sleep(3)


    # APPLY THE PATCHES NOW
    Simulation.get_tot_gpu_time = timing_based_get_tot_gpu_time

    VirtualQueue.update = local_update
    VirtualQueue._submit_job = _submit_locally

    Job.slurm_outfile = property(local_slurm_outfile)
    Job.has_failed = local_has_failed

    VirtualQueue.submit = quiet_submit

    if use_faster_wait:
        # Patch the wait method to use faster polling for local execution
        # This is useful for local testing to avoid long waits
        VirtualQueue.wait = _local_wait
        Stage.wait = _local_stage_wait
        SimulationRunner.wait = _local_sim_runner_wait

    logger.info("A3FE._virtual_queue was successfully patched for local execution")



# NOTE: THIS PATCH IS FOR TESTING AND DEBUGGING PURPOSES ONLY
def _debug_patch_stage_skip_adaptive_efficiency():
    """
    Patch Stage._run_without_threading to optionally skip the adaptive efficiency loop.
    This is useful for debugging and testing or when need to skip the resource-intensive optimization phase.
    """
    # Set up colored logger for this function
    logger = logging.getLogger(__name__ + ".STAGE_PATCH")
    logger.handlers.clear()  # Clear any existing handlers
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't propagate to avoid duplicate messages    
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    handler.addFilter(shared_filter)
    logger.addHandler(handler)
    
    def patched_run_without_threading(
        self,
        run_nos,
        adaptive=True,
        runtime=None,
        max_runtime=60,
    ):
        """Modified _run_without_threading that can skip the adaptive efficiency loop"""
        try:
            self.kill_thread = False
            if not adaptive and runtime is None:
                raise ValueError(
                    "If adaptive equilibration detection is disabled, a runtime must be supplied."
                )
            if adaptive and runtime is not None:
                raise ValueError(
                    "If adaptive equilibration detection is enabled, a runtime cannot be supplied."
                )
            if not adaptive:
                self._logger.info(
                    f"Starting {self}. Adaptive equilibration = {adaptive}..."
                )
            elif adaptive:
                self._logger.info(
                    f"Starting {self}. Adaptive equilibration = {adaptive}..."
                )
                if runtime is None:
                    runtime = 0.2  # ns

            # Run initial SOMD simulations
            if SKIP_ADAPTIVE_EFFICIENCY:
                self._logger.info("SKIP_ADAPTIVE_EFFICIENCY=True: Skipping initial SOMD simulations (dry run mode)")
                self.running_wins = []
            else:
                for win in self.lam_windows:
                    win.run(run_nos=run_nos, runtime=runtime)
                    win._update_log()
                    self._dump()

            if not SKIP_ADAPTIVE_EFFICIENCY:
                self.running_wins = self.lam_windows.copy()
            self._dump()

            if adaptive:
                # NEW: Check if we should skip adaptive efficiency
                if SKIP_ADAPTIVE_EFFICIENCY:
                    self._logger.info("SKIP_ADAPTIVE_EFFICIENCY=True: Skipping adaptive efficiency optimization loop")
                    self._maximally_efficient = True
                else:
                    self._logger.info("Running adaptive efficiency optimization loop")
                    self._run_loop_adaptive_efficiency(
                        run_nos=run_nos, max_runtime=max_runtime
                    )                
                self._run_loop_adaptive_equilibration_multiwindow(
                    run_nos=run_nos, max_runtime=max_runtime
                )
            else:
                self._run_loop_non_adaptive()
            self._logger.info(f"All simulations in {self} have finished.")

        except Exception as e:
            self._logger.exception("")
            raise e
    
    # APPLY THE PATCH
    Stage._run_without_threading = patched_run_without_threading
    logger.info(f"Stage._run_without_threading patched to {'skip' if SKIP_ADAPTIVE_EFFICIENCY else 'include'} adaptive efficiency loop")



def _debug_simulation_times(calc):
    """Debug simulation times to identify inconsistencies
    
    sometimes we get the following error:

    │ /Users/jingjinghuang/Documents/fep_workflow/a3fe_jh/a3fe/analyse/process_grads.py:638 in         │
    │ get_time_series_multiwindow                                                                      │
    │                                                                                                  │
    │   635 │   │   │   │   for i in range(n_runs)                                                     │
    │   636 │   │   │   ]                                                                              │
    │   637 │   │   ):                                                                                 │
    │ ❱ 638 │   │   │   raise ValueError(                                                              │
    │   639 │   │   │   │   "Total simulation times are not the same for all runs. Please ensure tha   │
    │   640 │   │   │   │   "the total simulation times are the same for all runs."                    │
    │   641 │   │   │   )                                                                              │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
    ValueError: Total simulation times are not the same for all runs. Please ensure that the total simulation times are the same for all runs.
    """
    print("=== DEBUGGING SIMULATION TIMES ===")
    
    issues_found = []
    
    for leg in calc.legs:
        print(f"\n=== {leg.leg_type.name} LEG ===")
        for stage in leg.stages:
            print(f"\n--- {stage.stage_type.name} STAGE ---")
            
            stage_issues = []
            for win in stage.lam_windows:
                print(f"\nLambda {win.lam:.3f}:")
                simtimes = []
                
                for i, sim in enumerate(win.sims, 1):
                    simtime = sim.get_tot_simtime()
                    simtimes.append(simtime)
                    print(f"  Run {i}: {simtime:.6f} ns")
                    
                    # Check if simulation output files exist
                    simfile_path = f"{sim.output_dir}/simfile.dat"
                    if not os.path.exists(simfile_path):
                        issue = f"Missing simfile.dat for {stage.stage_type.name} lambda {win.lam:.3f} run {i}"
                        print(f"    ERROR: {issue}")
                        issues_found.append(issue)
                    elif os.path.getsize(simfile_path) == 0:
                        issue = f"Empty simfile.dat for {stage.stage_type.name} lambda {win.lam:.3f} run {i}"
                        print(f"    ERROR: {issue}")
                        issues_found.append(issue)
                
                # Check consistency within this lambda window
                if len(set(f"{t:.6f}" for t in simtimes)) > 1:
                    issue = f"Inconsistent times in {stage.stage_type.name} lambda {win.lam:.3f}: {simtimes}"
                    print(f"    ERROR: {issue}")
                    stage_issues.append((win.lam, simtimes))
                    issues_found.append(issue)
                else:
                    print(f"    ✓ All runs consistent: {simtimes[0]:.6f} ns")
            
            # Check consistency across lambda windows in this stage
            if stage_issues:
                print(f"\n  STAGE {stage.stage_type.name} HAS TIMING ISSUES:")
                for lam, times in stage_issues:
                    print(f"    Lambda {lam:.3f}: {times}")
    
    return issues_found


if __name__ == "__main__":
    # Set up global logging first
    setup_global_logging()

    # Configure via environment variables
    FORCE_LOCAL_EXECUTION = True
    FORCE_CPU_PLATFORM = False
    SKIP_ADAPTIVE_EFFICIENCY = True  # skip the adaptive efficiency optimization loop
   

    patch_virtual_queue_for_local_execution()

    _debug_patch_stage_skip_adaptive_efficiency()
    
    sysprep_cfg = SystemPreparationConfig(slurm=True) # use default settings

    calc = a3.Calculation(base_dir="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again7",
                          input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again7/input")

    # calc.setup(
    #     bound_leg_sysprep_config=sysprep_cfg,
    #     free_leg_sysprep_config=sysprep_cfg,
    # )

    # add_filter_recursively(calc)

    # calc.get_optimal_lam_vals(delta_er=0.5)
    calc.run(adaptive=True, 
             parallel=False,
             runtime_constant=0.0005)              
    
    calc.wait()
    for leg in calc.legs:
        for stage in leg.stages:
            equilibrated = stage.is_equilibrated()
            print(f"{leg.leg_type.name} {stage.stage_type.name}: equilibrated = {equilibrated}")

    calc.analyse()
    calc.save()
