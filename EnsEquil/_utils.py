"""Utilities for the Ensemble, Window, and Simulation Classes"""

from dataclasses import dataclass as _dataclass
from matplotlib import pyplot as _plt
import logging as _logging
import numpy as _np
import scipy.stats as _stats
import subprocess as _subprocess
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

def plot(x_vals: _np.ndarray, y_vals: _np.ndarray, x_label: str, y_label: str, 
         outfile: str, vline_val: _Optional[float] = None, 
         hline_val: _Optional[float] = None) -> None:
    """ 
    Plot several sets of y_vals against one set of x vals, and show confidence
    intervals based on inter-y-set deviations (assuming normality).

    Parameters
    ----------
    x_vals : np.ndarray
        1D array of x values.
    y_vals : np.ndarray
        1 or 2D array of y values, with shape (n_sets, n_vals). Assumes that
        the sets of data are passed in the same order as the runs.
    x_label : str
        Label for the x axis.
    y_label : str
        Label for the y axis.
    outfile : str
        Name of the output file.
    vline_val : float, Optional
        x value to draw a vertical line at, for example the time taken for
        equilibration.
    hline_val : float, Optional
        y value to draw a horizontal line at.
    """
    y_avg = _np.mean(y_vals, axis=0)
    conf_int = _stats.t.interval(0.95, len(y_vals[:,0])-1, loc=y_avg, scale=_stats.sem(y_vals,axis=0)) # 95 % C.I.

    fig, ax = _plt.subplots(figsize=(8,6))
    ax.plot(x_vals, y_avg, label="Mean", linewidth=2)
    for i, entry in enumerate(y_vals):
        ax.plot(x_vals, entry, alpha=0.5, label=f"run {i+1}")
    if vline_val is not None:
        ax.axvline(x=vline_val, color='red', linestyle='dashed')
    if hline_val is not None:
        ax.axhline(y=hline_val, color='black', linestyle='dashed')
    # Add confidence intervals
    ax.fill_between(x_vals, conf_int[0], conf_int[1], alpha=0.5, facecolor='#ffa500')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    
@_dataclass
class Job():
    """Class to hold information about a job"""
    virtual_job_id: int
    command: str
    slurm_job_id: _Optional[int] = None

    def __str__(self) -> str:
        # Avoid printing the command, which may be long
        return f"Job (virtual_job_id = {self.virtual_job_id}, slurm_job_id= {self.slurm_job_id})"

class VirtualQueue():
    """A virtual slurm queue which has no limit on the number
    of queued jobs, which submits slurm jobs to the real queue
    when there are few enough jobs queued. This gets round slurm
    queue limits."""

    def __init__(self, que_len_lim: int =  2500, log_dir: str = "./output") -> None:
        """ 
        Initialise the virtual queue.
        
        Parameters
        ----------
        queue_len_lim : int, Optional, default: 2500
            The maximum number of jobs to queue in the real queue.
        log_dir : str, Optional, default: "./output"
            The directory to write the log to.
        
        Returns
        -------
        None
        """
        self._slurm_queue = []
        self._pre_queue = []
        self._available_virt_job_id = 0
        self.queue_len_lim = que_len_lim

        # Set up logging
        self._logger = _logging.getLogger(str(self))
        # For the file handler, we want to log everything
        self._logger.setLevel(_logging.DEBUG)
        file_handler = _logging.FileHandler(f"{log_dir}/virtual_queue.log")
        file_handler.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self._logger.addHandler(file_handler)

        # Write out initial settings
        self._update_log()

    @property
    def queue(self) -> _List[Job]:
        """The queue of jobs, both real and virtual."""
        return self._slurm_queue + self._pre_queue

    def submit(self, command: str) -> Job:
        """ 
        Submit a job to the virtual queue.
        
        Parameters
        ----------
        command : str
            The command to be run by sbatch.
        
        Returns
        -------
        job : Job
            The Job submitted to the virtual queue."""
        virtual_job_id = self._available_virt_job_id
        # Increment this so that it is never used again for this queue
        self._available_virt_job_id += 1
        job = Job(virtual_job_id, command)
        self._pre_queue.append(job)
        self._logger.info(f"{job} submitted")
        # Now update - the job will be moved to the real queue if there is space
        self.update()
        return job
    
    def kill(self, job: Job) -> None:
        """Kill and remove a job from the real and virtual queues."""
        # All jobs are in the joint queue, so remove it from there
        self.queue.remove(job)
        # If the job is in the real queue, kill it
        if job in self._slurm_queue:
            self._slurm_queue.remove(job)
            _subprocess.run(["scancel", str(job.slurm_job_id)])
        else: # Job is in the pre-queue
            self._pre_queue.remove(job)

    def update(self) -> None:
        """Remove jobs from the queue if they have finished, then move jobs from
        the pre-queue to the real queue if there is space."""
        # First, remove jobs from the queue if they have finished
        # Get job ids of currently running jobs. This assumes no array jobs.
        cmd = r"squeue -h -u $USER | awk '{print $1}' | grep -v -E '\[|_' | paste -s -d, -"
        process = _subprocess.Popen(cmd, shell=True, stdin=_subprocess.PIPE,
                                    stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
                                    close_fds=True)
        output = process.communicate()[0]
        running_slurm_job_ids = [int(job_id) for job_id in output.decode('utf-8').strip().split(",") if job_id != ""]
        n_running_slurm_jobs = len(running_slurm_job_ids)
        # Remove completed jobs from the queues
        self._slurm_queue = [job for job in self._slurm_queue if job.slurm_job_id in running_slurm_job_ids]

        # Submit jobs if possible
        if n_running_slurm_jobs < self.queue_len_lim:
            # Move jobs from the pre-queue to the real queue
            n_jobs_to_move = self.queue_len_lim - n_running_slurm_jobs
            jobs_to_move = self._pre_queue[:n_jobs_to_move]
            self._pre_queue = self._pre_queue[n_jobs_to_move:]
            self._slurm_queue += jobs_to_move
            # Submit the jobs
            for job in jobs_to_move:
                #cmd = f"sbatch {job.command}"
                # Sketchy hack to get sbatch to work on my system
                cmd = f"~/Documents/research/scripts/abfe/rbatch.sh {job.command}"
                process = _subprocess.Popen(cmd, shell=True, stdin=_subprocess.PIPE,
                                            stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
                                            close_fds=True)
                if process.stdout is None:
                    raise ValueError("Could not get stdout from process.")
                process_output = process.stdout.read()
                job.slurm_job_id = int((process_output.split()[-1]))

        self._logger.info(f"Queue updated")
        self._logger.info(f"Slurm queue slurm job ids: {[job.slurm_job_id for job in self._slurm_queue]}")
        self._logger.info(f"Slurm queue virtual job ids: {[job.virtual_job_id for job in self._slurm_queue]}")
        self._logger.info(f"Pre-queue virtual job ids: {[job.virtual_job_id for job in self._pre_queue]}")

    def _update_log(self) -> None:
        """Update the log file with the current status of the queue."""
        self._logger.debug("##############################################")
        for var in vars(self):
            self._logger.debug(f"{var}: {getattr(self, var)} ")
        self._logger.debug("##############################################")