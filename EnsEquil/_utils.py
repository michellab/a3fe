"""Utilities for the Ensemble, Window, and Simulation Classes"""

from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
import glob as _glob
import logging as _logging
import os as _os
import subprocess as _subprocess
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

def read_mbar_outfile(outfile: str) -> _Tuple[float, float]:
    """ 
    Read the output file from MBAR, and return the free energy and error.

    Parameters
    ----------
    outfile : str
        The name of the output file.

    Returns
    -------
    free_energy : float
        The free energy in kcal/mol.
    free_energy_err : float
        The error on the free energy in kcal/mol.
    """
    with open(outfile, 'r') as f:
        lines = f.readlines()
    # The free energy is the 5th last line of the file
    free_energy = float(lines[-4].split(",")[0])
    free_energy_err = float(lines[-4].split(",")[1].split()[0])

    return free_energy, free_energy_err

class JobStatus(_Enum):
    """An enumeration of the possible job statuses"""
    NONE = 0
    QUEUED = 1
    FINISHED = 2
    FAILED = 3
    KILLED = 4


@_dataclass
class Job():
    """Class to hold information about a job"""
    virtual_job_id: int
    command: str
    slurm_job_id: _Optional[int] = None
    status: JobStatus = JobStatus.NONE
    slurm_file_base: _Optional[str] = None

    def __str__(self) -> str:
        # Avoid printing the command, which may be long
        return f"Job (virtual_job_id = {self.virtual_job_id}, slurm_job_id= {self.slurm_job_id}), status = {self.status}"

    @property
    def slurm_outfile(self) -> str:
        if self.slurm_file_base == None:
            raise AttributeError(f"{self} has no slurm_outfile")
        matching_files = _glob.glob(f"{self.slurm_file_base}*")
        if len(matching_files) == 0:
            raise FileNotFoundError(f"No files matching {self.slurm_file_base}*")
        # Take the most recent file
        newest_file = max(matching_files, key=_os.path.getctime)
        return newest_file

    def has_failed(self) -> bool:
        """Check whether the job has failed"""
        with open(self.slurm_outfile, 'r') as f:
            for line in f.readlines():
                error_statements = ["NaN or Inf has been generated along the simulation",]
                for error in error_statements:
                    if error in line:
                        return True
                
        return False


class VirtualQueue():
    """A virtual slurm queue which has no limit on the number
    of queued jobs, which submits slurm jobs to the real queue
    when there are few enough jobs queued. This gets round slurm
    queue limits."""

    def __init__(self, queue_len_lim: int = 2500, log_dir: str = "./output") -> None:
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
        self._slurm_queue: _List[Job] = []
        self._pre_queue: _List[Job] = []
        self._available_virt_job_id = 0
        self.queue_len_lim = queue_len_lim
        self.log_dir = log_dir

        # Set up logging
        self._set_up_logging()

        # Write out initial settings
        self._update_log()

    def _set_up_logging(self) -> None:
        """Set up logging for the virtual queue """
        # If logging has already been set up, remove it
        if hasattr(self, "_logger"):
            handlers = self._logger.handlers[:]
            for handler in handlers:
                self._logger.removeHandler(handler)
                handler.close()
            del(self._logger)
        self._logger = _logging.getLogger(str(self))
        # For the file handler, we want to log everything
        self._logger.setLevel(_logging.DEBUG)
        file_handler = _logging.FileHandler(f"{self.log_dir}/virtual_queue.log")
        file_handler.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self._logger.addHandler(file_handler)

    @property
    def queue(self) -> _List[Job]:
        """The queue of jobs, both real and virtual."""
        return self._slurm_queue + self._pre_queue

    def submit(self, command: str, slurm_file_base: str) -> Job:
        """ 
        Submit a job to the virtual queue.

        Parameters
        ----------
        command : str
            The command to be run by sbatch.
        slurm_file_base : str
            The base name of the slurm file to be written. This allows
            the slurm file to be checked for errors.

        Returns
        -------
        job : Job
            The Job submitted to the virtual queue."""
        virtual_job_id = self._available_virt_job_id
        # Increment this so that it is never used again for this queue
        self._available_virt_job_id += 1
        job = Job(virtual_job_id, command, slurm_file_base=slurm_file_base)
        job.status = JobStatus.QUEUED
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
        else:  # Job is in the pre-queue
            self._pre_queue.remove(job)
        job.status = JobStatus.KILLED

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
        # Remove completed jobs from the queues and update their status
        for job in self._slurm_queue:
            if job.slurm_job_id not in running_slurm_job_ids:
                # Check if it has failed
                if job.has_failed():
                    job.status = JobStatus.FAILED
                else:
                    job.status = JobStatus.FINISHED
        # Update the slurm queue
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
                process_output = process_output.decode('utf-8').strip()
                #if process_output.startswith("sbatch: error"):
                    #raise RuntimeError(f"Error submitting job: {process_output}")
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
