"""Unit and regression tests for the _virtual_queue module."""

import os
from tempfile import TemporaryDirectory

from ..run._virtual_queue import Job, VirtualQueue, _JobStatus


def test_job():
    """Test that the Job class works correctly"""
    job = Job(1, "echo hello")
    assert job.virtual_job_id == 1
    assert job.command == "echo hello"
    assert job.slurm_job_id is None
    job.slurm_job_id = 1234
    assert job.slurm_job_id == 1234


def test_virtual_queue():
    """Check that the virtual queue works correctly. Note that we
    can't test submit, kill, or update as these require a slurm queue to
     submit to, which we don't have on the CI server."""
    with TemporaryDirectory() as dirname:
        # Set up the virtual queue and mess around with queues
        v_queue = VirtualQueue(queue_len_lim=30, log_dir=dirname)
        # Manually do most of submit
        virtual_job_id = v_queue._available_virt_job_id
        # Increment this so that it is never used again for this queue
        v_queue._available_virt_job_id += 1
        job1 = Job(virtual_job_id, "echo hello")
        v_queue._pre_queue.append(job1)
        v_queue._logger.info(f"{job1} submitted")
        # Add a job straight to the slurm queue
        job2 = Job(virtual_job_id, "echo hello")
        v_queue._slurm_queue.append(job2)
        v_queue._update_log()

        # Check that the total queue is the combination of the slurm and prequeue
        assert v_queue.queue == [job1, job2]
        # Check that logging is working
        assert os.path.isfile(os.path.join(dirname, "virtual_queue.log"))

        # Check that killing and flushing works correctly.
        v_queue.kill(job1)
        assert job1.slurm_job_id is None
        assert job1.status == _JobStatus.KILLED

        # Flusing pays no attention to the status of the jobs
        v_queue._flush()
        assert v_queue.queue == []
