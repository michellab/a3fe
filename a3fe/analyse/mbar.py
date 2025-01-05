"""
Functionality for running mbar on SOMD output files. This uses
pymbar through SOMD
"""

__all__ = ["run_mbar"]

import glob as _glob
import os as _os
import subprocess as _subprocess
from time import sleep as _sleep
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple

import numpy as _np

from ..read._process_somd_files import read_mbar_gradients as _read_mbar_gradients
from ..read._process_somd_files import read_mbar_result as _read_mbar_result
from ..read._process_somd_files import (
    write_truncated_sim_datafile as _write_truncated_sim_datafile,
)
from ..run._virtual_queue import Job as _Job
from ..run._virtual_queue import VirtualQueue as _VirtualQueue

from ..configuration import SlurmConfig as _SlurmConfig


def run_mbar(
    output_dir: str,
    run_nos: _List[int],
    percentage_end: float = 100,
    percentage_start: float = 0,
    subsampling: bool = False,
    delete_outfiles: bool = False,
    equilibrated: bool = True,
) -> _Tuple[_np.ndarray, _np.ndarray, _List[str], _Dict[str, _Dict[str, _np.ndarray]]]:
    """
    Run MBAR on SOMD output files.

    Parameters
    ----------
    output_dir : str
        The path to the output directory
    run_nos : List[int]
        The run numbers to use for MBAR.
    percentage_end : float, Optional, default: 100
        The percentage of data after which to truncate the datafiles.
        For example, if 100, the full datafile will be used. If 50, only
        the first 50% of the data will be used.
    percentage_start : float, Optional, default: 0
        The percentage of data before which to truncate the datafiles.
        For example, if 0, the full datafile will be used. If 50, only
        the last 50% of the data will be used.
    subsampling : bool, Optional, default: False
        Whether to use subsampling for MBAR.
    delete_outfiles : bool, Optional, default: False
        Whether to delete the MBAR analysis output files after the free
        energy change and errors have been extracted.
    equilibrated : bool, Optional, default: True
        Whether to use the equilibrated datafiles or the full datafiles.
        If true, the files name simfile_equilibrated.dat will be used,
        otherwise simfile.dat will be used.

    Returns
    -------
    free_energies : np.ndarray
        The free energies from each run, in kcal mol-1.
    errors : np.ndarray
        The mbar errors on the free energies from each run, in kcal mol-1.
    mbar_out_files : List[str]
        The paths to the MBAR output files.
    mbar_grads: Dict[str, Dict[str, np.ndarray]]
        The gradients of the free energies obtained from the MBAR results (not TI),
        for each run.
    """
    tmp_simfiles = _prepare_simfiles(
        output_dir=output_dir,
        run_nos=run_nos,
        percentage_end=percentage_end,
        percentage_start=percentage_start,
        equilibrated=equilibrated,
    )

    # Run MBAR using pymbar through SOMD
    mbar_out_files = []
    for run_no in run_nos:
        outfile = f"{output_dir}/freenrg-MBAR-run_{str(run_no).zfill(2)}_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.dat"
        mbar_out_files.append(outfile)
        with open(outfile, "w") as ofile:
            cmd_list = [
                "analyse_freenrg",
                "mbar",
                "-i",
                f"{output_dir}/lambda*/run_{str(run_no).zfill(2)}/simfile_truncated_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.dat",
                "-p",
                "100",
                "--overlap",
            ]
            if subsampling:
                cmd_list.append("--subsampling")
            _subprocess.run(cmd_list, stdout=ofile)

    free_energies = _np.array([_read_mbar_result(ofile)[0] for ofile in mbar_out_files])
    errors = _np.array([_read_mbar_result(ofile)[1] for ofile in mbar_out_files])

    # Get the gradients from the MBAR results
    mbar_grads = {}
    for i, run_no in enumerate(run_nos):
        mbar_grads[f"run_{run_no}"] = {}
        lam_vals, grads, grad_errors = _read_mbar_gradients(mbar_out_files[i])
        mbar_grads[f"run_{run_no}"]["lam_vals"] = lam_vals
        mbar_grads[f"run_{run_no}"]["grads"] = grads
        mbar_grads[f"run_{run_no}"]["grad_errors"] = grad_errors

    if delete_outfiles:
        for ofile in mbar_out_files:
            _subprocess.run(["rm", ofile])
        mbar_out_files = []

    # Clean up temporary simfiles
    for tmp_simfile in tmp_simfiles:
        _subprocess.run(["rm", tmp_simfile])

    return free_energies, errors, mbar_out_files, mbar_grads


def submit_mbar_slurm(
    output_dir: str,
    virtual_queue: _VirtualQueue,
    slurm_config: _SlurmConfig,
    run_nos: _List[int],
    percentage_end: float = 100,
    percentage_start: float = 0,
    subsampling: bool = False,
    equilibrated: bool = True,
    wait: bool = False,
) -> _Tuple[_List[_Job], _List[str], _List[str]]:
    """
    Submit slurm jobs to run MBAR on SOMD output files.

    Parameters
    ----------
    output_dir : str
        The path to the output directory
    virtual_queue : VirtualQueue
        The virtual queue to submit the MBAR jobs to.
    slurm_config: SlurmConfig
        The SLURM configuration to use for the jobs.
    run_nos : List[int]
        The run numbers to use for MBAR.
    percentage_end : float, Optional, default: 100
        The percentage of data after which to truncate the datafiles.
        For example, if 100, the full datafile will be used. If 50, only
        the first 50% of the data will be used.
    percentage_start : float, Optional, default: 0
        The percentage of data before which to truncate the datafiles.
        For example, if 0, the full datafile will be used. If 50, only
        the last 50% of the data will be used.
    subsampling : bool, Optional, default: False
        Whether to use subsampling for MBAR.
    equilibrated : bool, Optional, default: True
        Whether to use the equilibrated datafiles or the full datafiles.
        If true, the files name simfile_equilibrated.dat will be used,
        otherwise simfile.dat will be used.
    wait: bool, default: False
        Whether to wait for the job to complete or not.

    Returns
    -------
    jobs : List[Job]
        The jobs submitted to the virtual queue.

    mbar_out_files : List[str]
        The paths to the MBAR output files, which will be created
        once the jobs complete.

    tmp_files : List[str]
        The paths to temporary files (truncated simfiles and submission scripts),
        so that they can be cleaned up later.
    """
    tmp_files = _prepare_simfiles(
        output_dir=output_dir,
        run_nos=run_nos,
        percentage_end=percentage_end,
        percentage_start=percentage_start,
        equilibrated=equilibrated,
    )

    # Add MBAR command and run for each run.
    mbar_out_files = []
    jobs = []

    for run_no in run_nos:
        # Get the name of the output files - the first for the MBAR output, and the second for the SLURM output
        outfile = f"{output_dir}/freenrg-MBAR-run_{str(run_no).zfill(2)}_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.dat"
        mbar_out_files.append(outfile)
        slurm_outfile = f"slurm_freenrg-MBAR-run_{str(run_no).zfill(2)}_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.out"
        slurm_config.output = slurm_outfile

        # Create the command.
        cmd_list = [
            "analyse_freenrg",
            "mbar",
            "-i",
            f"{output_dir}/lambda*/run_{str(run_no).zfill(2)}/simfile_truncated_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.dat",
            "-p",
            "100",
            "--overlap",
            "--output",
            outfile,
        ]
        if subsampling:
            cmd_list.append("--subsampling")
        slurm_cmd = " ".join(cmd_list)

        # Create and submit the job
        script_name = f"{output_dir}/freenrg-MBAR-run_{str(run_no).zfill(2)}_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start"
        submission_args = slurm_config.get_submission_cmds(
            slurm_cmd, output_dir, script_name
        )
        job = virtual_queue.submit(
            submission_args,
            slurm_file_base=slurm_config.get_slurm_output_file_base(output_dir),
        )
        tmp_files += [script_name + ".sh", _os.path.join(output_dir, slurm_outfile)]

        # Update the virtual queue to submit the job to the real queue
        virtual_queue.update()
        jobs.append(job)

    # Wait for the job to complete if we've specified wait
    if wait:
        for job in jobs:
            while job in virtual_queue.queue:
                _sleep(30)
                virtual_queue.update()

    return jobs, mbar_out_files, tmp_files


def collect_mbar_slurm(
    output_dir: str,
    run_nos: _List[int],
    jobs: _List[_Job],
    mbar_out_files: _List[str],
    virtual_queue: _VirtualQueue,
    delete_outfiles: bool = False,
    tmp_files: _List[str] = [],
) -> _Tuple[_np.ndarray, _np.ndarray, _List[str], _Dict[str, _Dict[str, _np.ndarray]]]:
    """
    Collect the results from MBAR slurm jobs.

    Parameters
    ----------
    output_dir : str
        The path to the output directory
    run_nos : List[int]
        The run numbers to use for MBAR.
    jobs : List[Job]
        The jobs submitted to the virtual queue.
    mbar_out_files : List[str]
        The paths to the MBAR
    virtual_queue : VirtualQueue
        The virtual queue to submit the MBAR jobs to.
    delete_outfiles : bool, Optional, default: False
        Whether to delete the MBAR analysis output files after the free
        energy change and errors have been extracted.
    tmp_files : List[str], Optional, default: []
        The paths to temporary files (truncated simfiles and submission scripts),
        so that they can be cleaned up later.

    Returns
    -------
    free_energies : np.ndarray
        The free energies from each run, in kcal mol-1.
    errors : np.ndarray
        The mbar errors on the free energies from each run, in kcal mol-1.
    mbar_out_files : List[str]
        The paths to the MBAR output files. Returned for consistency with the non-
        slurm function.
    mbar_grads: Dict[str, Dict[str, np.ndarray]]
        The gradients of the free energies obtained from the MBAR results (not TI),
        for each run.
    """
    # Wait for the jobs to finish.
    for job in jobs:
        while job in virtual_queue.queue:
            _sleep(30)
            virtual_queue.update()

    # Try to read the results
    try:
        free_energies = _np.array(
            [_read_mbar_result(ofile)[0] for ofile in mbar_out_files]
        )
        errors = _np.array([_read_mbar_result(ofile)[1] for ofile in mbar_out_files])
    except FileNotFoundError:
        raise FileNotFoundError(
            "The MBAR output files could not be found. Checkk the output of the slurm jobs in"
            f" {output_dir} to see if there were any errors."
        )

    # Get the gradients from the MBAR results
    mbar_grads = {}
    for i, run_no in enumerate(run_nos):
        mbar_grads[f"run_{run_no}"] = {}
        lam_vals, grads, grad_errors = _read_mbar_gradients(mbar_out_files[i])
        mbar_grads[f"run_{run_no}"]["lam_vals"] = lam_vals
        mbar_grads[f"run_{run_no}"]["grads"] = grads
        mbar_grads[f"run_{run_no}"]["grad_errors"] = grad_errors

    if delete_outfiles:
        for ofile in mbar_out_files:
            _subprocess.run(["rm", ofile])
        mbar_out_files = []

    # Clean up temporary simfiles
    for tmp_file in tmp_files:
        _subprocess.run(["rm", tmp_file])

    return free_energies, errors, mbar_out_files, mbar_grads


def _prepare_simfiles(
    output_dir: str,
    run_nos: _List[int],
    percentage_end: float = 100,
    percentage_start: float = 0,
    equilibrated: bool = True,
) -> _List[str]:
    """
    Helper function to prepare the simfiles for MBAR analysis. Returns
    the paths to the temporary truncated simfiles, so that they can be
    cleaned up later.
    """
    # Check that the simfiles actually exist
    file_name = "simfile_equilibrated.dat" if equilibrated else "simfile.dat"
    simfiles = _glob.glob(f"{output_dir}/lambda*/run_*/{file_name}")
    # Filter by run numbers
    if run_nos is not None:
        simfiles = [
            simfile
            for simfile in simfiles
            if int(simfile.split("/")[-2].split("_")[-1]) in run_nos
        ]

    if len(simfiles) == 0:
        raise FileNotFoundError(
            "No equilibrated simfiles found. Have you run the simulations "
            "and checked for equilibration?"
        )

    # Create temporary truncated simfiles
    tmp_simfiles = []  # Clean these up afterwards
    for simfile in simfiles:
        tmp_simfile = _os.path.join(
            _os.path.dirname(simfile),
            f"simfile_truncated_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.dat",
        )
        tmp_simfiles.append(tmp_simfile)
        _write_truncated_sim_datafile(
            simfile,
            tmp_simfile,
            fraction_final=percentage_end / 100,
            fraction_initial=percentage_start / 100,
        )

    return tmp_simfiles
