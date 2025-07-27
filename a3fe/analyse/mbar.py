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
from typing import Optional as _Optional

import numpy as _np

from ..read._process_slurm_files import get_slurm_file_base as _get_slurm_file_base
from ..read._process_somd_files import read_mbar_gradients as _read_mbar_gradients
from ..read._process_somd_files import read_mbar_result as _read_mbar_result
from ..read._process_somd_files import (
    write_truncated_sim_datafile as _write_truncated_sim_datafile,
)
from ..run._virtual_queue import Job as _Job
from ..run._virtual_queue import VirtualQueue as _VirtualQueue
from ..run.slurm_script_generator import A3feSlurmGenerator, A3feSlurmParameters


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
    run_nos: _List[int],
    run_somd_dir: str,
    percentage_end: float = 100,
    percentage_start: float = 0,
    subsampling: bool = False,
    equilibrated: bool = True,
    wait: bool = False,
    slurm_generator: _Optional[A3feSlurmGenerator] = None,
    custom_slurm_overrides: _Optional[_Dict] = None,
) -> _Tuple[_List[_Job], _List[str], _List[str]]:
    """
    Submit slurm jobs to run MBAR on SOMD output files.

    Parameters
    ----------
    output_dir : str
        The path to the output directory
    virtual_queue : VirtualQueue
        The virtual queue to submit the MBAR jobs to.
    run_nos : List[int]
        The run numbers to use for MBAR.
    run_somd_dir : str
        The directory in which to find the `run_somd.sh` script, from
        which the slurm header will be copied.
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

    tmp_simfiles : List[str]
        The paths to the temporary truncated simfiles, so that they can be
        cleaned up later.
    """
    tmp_simfiles = _prepare_simfiles(
        output_dir=output_dir,
        run_nos=run_nos,
        percentage_end=percentage_end,
        percentage_start=percentage_start,
        equilibrated=equilibrated,
    )

    # Initialize SLURM generator if not provided
    if slurm_generator is None:
        slurm_generator = A3feSlurmGenerator()

    # Set up MBAR-specific SLURM parameters
    mbar_slurm_params = A3feSlurmParameters(
        job_name="mbar_analysis",
        cpus_per_task=1,
        gres="",  # CPU-only for MBAR analysis
        time="02:00:00",
        mem="4G",
        modules_to_load=["StdEnv/2020", "gcc/9.3.0", "openmpi/4.0.3"],
        setup_cuda_env=False
    )

    # Apply custom overrides if provided
    if custom_slurm_overrides:
        for key, value in custom_slurm_overrides.items():
            if hasattr(mbar_slurm_params, key):
                setattr(mbar_slurm_params, key, value)
            else:
                mbar_slurm_params.custom_directives[key] = str(value)


    # Add MBAR command and run for each run.
    mbar_out_files = []
    jobs = []
    for run_no in run_nos:
        # Get the name of the output file
        outfile = f"{output_dir}/freenrg-MBAR-run_{str(run_no).zfill(2)}_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.dat"
        mbar_out_files.append(outfile)
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

        mbar_command = " ".join(cmd_list)

        # Generate job name for this specific run
        job_name = f"mbar_run_{str(run_no).zfill(2)}_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start"
     
        # Generate SLURM script using the generator
        try:
            script_content = slurm_generator.generate_prep_script(
                job_name=job_name,
                python_command="",  # We'll override this with the direct command
                custom_overrides=mbar_slurm_params.dict()
            )
            
            # Replace the Python command placeholder with the MBAR command
            script_content = script_content.replace(
                "# Main execution\n",
                f"# Main execution\n{mbar_command}\n"
            )
            
        except Exception as e:
            # Fallback to legacy method if generator fails
            print(f"Warning: SLURM generator failed ({e}), falling back to legacy method")
            script_content = _generate_legacy_mbar_script(run_somd_dir, mbar_command)

        # Write the slurm file
        slurm_file = f"{output_dir}/freenrg-MBAR-run_{str(run_no).zfill(2)}_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.sh"
        with open(slurm_file, "w") as file:
            file.write(script_content)

        # Make script executable
        _os.chmod(slurm_file, 0o755)

        # Submit to the virtual queue
        cmd_list = [
            "--chdir",
            f"{output_dir}",
            f"{slurm_file}",
        ]  # The virtual queue adds sbatch
        slurm_file_base = _get_slurm_file_base(slurm_file)
        job = virtual_queue.submit(cmd_list, slurm_file_base=slurm_file_base)
        # Update the virtual queue to submit the job
        virtual_queue.update()
        jobs.append(job)

    # Wait for the job to complete if we've specified wait
    if wait:
        for job in jobs:
            while job in virtual_queue.queue:
                _sleep(30)
                virtual_queue.update()

    return jobs, mbar_out_files, tmp_simfiles

def _generate_legacy_mbar_script(run_somd_dir: str, mbar_command: str) -> str:
    """
    Legacy method for generating MBAR SLURM scripts by copying headers from run_somd.sh.
    Used as fallback if the SLURM generator fails.
    
    Parameters
    ----------
    run_somd_dir : str
        Directory containing run_somd.sh file
    mbar_command : str
        The MBAR command to execute
        
    Returns
    -------
    str
        SLURM script content
    """
    # Get the header from run_somd.sh
    header_lines = []
    try:
        with open(f"{run_somd_dir}/run_somd.sh", "r") as file:
            for line in file.readlines():
                if line.startswith("#SBATCH") or line.startswith("#!/bin/bash"):
                    header_lines.append(line)
                else:
                    break
    except FileNotFoundError:
        # Minimal fallback header
        header_lines = [
            "#!/bin/bash\n",
            "#SBATCH --job-name=mbar_analysis\n",
            "#SBATCH --time=02:00:00\n",
            "#SBATCH --cpus-per-task=1\n",
            "#SBATCH --mem=4G\n",
            "\n"
        ]
    
    # Add the MBAR command
    header_lines.append(f"{mbar_command}\n")
    
    return "".join(header_lines)


def collect_mbar_slurm(
    output_dir: str,
    run_nos: _List[int],
    jobs: _List[_Job],
    mbar_out_files: _List[str],
    virtual_queue: _VirtualQueue,
    delete_outfiles: bool = False,
    tmp_simfiles: _List[str] = [],
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
    tmp_simfiles : List[str], Optional, default: []
        The paths to the temporary truncated simfiles, so that they can be
        cleaned up later.

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
    for tmp_simfile in tmp_simfiles:
        _subprocess.run(["rm", tmp_simfile])

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
