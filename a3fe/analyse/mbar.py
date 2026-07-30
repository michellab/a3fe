"""Functionality for running MBAR on SOMD and GROMACS output files."""

__all__ = ["run_mbar"]

import glob as _glob
import os as _os
import subprocess as _subprocess
from time import sleep as _sleep
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple

import numpy as _np
from scipy.constants import gas_constant as _R

from ..configuration import EngineType as _EngineType
from ..configuration import SlurmConfig as _SlurmConfig
from ..read._process_gmx_files import write_truncated_xvg as _write_truncated_xvg
from ..read._process_somd_files import read_mbar_gradients as _read_mbar_gradients
from ..read._process_somd_files import read_mbar_result as _read_mbar_result
from ..read._process_somd_files import (
    write_truncated_sim_datafile as _write_truncated_sim_datafile,
)
from ..run._virtual_queue import Job as _Job
from ..run._virtual_queue import VirtualQueue as _VirtualQueue


def run_mbar(
    output_dir: str,
    run_nos: _List[int],
    percentage_end: float = 100,
    percentage_start: float = 0,
    subsampling: bool = False,
    delete_outfiles: bool = False,
    equilibrated: bool = True,
    engine_type: _EngineType = _EngineType.SOMD,
    temperature: float = 298.15,
) -> _Tuple[_np.ndarray, _np.ndarray, _List[str], _Dict[str, _Dict[str, _np.ndarray]]]:
    """
    Run MBAR on output files.

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
    engine_type : EngineType, Optional, default: EngineType.SOMD
        The engine type that produced the output files.
    temperature : float, Optional, default: 298.15
        Temperature in K to use when analysing GROMACS output files.

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
    if engine_type == _EngineType.GROMACS:
        return _run_mbar_gromacs(
            output_dir=output_dir,
            run_nos=run_nos,
            percentage_end=percentage_end,
            percentage_start=percentage_start,
            subsampling=subsampling,
            delete_outfiles=delete_outfiles,
            equilibrated=equilibrated,
            temperature=temperature,
        )

    tmp_files = _prepare_simfiles(
        output_dir=output_dir,
        run_nos=run_nos,
        percentage_end=percentage_end,
        percentage_start=percentage_start,
        equilibrated=equilibrated,
    )

    # Run MBAR using pymbar through SOMD
    mbar_out_files = []
    for run_no in run_nos:
        outfile = (
            f"{output_dir}/freenrg-MBAR-run_{str(run_no).zfill(2)}_"
            f"{round(percentage_end, 3)}_end_"
            f"{round(percentage_start, 3)}_start.dat"
        )
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
    for tmp_file in tmp_files:
        _subprocess.run(["rm", tmp_file])

    return free_energies, errors, mbar_out_files, mbar_grads


def _run_mbar_gromacs(
    output_dir: str,
    run_nos: _List[int],
    percentage_end: float = 100,
    percentage_start: float = 0,
    subsampling: bool = False,
    delete_outfiles: bool = False,
    equilibrated: bool = True,
    temperature: float = 298.15,
) -> _Tuple[_np.ndarray, _np.ndarray, _List[str], _Dict[str, _Dict[str, _np.ndarray]]]:
    """
    Run MBAR on GROMACS dhdl output files using alchemlyb and pymbar.
    """
    if subsampling:
        raise NotImplementedError("Subsampling is not implemented for GROMACS MBAR.")

    import pandas as _pd
    import pymbar as _pymbar
    from alchemlyb.parsing.gmx import extract_u_nk as _extract_u_nk

    tmp_files = _prepare_xvg_files(
        output_dir=output_dir,
        run_nos=run_nos,
        percentage_end=percentage_end,
        percentage_start=percentage_start,
        equilibrated=equilibrated,
    )

    mbar_out_files = []
    free_energies = []
    errors = []
    mbar_grads = {}
    kt = _R * temperature / 4184

    for run_no in run_nos:
        xvg_files = _get_gromacs_xvg_files(
            output_dir=output_dir,
            run_no=run_no,
            percentage_end=percentage_end,
            percentage_start=percentage_start,
        )
        u_nk = _pd.concat(
            [_extract_u_nk(xvg_file, temperature) for xvg_file in xvg_files]
        )
        u_nk = u_nk.sort_index(level=u_nk.index.names[1:])
        groups = u_nk.groupby(level=u_nk.index.names[1:])
        n_k = [
            len(groups.get_group(state)) if state in groups.groups else 0
            for state in u_nk.columns
        ]
        mbar = _pymbar.MBAR(u_nk.T, n_k)
        delta_f, d_delta_f, _ = mbar.getFreeEnergyDifferences(return_theta=True)

        outfile = (
            f"{output_dir}/freenrg-MBAR-run_{str(run_no).zfill(2)}_"
            f"{round(percentage_end, 3)}_end_"
            f"{round(percentage_start, 3)}_start.dat"
        )
        mbar_out_files.append(outfile)

        lam_vals = _get_lambda_values_from_xvg_files(xvg_files)
        delta_f = _np.array(delta_f) * kt
        d_delta_f = _np.array(d_delta_f) * kt
        overlap = _get_overlap_matrix(mbar)

        free_energies.append(delta_f[0, -1])
        errors.append(d_delta_f[0, -1])

        _write_gromacs_mbar_output(
            outfile=outfile,
            xvg_files=xvg_files,
            lam_vals=lam_vals,
            delta_f=delta_f,
            d_delta_f=d_delta_f,
            overlap=overlap,
            subsampling=subsampling,
        )

        mbar_grads[f"run_{run_no}"] = {}
        lam_vals, grads, grad_errors = _read_mbar_gradients(outfile)
        mbar_grads[f"run_{run_no}"]["lam_vals"] = lam_vals
        mbar_grads[f"run_{run_no}"]["grads"] = grads
        mbar_grads[f"run_{run_no}"]["grad_errors"] = grad_errors

    if delete_outfiles:
        for ofile in mbar_out_files:
            _subprocess.run(["rm", ofile])
        mbar_out_files = []

    for tmp_file in tmp_files:
        _subprocess.run(["rm", tmp_file])

    return _np.array(free_energies), _np.array(errors), mbar_out_files, mbar_grads


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
        outfile = (
            f"{output_dir}/freenrg-MBAR-run_{str(run_no).zfill(2)}_"
            f"{round(percentage_end, 3)}_end_"
            f"{round(percentage_start, 3)}_start.dat"
        )
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
                _sleep(slurm_config.job_submission_wait)
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
    tmp_files = []  # Clean these up afterwards
    for simfile in simfiles:
        tmp_file = _os.path.join(
            _os.path.dirname(simfile),
            f"simfile_truncated_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.dat",
        )
        tmp_files.append(tmp_file)
        _write_truncated_sim_datafile(
            simfile,
            tmp_file,
            fraction_final=percentage_end / 100,
            fraction_initial=percentage_start / 100,
        )

    return tmp_files


def _prepare_xvg_files(
    output_dir: str,
    run_nos: _List[int],
    percentage_end: float = 100,
    percentage_start: float = 0,
    equilibrated: bool = True,
) -> _List[str]:
    """
    Helper function to prepare the xvg files for GROMACS MBAR analysis.
    """
    file_name = "prod_equilibrated.xvg" if equilibrated else "prod.xvg"
    xvg_files = _glob.glob(f"{output_dir}/lambda*/run_*/prod/{file_name}")
    if run_nos is not None:
        xvg_files = [
            xvg_file
            for xvg_file in xvg_files
            if int(
                _os.path.basename(_os.path.dirname(_os.path.dirname(xvg_file))).split(
                    "_"
                )[-1]
            )
            in run_nos
        ]

    if len(xvg_files) == 0:
        raise FileNotFoundError(
            "No equilibrated GROMACS xvg files found. Have you run the simulations "
            "and checked for equilibration?"
        )

    tmp_files = []
    for xvg_file in xvg_files:
        tmp_file = _os.path.join(
            _os.path.dirname(xvg_file),
            f"prod_truncated_{round(percentage_end, 3)}_end_"
            f"{round(percentage_start, 3)}_start.xvg",
        )
        tmp_files.append(tmp_file)
        _write_truncated_xvg(
            xvg_file,
            tmp_file,
            fraction_final=percentage_end / 100,
            fraction_initial=percentage_start / 100,
        )

    return tmp_files


def _get_gromacs_xvg_files(
    output_dir: str,
    run_no: int,
    percentage_end: float = 100,
    percentage_start: float = 0,
) -> _List[str]:
    """Return the truncated GROMACS xvg files for a single run."""
    xvg_files = _glob.glob(
        f"{output_dir}/lambda*/run_{str(run_no).zfill(2)}/prod/"
        f"prod_truncated_{round(percentage_end, 3)}_end_"
        f"{round(percentage_start, 3)}_start.xvg"
    )
    xvg_files = sorted(xvg_files, key=_get_lambda_value_from_xvg_file)
    if len(xvg_files) == 0:
        raise FileNotFoundError(f"No GROMACS xvg files found for run {run_no}.")
    return xvg_files


def _get_lambda_values_from_xvg_files(xvg_files: _List[str]) -> _np.ndarray:
    """Return lambda values inferred from lambda directory names."""
    return _np.array(
        [_get_lambda_value_from_xvg_file(xvg_file) for xvg_file in xvg_files]
    )


def _get_lambda_value_from_xvg_file(xvg_file: str) -> float:
    """Return the lambda value inferred from a GROMACS xvg file path."""
    lambda_dir = _os.path.basename(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(xvg_file)))
    )
    return float(lambda_dir.split("_")[-1])


def _get_overlap_matrix(mbar) -> _np.ndarray:
    """Return an overlap matrix from a pymbar or alchemlyb MBAR estimator."""
    overlap = getattr(mbar, "overlap_matrix", None)
    if overlap is not None:
        return _np.array(overlap)
    mbar_objects = [mbar]
    if hasattr(mbar, "_mbar"):
        mbar_objects.append(mbar._mbar)
    for mbar_object in mbar_objects:
        for method_name in ["computeOverlap", "compute_overlap"]:
            if hasattr(mbar_object, method_name):
                overlap = getattr(mbar_object, method_name)()
                if isinstance(overlap, dict):
                    overlap = overlap["matrix"]
                return _np.array(overlap)
    raise AttributeError("Could not extract an overlap matrix from the MBAR result.")


def _write_gromacs_mbar_output(
    outfile: str,
    xvg_files: _List[str],
    lam_vals: _np.ndarray,
    delta_f: _np.ndarray,
    d_delta_f: _np.ndarray,
    overlap: _np.ndarray,
    subsampling: bool = False,
) -> None:
    """Write GROMACS MBAR results in the existing a3fe MBAR output format."""
    warning = ""
    if not subsampling:
        warning = (
            "  #WARNING SUBSAMPLING DISABLED, CONSIDER RERUN WITH SUBSAMPLING OPTION"
        )

    with open(outfile, "w") as ofile:
        ofile.write(f"# Analysing data contained in file(s) {xvg_files}\n")
        ofile.write("# MBAR analysis performed with alchemlyb on GROMACS xvg files\n")
        ofile.write("#Overlap matrix\n")
        for row in overlap:
            ofile.write(" ".join(f"{val:.4f}" for val in row) + "\n")

        ofile.write("#DG from neighbouring lambda in kcal/mol\n")
        for i in range(len(lam_vals) - 1):
            ofile.write(
                f"{lam_vals[i]:.4f} {lam_vals[i + 1]:.4f} "
                f"{delta_f[i, i + 1]:.4f} {d_delta_f[i, i + 1]:.4f}\n"
            )

        ofile.write("#PMF from MBAR in kcal/mol\n")
        for i, lam_val in enumerate(lam_vals):
            ofile.write(f"{lam_val:.4f} {delta_f[0, i]:.4f} {d_delta_f[0, i]:.4f}\n")

        ofile.write("#MBAR free energy difference in kcal/mol: \n")
        ofile.write(f"{delta_f[0, -1]:.6f}, {d_delta_f[0, -1]:.6f}{warning}\n")
