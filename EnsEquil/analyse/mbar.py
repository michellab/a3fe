"""
Functionality for running mbar on SOMD output files. This uses
pymbar through SOMD
"""

__all__ = ["run_mbar"]

import glob as _glob
import os as _os
import subprocess as _subprocess
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np

from ..read._process_somd_files import \
    read_mbar_gradients as _read_mbar_gradients
from ..read._process_somd_files import read_mbar_result as _read_mbar_result
from ..read._process_somd_files import \
    write_truncated_sim_datafile as _write_truncated_sim_datafile


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
