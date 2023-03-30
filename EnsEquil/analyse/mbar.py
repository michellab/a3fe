"""
Functionality for running mbar on SOMD output files. This uses
pymbar through SOMD
"""

import numpy as _np
import os as _os
import glob as _glob
import subprocess as _subprocess
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

from ..read._process_somd_files import (
    read_mbar_result as _read_mbar_result, 
    write_truncated_sim_datafile as _write_truncated_sim_datafile
)

def run_mbar(output_dir: str,
             ensemble_size: int,
             percentage: float = 100,
             subsampling: bool = False,
             temperature: float = 298,
             delete_outfiles = False) -> _Tuple[_np.ndarray, _np.ndarray, _List[str]]:
    """
    Run MBAR on SOMD output files.

    Parameters
    ----------
    output_dir : str
        The path to the output directory
    ensemble_size : int
        The number of simulations in the ensemble
    percentage : float, Optional, default: 100
        The percentage of the data to use for MBAR, starting from 
        the start of the simulation. If this is less than 100,
        data will be discarded from the end of the simulation.
        data will be discarded from the start of the simulation.
    subsampling : bool, Optional, default: False
        Whether to use subsampling for MBAR.
    temperature : float, Optional, default: 298
        The temperature of the simulations, in Kelvin.
    delete_outfiles : bool, Optional, default: False
        Whether to delete the MBAR analysis output files after the free
        energy change and errors have been extracted.

    Returns
    -------
    free_energies : np.ndarray
        The free energies from each run, in kcal mol-1.
    errors : np.ndarray
        The mbar errors on the free energies from each run, in kcal mol-1.
    mbar_out_files : List[str]
        The paths to the MBAR output files.
    """
    # Check that the simfiles actually exist
    simfiles = _glob.glob(f"{output_dir}/lambda*/run_*/simfile_equilibrated.dat")
    if len(simfiles) == 0:
        raise FileNotFoundError("No equilibrated simfiles found. Have you run the simulations "
                                 "and checked for equilibration?")

    # If percent is less than 100, create temporary truncated simfiles
    tmp_simfiles = [] # Clean these up afterwards
    if percentage < 100:
        for simfile in simfiles:
            tmp_simfile = _os.path.join(_os.path.dirname(simfile), "simfile_truncated.dat")
            tmp_simfiles.append(tmp_simfile)
            _write_truncated_sim_datafile(simfile, tmp_simfile, percentage/100)

    # Run MBAR using pymbar through SOMD
    mbar_out_files = []
    for run in range(1, ensemble_size + 1):
        outfile = f"{output_dir}/freenrg-MBAR-run_{str(run).zfill(2)}_{round(percentage)}_percent.dat"
        mbar_out_files.append(outfile)
        with open(outfile, "w") as ofile:
            cmd_list = ["analyse_freenrg",
                         "mbar", 
                         "-i", f"{output_dir}/lambda*/run_{str(run).zfill(2)}/simfile_truncated.dat",
                         "-p", "100", 
                         "--overlap", 
                         "--temperature", f"{temperature}"]
            if subsampling:
                cmd_list.append("--subsampling")
            _subprocess.run(cmd_list, stdout=ofile)

    free_energies = _np.array([_read_mbar_result(ofile)[0] for ofile in mbar_out_files]) 
    errors = _np.array([_read_mbar_result(ofile)[1] for ofile in mbar_out_files])

    if delete_outfiles:
        for ofile in mbar_out_files:
            _subprocess.run(["rm", ofile])
        mbar_out_files = []

    # Clean up temporary simfiles
    for tmp_simfile in tmp_simfiles:
        _subprocess.run(["rm", tmp_simfile])

    return free_energies, errors, mbar_out_files
