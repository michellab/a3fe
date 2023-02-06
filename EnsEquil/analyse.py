"""Functions for analysing the results of the ensemble of simulations."""

import os as _os
import subprocess as _subprocess
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional


def analyse(lam_winds: _List["LamWindow"]) -> None:
    """
    Analyse the results of the ensemble of simulations. Requires that 
    all lambda windows have equilibrated.

    Parameters
    ----------
    lam_winds : List[EnsEquil.run.LamWindow]
        List of lambda windows.

    Returns
    -------
    None
    """
    # Check that all lambda windows have equilibrated
    for win in lam_winds:
        if not win.equilibrated:
            raise ValueError(f"Analysis failed: lambda window {win.lam} has not equilibrated.")
       
    # Remove unequilibrated data from the equilibrated output directory
    for win in lam_winds:
        equil_time = win.equil_time
        equil_index = int(equil_time / (win.sims[0].timestep * win.sims[0].nrg_freq))
        for sim in win.sims:
            in_simfile = sim.output_subdir + "/simfile.dat"
            out_simfile = sim.output_subdir + "/simfile_equilibrated.dat"
            _write_equilibrated_data(in_simfile, out_simfile, equil_index)

    # Analyse data with MBAR and compute uncertainties
    output_dir = lam_winds[0].output_dir
    ensemble_size = lam_winds[0].ensemble_size
    # This is nasty - assumes naming always the same
    for run in range(1, ensemble_size + 1):
        _subprocess.run(["/home/finlayclark/sire.app/bin/analyse_freenrg",
                         "mbar", "-i", f"{output_dir}/lambda*/run_0{run}/simfile.dat",
                         "-p", "100", "--overlap","--temperature", 
                         "298.0", ">", f"freenrg-MBAR-run_{run}.dat"])
    
    # TODO: Make convergence plots (which should be flat)


def _write_equilibrated_data(in_simfile: str, out_simfile: str, equil_index: int) -> None:
    """
    Remove unequilibrated data from a simulation file and write a new
    file containing only the equilibrated data.

    Parameters
    ----------
    in_simfile : str
        Path to simulation file.
    out_simfile : str
        Path to new simulation file, containing only the equilibrated data
        from the original file.
    equil_index : int
        Index of the first equilibrated frame, given by 
        equil_time / (timestep * nrg_freq)

    Returns
    -------
    None
    """
    with open(in_simfile, "r") as ifile:
        lines = ifile.readlines()

    # Figure out how many lines come before the data
    non_data_lines = 0
    for line in lines:
        if line.startswith("#"):
            non_data_lines += 1
        else:
            break

    # Overwrite the original file with one containing only the equilibrated data
    with open(out_simfile, "w") as ofile:
        for line in lines[equil_index + non_data_lines:]:
            ofile.write(line)



