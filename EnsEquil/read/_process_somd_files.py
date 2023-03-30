"""Functionality to manipulate SOMD files."""

import numpy as _np
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

def read_simfile_option(simfile: str, option: str) -> str:
    """Read an option from a SOMD simfile.

    Parameters
    ----------
    simfile : str
        The path to the simfile.
    option : str
        The option to read.
    Returns
    -------
    value : str
        The value of the option.
    """
    with open(simfile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.split("=")[0].strip() == option:
            value = line.split("=")[1].strip()
            return value
    raise ValueError(f"Option {option} not found in simfile {simfile}")

def write_simfile_option(simfile: str, option: str, value: str) -> None:
    """Write an option to a SOMD simfile.

    Parameters
    ----------
    simfile : str
        The path to the simfile.
    option : str
        The option to write.
    value : str
        The value to write.
    Returns
    -------
    None
    """
    # Read the simfile and check if the option is already present
    with open(simfile, 'r') as f:
        lines = f.readlines()
    option_line_idx = None
    for i, line in enumerate(lines):
        if line.split("=")[0].strip() == option:
                option_line_idx = i
                break

    # If the option is not present, append it to the end of the file
    if option_line_idx is None:
        lines.append(f"{option} = {value}\n")
    # Otherwise, replace the line with the new value
    else:
        lines[option_line_idx] = f"{option} = {value}\n"

    # Write the updated simfile
    with open(simfile, 'w') as f:
        f.writelines(lines)

def read_mbar_result(outfile: str) -> _Tuple[float, float]:
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

def read_overlap_mat(outfile: str) -> _np.ndarray:
    """ 
    Read the overlap matrix from the mbar outfile.

    Parameters
    ----------
    outfile : str
        The name of the output file.

    Returns
    -------
    overlap_mat : np.ndarray
        The overlap matrix.
    """
    with open(outfile, 'r') as f:
        lines = f.readlines()
    overlap_mat = []
    in_overlap_mat = False
    for line in lines:
        if line.startswith("#Overlap matrix"):
            in_overlap_mat = True
            continue
        if line.startswith("#"):
            in_overlap_mat = False
            continue
        if in_overlap_mat:
            overlap_mat.append([float(x) for x in line.split()])

    return _np.array(overlap_mat)

def write_truncated_sim_datafile(simfile: str, outfile: str, fraction: float) -> None:
    """ 
    Write a truncated simfile, with a certain percentage of the final number of steps
    discarded.

    Parameters
    ----------
    simfile : str
        The name of the input simfile.
    outfile : str
        The name of the output simfile.
    fraction : float
        The fraction of the final number of steps to discard.

    Returns
    -------
    None
    """
    with open(simfile, 'r') as f:
        lines = f.readlines()

    # Find the start of the data, the end of the data, and the number of steps
    with open(outfile, 'w') as f:
        n_lines = len(lines)
        start_data_idx = None
        end_data_idx = None
        for i, line in enumerate(lines):
            if start_data_idx is None:
                if not line.startswith("#"):
                    start_data_idx = lines.index(line)
                    end_data_idx = round((n_lines - start_data_idx) * fraction) + start_data_idx + 1 # +1 as no data written at t = 0
            else:
                if i == end_data_idx:
                    break
            f.write(line)
