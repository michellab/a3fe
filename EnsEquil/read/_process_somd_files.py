"""Functionality to manipulate SOMD files."""

from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from warnings import warn as _warn

import numpy as _np

from .exceptions import ReadError as _ReadError


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
    with open(simfile, "r") as f:
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
    with open(simfile, "r") as f:
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
    with open(simfile, "w") as f:
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
    with open(outfile, "r") as f:
        lines = f.readlines()
    try:
        # The free energy is the 5th last line of the file
        free_energy = float(lines[-4].split(",")[0])
        free_energy_err = float(lines[-4].split(",")[1].split()[0])
    except (IndexError, ValueError):
        raise _ReadError(f"Could not read free energy from {outfile}.")

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
    with open(outfile, "r") as f:
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


def read_mbar_pmf(outfile: str) -> _Tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
    Read the PMF from the mbar outfile.

    Parameters
    ----------
    outfile : str
        The name of the output file.

    Returns
    -------
    lam : np.ndarray
        The lambda values.
    pmf : np.ndarray
        The PMF in kcal/mol.
    pmf_err : np.ndarray
        The MBAR error in the PMF in kcal/mol.
    """
    with open(outfile, "r") as f:
        lines = f.readlines()
    lam = []
    pmf = []
    pmf_err = []
    in_pmf = False
    for line in lines:
        if line.startswith("#PMF from MBAR in kcal/mol"):
            in_pmf = True
            continue
        if line.startswith("#"):
            in_pmf = False
            continue
        if in_pmf:
            lam.append(float(line.split()[0]))
            pmf.append(float(line.split()[1]))
            pmf_err.append(float(line.split()[2]))

    return _np.array(lam), _np.array(pmf), _np.array(pmf_err)


def read_mbar_gradients(
    outfile: str,
) -> _Tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
    Get the gradients of the free energy changes from the MBAR (not TI) results.
    Note that there will be n-1 free energy changes for n lambda windows.

    Parameters
    ----------
    outfile : str
        The name of the output file.

    Returns
    -------
    av_lams : np.ndarray
        The average lambda value for the free energy change. For example,
        if the change is between lam = 0 and lam = 0.1, then the average
        lambda value will be 0.05.
    grads : np.ndarray
        The gradients of the free energy changes in kcal/mol.
    grads_err : np.ndarray
        The MBAR error in the gradients in kcal/mol.
    """
    with open(outfile, "r") as f:
        lines = f.readlines()
    av_lams = []
    inter_dgs = []
    inter_dgs_err = []
    in_section = False
    for line in lines:
        if line.startswith("#DG from neighbouring lambda in kcal/mol"):
            in_section = True
            continue
        if line.startswith("#"):
            in_section = False
            continue
        if in_section:
            delta_lam = float(line.split()[1]) - float(line.split()[0])
            av_lam = (float(line.split()[0]) + float(line.split()[1])) / 2
            av_lams.append(av_lam)
            inter_dgs.append(float(line.split()[2]) / delta_lam)
            inter_dgs_err.append(float(line.split()[3]) / delta_lam)

    return _np.array(av_lams), _np.array(inter_dgs), _np.array(inter_dgs_err)


def write_truncated_sim_datafile(
    simfile: str, outfile: str, fraction_final: float, fraction_initial: float = 0
) -> None:
    """
    Write a truncated simfile, discarding the specified fraction of the initial
    and final steps.

    Parameters
    ----------
    simfile : str
        The name of the input simfile.
    outfile : str
        The name of the output simfile.
    fraction_final : float
        The fraction of the data after which data should be discarded. For
        example, if the simulation was run for 1000 steps, and fraction_final =
        0.9, then the final 100 steps will be discarded.
    fraction_initial : float, optional, default=0
        The fraction of the initial number of steps to discard. For example, if
        the simulation was run for 1000 steps, and fraction_initial = 0.1, then
        the initial 100 steps will be discarded.

    Notes
    -----
    No data is written at t = 0. Hence, if fraction initial = 0, then the
    written truncated simfile will actually start from the time point closest
    to zero.

    Returns
    -------
    None
    """
    # Check that the fractions are valid
    for frac in [fraction_final, fraction_initial]:
        if frac < 0 or frac > 1:
            raise ValueError(f"Invalid fraction: {frac}. Must be between 0 and 1.")
    if fraction_final <= fraction_initial:
        raise ValueError(f"Invalid fractions: {fraction_final} <= {fraction_initial}.")

    # Read the data
    with open(simfile, "r") as f:
        lines = f.readlines()
    final_idx = len([line for line in lines if line.strip() != ""]) - 1

    # First, find the indices from which to start and stop reading
    # The start data index is the first line which isn't a comment
    start_data_idx = None
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            start_data_idx = i
            break
    if start_data_idx is None:
        raise ValueError(f"No data found in simfile: {simfile}.")
    start_reading_idx = (
        # + 1 and -1 because no data is written at t = 0
        round((final_idx - start_data_idx + 1) * fraction_initial)
        + start_data_idx
        - 1
    )
    end_reading_idx = (
        round((final_idx - start_data_idx + 1) * fraction_final) + start_data_idx - 1
    )
    if start_reading_idx < start_data_idx:
        # This can occur because no data is written at t = 0
        start_reading_idx = start_data_idx

    # Check that we have data to write and warm the user if not
    if start_reading_idx == end_reading_idx:
        raise ValueError(f"Insufficient data to write truncated simfile: {simfile}.")
    if end_reading_idx - start_reading_idx < 50:
        _warn(f"Very little data (< 50 lines) to write truncated simfile: {simfile}.")

    # Write the output file
    with open(outfile, "w") as f:
        for i, line in enumerate(lines):
            # Write header
            if i < start_data_idx:
                f.write(line)
            # Write desired data
            if i >= start_reading_idx:
                f.write(line)
            if i == end_reading_idx:
                break
        # Finish by writing an empty line
        f.write("\n")
