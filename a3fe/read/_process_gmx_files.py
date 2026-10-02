"""Functionality to process GROMACS output files."""

from typing import Tuple as _Tuple
from warnings import warn as _warn

import numpy as _np


def read_xvg_dhdl(xvg_file: str) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Read dH/dlambda and delta H data from GROMACS .xvg file.

    Parameters
    ----------
    xvg_file : str
        Path to .xvg file

    Returns
    -------
    times : np.ndarray
        Time points in ps
    data : np.ndarray
        Energy data matrix [n_samples, n_columns] in kJ/mol
        Column 0: Total or potential energy
        Columns 1-3: Coulomb, van der Waals, and bonded dH/dlambda
        Columns 4 onwards: Delta H to each lambda state, followed by a pV term
        when pressure coupling is active
    """
    times = []
    data = []

    with open(xvg_file, "r") as f:
        for line in f:
            if line.startswith(("#", "@", "&")) or not line.strip():
                continue
            parts = line.split()
            times.append(float(parts[0]))
            data.append([float(x) for x in parts[1:]])

    return _np.array(times), _np.array(data)


def write_truncated_xvg(
    xvg_file: str, outfile: str, fraction_final: float, fraction_initial: float = 0
) -> None:
    """
    Write truncated .xvg file (mirrors write_truncated_sim_datafile for SOMD).

    Parameters
    ----------
    xvg_file : str
        Input .xvg file
    outfile : str
        Output .xvg file
    fraction_final : float
        Fraction of data to keep (0-1)
    fraction_initial : float
        Fraction of initial data to discard (0-1)
    """
    for frac in [fraction_final, fraction_initial]:
        if frac < 0 or frac > 1:
            raise ValueError(f"Invalid fraction: {frac}. Must be between 0 and 1.")
    if fraction_final <= fraction_initial:
        raise ValueError(f"Invalid fractions: {fraction_final} <= {fraction_initial}.")

    with open(xvg_file, "r") as f:
        lines = f.readlines()

    header_lines = [line for line in lines if line.startswith(("#", "@"))]
    data_lines = [
        line for line in lines if line.strip() and not line.startswith(("#", "@", "&"))
    ]
    if not data_lines:
        raise ValueError(f"No data found in xvg file: {xvg_file}.")

    n_data = len(data_lines)
    start_idx = round(n_data * fraction_initial)
    end_idx = round(n_data * fraction_final)
    if start_idx >= end_idx:
        raise ValueError(f"Insufficient data to write truncated xvg file: {xvg_file}.")
    if end_idx - start_idx < 50:
        _warn(f"Very little data (< 50 lines) to write truncated xvg file: {xvg_file}.")

    with open(outfile, "w") as f:
        f.writelines(header_lines)
        f.writelines(data_lines[start_idx:end_idx])
