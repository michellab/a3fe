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
        Column 0: Total energy
        Column 1: dH/dlambda
        Column 2+: Delta H to other lambda states
        Last column: pV term
    """
    times = []
    data = []

    with open(xvg_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("@"):
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

    # Find start of data
    start_data_idx = None
    for i, line in enumerate(lines):
        if not line.startswith("#") and not line.startswith("@"):
            start_data_idx = i
            break

    if start_data_idx is None:
        raise ValueError(f"No data found in xvg file: {xvg_file}.")

    # Count data lines
    data_lines = [line for line in lines[start_data_idx:] if line.strip() != ""]
    n_data = len(data_lines)

    # Calculate indices
    start_reading_idx = round(n_data * fraction_initial) + start_data_idx
    end_reading_idx = round(n_data * fraction_final) + start_data_idx - 1

    if start_reading_idx < start_data_idx:
        start_reading_idx = start_data_idx

    if start_reading_idx >= end_reading_idx:
        raise ValueError(f"Insufficient data to write truncated xvg file: {xvg_file}.")
    if end_reading_idx - start_reading_idx < 50:
        _warn(f"Very little data (< 50 lines) to write truncated xvg file: {xvg_file}.")

    # Write output
    with open(outfile, "w") as f:
        for i, line in enumerate(lines):
            # Write header
            if i < start_data_idx:
                f.write(line)
            # Write desired data
            if i >= start_reading_idx and i <= end_reading_idx:
                f.write(line)
