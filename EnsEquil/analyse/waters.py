"""Functions to analyse binding site waters."""

from typing import (
    List as _List,
    Tuple as _Tuple,
    Optional as _Optional,
    Callable as _Callable,
)

import glob as _glob
import MDAnalysis as _mda
import numpy as _np
import os as _os

from multiprocessing import Pool as _Pool


def get_av_waters_simulation(
    input_dir: str,
    percent_traj: float,
    index: int,
    length: float,
    index2: _Optional[int] = None,
    length2: _Optional[float] = None,
) -> float:
    """
    Calculate average number of waters within given distance
    of an atom (or two atoms) with given index over the
    specified percentage of the end of the trajectory.

    Parameters
    ----------
    input_dir: str
        Absolute path to the input directory containing the
        trajectory information and the topology files.

    percent_traj : float
        percentage of trajectory (beginning from end) over
        which to average

    index : int
        Atom from which distance is calculated

    length : float
        Distance in Angstrom

    index2 : int, optional, default=None
        Optional. Index of second atom from which water must be within a specified distance

    length2 : float, optional, default=None
        Optional. Distance (Angstrom) from second atom which water must be within

    Returns
    -------
    avg_close_waters : float
        Average number of waters within the specified distance(s) of the specified atom(s)
    """
    # Read all the trajectories into a single MDA universe
    traj_files = _glob.glob(_os.path.join(input_dir, "*.dcd"))
    top_file = _os.path.join(input_dir, "somd.parm7")
    # Create a soft link to the topology file named parm7 so that MDAnalysis can read it
    # Only create the symlink if it doesn't already exist
    if not _os.path.exists(top_file):
        _os.symlink(_os.path.join(input_dir, "somd.prm7"), top_file)

    no_close_waters = []
    selection_str = (
        f"resname WAT and sphzone {length} index {index}"
        if index2 is None
        else f"resname WAT and (sphzone {length} index {index}) and (sphzone {length2} index {index2})"
    )

    # Can't load all trajectories at once with dcd trajectories, so iterate through
    for traj_file in traj_files:
        u = _mda.Universe(top_file, traj_file)
        for frame in u.trajectory:
            no_close_waters.append(
                len(u.select_atoms(selection_str)) / 3
            )  # /3 as returns all atoms in water

    n_frames = len(no_close_waters)
    first_frame = round(
        n_frames - ((percent_traj / 100) * n_frames)
    )  # Index of first frame to be used for analysis

    avg_close_waters = _np.mean(no_close_waters[first_frame:])
    return avg_close_waters


def get_av_waters_lambda_window(
    simulations: _List["Simulation"],
    percent_traj: float,
    lam_val: float,
    index: int,
    length: float,
    run_nos: _List[int],
    index2: _Optional[int] = None,
    length2: _Optional[float] = None,
    print_fn: _Callable[[str], None] = print,
) -> _np.ndarray:
    """
    Calculate average number of waters within given distance
    of an atom (or two atoms) with given index over the
    specified percentage of the end of the trajectory,
    for all simulations for all runs for all lambda windows
    supplied.

    Parameters
    ----------
    simulations : List[LamWindow]
        List of Simulatino objects for which to calculate average number of waters

    percent_traj : float
        percentage of trajectory (beginning from end) over
        which to average

    lam_val : float
        Current value of lambda. Used for printing output.

    index : int
        Atom from which distance is calculated

    length : float
        Distance in Angstrom

    run_nos : List[int]
        List of run numbers to include in the analysis.

    index2 : int, optional, default=None
        Optional. Index of second atom from which water must be within a specified distance

    length2 : float, optional, default=None
        Optional. Distance (Angstrom) from second atom which water must be within

    print_fn : Optional[Callable[[str], None]], default=print
        Optional. Function to use for printing output.

    Returns
    -------
    avg_close_waters: _np.ndarray
        Average number of waters within the specified distance(s) of the specified atom(s) for each
        lambda window for each run. Shape is (n_runs).
    """
    run_nos = simulations[0]._get_valid_run_nos(run_nos)
    n_runs = len(run_nos)
    avg_close_waters = _np.full(n_runs, _np.nan)
    for i, run_no in enumerate(run_nos):
        print_fn(
            f"Calculating average number of waters for run {i+1} of {n_runs} for lambda window {lam_val}"
        )
        sim = simulations[run_no - 1]
        avg_close_waters[i] = get_av_waters_simulation(
            sim.output_dir,
            percent_traj,
            index,
            length,
            index2,
            length2,
        )

    return avg_close_waters


def get_av_waters_stage(
    lam_windows: _List["LamWindow"],
    percent_traj: float,
    index: int,
    length: float,
    index2: _Optional[int] = None,
    length2: _Optional[float] = None,
    run_nos: _Optional[_List[int]] = None,
    print_fn: _Optional[_Callable[[str], None]] = print,
) -> _np.ndarray:
    """
    Calculate average number of waters within given distance
    of an atom (or two atoms) with given index over the
    specified percentage of the end of the trajectory,
    for all simulations for all runs for all lambda windows
    supplied.

    Parameters
    ----------
    lam_windows : List[LamWindow]
        List of LamWindow objects for which to calculate average number of waters

    percent_traj : float
        percentage of trajectory (beginning from end) over
        which to average

    index : int
        Atom from which distance is calculated

    length : float
        Distance in Angstrom

    index2 : int, optional, default=None
        Optional. Index of second atom from which water must be within a specified distance

    length2 : float, optional, default=None
        Optional. Distance (Angstrom) from second atom which water must be within

    run_nos : Optional[List[int]], default=None
        Optional. List of run numbers to include in the analysis. If None, all runs will be included.

    print_fn : Optional[Callable[[str], None]], default=print
        Optional. Function to use for printing output.

    Returns
    -------
    avg_close_waters: _np.ndarray
        Average number of waters within the specified distance(s) of the specified atom(s) for each
        lambda window for each run. Shape is (n_runs, n_lam_windows).
    """
    run_nos: _List["LamWindow"] = lam_windows[0]._get_valid_run_nos(run_nos)
    # Fill the array with Nans to start with
    avg_close_waters = _np.full((len(run_nos), len(lam_windows)), _np.nan)
    # Process the lambda windows in parallel
    with _Pool() as pool:
        per_lam_results = pool.starmap(
            get_av_waters_lambda_window,
            [
                (
                    lam_window.sims,
                    percent_traj,
                    lam_window.lam,
                    index,
                    length,
                    run_nos,
                    index2,
                    length2,
                    run_nos,
                    print_fn,
                )
                for lam_window in lam_windows
            ],
        )

    # Unpack the results into the array
    for i in range(len(lam_windows)):
        for j in range(len(run_nos)):
            avg_close_waters[j, i] = per_lam_results[i][j]

    return avg_close_waters
