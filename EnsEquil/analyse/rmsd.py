"""Functionality to compute RMSDs"""

import BioSimSpace as _BSS
import glob as _glob
from MDAnalysis.analysis import align as _align
from MDAnalysis.analysis.rms import RMSD as _RMSD
from MDAnalysis import Universe as _Universe
import numpy as _np
import os as _os
import subprocess as _subprocess
from tempfile import TemporaryDirectory as _TemporaryDirectory
from typing import List as _List, Tuple as _Tuple

def get_rmsd(input_dirs: _List[str],
                 selection: str,
                 tot_simtime: float) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Return the RMSD for a given MDAnalysis selection. Reference frame taken as
    the first frame, regardless of whether the total equilibration time is 0 or not.

    Parameters
    ----------
    input_dirs : List[str]
        List of absolute paths to the output directories containing the trajectory information and the topology files.
    selection: str
        The selection, written using the MDAnalysis selection language, to use for the calculation of RMSD.
    tot_simtime : float
        The total simulation time per simulation.

    Returns
    -------
    rmsds: _np.ndarray
        An array of the RMSDs with reference to the first frame
    times: _np.ndarray
        An array of the times, in ns, corresponding to the RMSDs
    """
    rmsds_list = []

    # Iterate through the runs and collect results
    for i, input_dir in enumerate(input_dirs):

        # Get the topology and trajectory files
        # Need to copy the topology file to a parm7 extension so that this is recognised by MDAnanlysis
        with _TemporaryDirectory() as tmpdir:
            top_file = _os.path.join(tmpdir, "somd.parm7")
            _subprocess.run(["cp", f"{input_dir}/somd.prm7" , top_file],
                             check=True)

            # Convert to .gro file
            sys = _BSS.IO.readMolecules([top_file, _os.path.join(input_dir, "somd.rst7")])
            _BSS.IO.saveMolecules(_os.path.join(tmpdir, "somd"), sys, "grotop")
            top_file = _os.path.join(tmpdir, "somd.gro")
            
            # There could be multiple trajectory files if the simulation has been restarted
            traj_files = _glob.glob(_os.path.join(input_dir, "*.dcd"))

            # Collect rmsds accross all trajectory files for this run
            rmsds_run = []
            for traj_file in traj_files:
                mobile = _Universe(top_file, traj_file)
                R = _RMSD(mobile, select=selection, groupselections=[selection], ref_frame=0) 
                R.run()
                rmsd_results = R.results.rmsd.T
                rmsds_run.extend(rmsd_results[3])
            rmsds_list.append(rmsds_run)

            # Calculate the times - this only needs to be done once as all the times should be the same
            if i == 0:
                times = _np.linspace(start=0, stop=tot_simtime, num=len(rmsds_run))

    rmsds = _np.array(rmsds_list)

    return rmsds, times
