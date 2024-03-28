"""Functionality to compute RMSDs"""

__all__ = ["get_rmsd"]

import glob as _glob
import os as _os
import subprocess as _subprocess
from tempfile import TemporaryDirectory as _TemporaryDirectory
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np
from MDAnalysis import Universe as _Universe
from MDAnalysis.analysis.rms import RMSD as _RMSD


def get_rmsd(
    input_dirs: _List[str],
    selection: str,
    tot_simtime: float,
    reference_traj: str,
    group_selection: _Optional[str] = None,
) -> _Tuple[_np.ndarray, _np.ndarray]:
    """
    Return the RMSD for a given MDAnalysis selection. Reference frame taken as
    the first frame, regardless of whether the total equilibration time is 0 or not.

    Note that if group_selection is None, the same selection as the one used for the
    RMSD calculation will be used, otherwise the group_selection will be used for the
    RMSD calculation.

    Parameters
    ----------
    input_dirs : List[str]
        List of absolute paths to the output directories containing the trajectory information and the topology files.
    selection: str
        The selection, written using the MDAnalysis selection language, to use for the calculation of RMSD.
    tot_simtime : float
        The total simulation time per simulation.
    reference_traj : str
        The absolute path to the reference trajectory file. The reference will be taken as the first frame of
        this trajectory.
    group_selection : Optional[str], default=None
        The selection, written using the MDAnalysis selection language, to use for the calculation of RMSD, after the
        alignment of the trajectory according to the selection passed as "selection". If None, the same selection as
        the one used for the RMSD calculation will be used.

    Returns
    -------
    rmsds: _np.ndarray
        An array of the RMSDs, in nm, for each simulation with reference to the
        first frame of the reference trajectory.
    times: _np.ndarray
        An array of the times, in ns, corresponding to the RMSDs
    """
    rmsds_list = []
    group_selection = selection if group_selection is None else group_selection

    # Iterate through the runs and collect results
    for i, input_dir in enumerate(input_dirs):
        # Get the topology and trajectory files
        # Need to copy the topology file to a parm7 extension so that this is recognised by MDAnanlysis
        with _TemporaryDirectory() as tmpdir:
            top_file = _os.path.join(tmpdir, "somd.parm7")
            _subprocess.run(["cp", f"{input_dir}/somd.prm7", top_file], check=True)
            reference = _Universe(top_file, reference_traj)
            # Ensure that the protein and ligand are whole and centered - note this is super slow
            prot_and_or_lig = reference.select_atoms("protein or resname LIG")
            not_prot_or_lig = reference.select_atoms("not (protein or resname LIG)")
            # transforms = [
            # _trans.unwrap(prot_and_or_lig),
            # _trans.center_in_box(prot_and_or_lig),
            # _trans.wrap(not_prot_or_lig),
            # ]
            # reference.trajectory.add_transformations(*transforms)

            # To avoid the atom selection changing between the reference and mobile (e.g. use of "within" keyword
            # followed by movement of the protein), extract the atoms matching the selections on the reference
            # and convert this into a new selection string
            # reference_selection = reference.select_atoms(selection)
            # selection = " ".join(
            # [f"index {i} or" for i in reference_selection.indices]
            # )[:-3]
            # reference_group_selection = reference.select_atoms(group_selection)
            # group_selection = " ".join(
            # [f"index {i} or" for i in reference_group_selection.indices]
            # )[:-3]

            # There could be multiple trajectory files if the simulation has been restarted
            traj_files = _glob.glob(_os.path.join(input_dir, "*.dcd"))

            # Collect rmsds accross all trajectory files for this run
            rmsds_run = []
            for traj_file in traj_files:
                mobile = _Universe(top_file, traj_file)
                # mobile.trajectory.add_transformations(*transforms)
                R = _RMSD(
                    atomgroup=mobile,
                    reference=reference,
                    select=selection,
                    groupselections=[group_selection],
                    ref_frame=0,
                )
                R.run()
                rmsd_results = R.results.rmsd.T
                rmsds_run.extend(rmsd_results[3])
            rmsds_list.append(rmsds_run)

            # Calculate the times - this only needs to be done once as all the times should be the same
            if i == 0:
                times = _np.linspace(start=0, stop=tot_simtime, num=len(rmsds_run))

    rmsds = _np.array(rmsds_list)

    return rmsds, times
