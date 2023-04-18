"""Functionality for processing slurm files."""

import os as _os

def get_slurm_file_base(slurm_file: str) -> str:
    """
    Find out what the slurm output file will be called.
    
    Parameters
    ----------
    slurm_file : str
        The absolute path to the slurm job file.

    Returns
    -------
    slurm_file_base : str
        The file base for for any output written by the slurm job.
    """
    # Get the path to the base dir
    base_dir = _os.path.dirname(slurm_file)
    # Find the slurm output file
    with open(slurm_file, "r") as f:
        for line in f:
            split_line = line.split()
            if len(split_line) > 0 and split_line[0] == "#SBATCH":
                if split_line[1] == "--output" or split_line[1] == "-o":
                    slurm_pattern = split_line[2]
                    if "%" in slurm_pattern:
                        file_base = slurm_pattern.split("%")[0]
                        return  _os.path.join(base_dir, file_base)
                    else:
                        return _os.path.join(base_dir, slurm_pattern)

    # We haven't returned - raise an error
    raise RuntimeError(f"Could not find slurm output file name in {slurm_file}")

