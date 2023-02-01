"""Functions for running free energy calculations with SOMD with automated
 equilibration detection based on an ensemble of simulations."""

import subprocess as _subprocess
from typing import Dict as _Dict, List as _List, Tuple as _Tuple


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


def test():
    print("Running calculation...")


def run_calc(ensemble_size: int = 5, input_dir: str = "./input", 
             output_dir: str = "./output") -> None:
    """
    Run ensemble of SOMD free energy calculation with automated 
    equilibration detection.

    Parameters
    ----------
    ensemble_size : int, Optional, default: 5
        Number of replica simulations to run.
    input_dir : str, Optional, default: "./input"
        Path to directory containing input files.
    output_dir : str, Optional, default: "./input"
        Path to the output directory.

    Returns
    -------
    None
    """

    create_output_dirs(ensemble_size, input_dir, output_dir)

    # Run initial SOMD simulations

    # Check the results of the simulations

    # Cycle of analysis/ resubmission as necessary

    # Save data and perform final analysis


def create_output_dirs(ensemble_size: int = 5, input_dir: str = "./input", 
                        output_dir: str = "./output") -> None:
    """
    Create required output directories.

    Parameters
    ----------
    input_dir : str
        Path to the input directory.
    output_dir : str
        Path to the output directory.

    Returns
    -------
    None
    """

    # Read number of lambda windows from input file
    lam_vals = []
    with open(input_dir + "/sim.cfg", "r") as ifile:
        lines = ifile.readlines()
        for line in lines:
            if line.startswith("lambda array ="):
                lam_vals = line.split("=")[1].split(",")
                break
    lam_vals = [float(lam) for lam in lam_vals]

    # Create subdirectories for each lambda window for each replica simulation
    for run in range(1, ensemble_size +1):
        for lam in lam_vals:
            output_subdir = output_dir + "/lambda_" + f"{lam:.3f}" + "/run_" + str(run).zfill(2)
            _subprocess.call(["mkdir", "-p", output_subdir])


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
