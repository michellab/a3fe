"""Functions for running free energy calculations with SOMD with automated
 equilibration detection based on an ensemble of simulations."""

import subprocess as _subprocess
import os as _os
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any


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


def run_calc(block_size: float = 2, ensemble_size: int = 5, input_dir: str = "./input", 
             output_dir: str = "./output") -> None:
    """
    Run ensemble of SOMD free energy calculation with automated 
    equilibration detection.

    Parameters
    ----------
    block_size : float, Optional, default: 2
        Size of blocks to use for equilibration detection, in ns.
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

    # Create all simulation objects. In doing so, their directories will be created.
    lam_vals = get_lam_vals(input_dir, output_dir)

    simulations = []
    for run in range(1, ensemble_size + 1):
        for lam in lam_vals:
            simulations.append(Simulation(lam, run, input_dir, output_dir))

    # Run initial SOMD simulations
    #for simulation in simulations:
        #simulation.run(block_size + 0.5)

    # Check the results of the simulations

    # Cycle of analysis/ resubmission as necessary

    # Save data and perform final analysis


class Simulation():
    """Class to store information about a single SOMD simulation."""
    
    def __init__(self, lam: float, run: int, input_dir: str = "./input",
                    output_dir: str = "./output",
                    equilibrated: str = False, running: bool = False,
                    jobid: str = None) -> None:
        """
        Initialise a Simulation object.

        Parameters
        ----------
        lam : float
            Lambda value for the simulation.
        run : int
            Index of repeat for the simulation.
        output_dir : str
            Path to the output directory.
        equilibrated : bool, Optional, default: False
            Whether or not the simulation has equilibrated.
        running : bool, Optional, default: False
            Whether or not the simulation is currently running.
        jobid : str, Optional, default: None
            SLURM job ID for the simulation.

        Returns
        -------
        None
        """

        self.lam = lam
        self.run = run
        self.input_dir = _os.path.abspath(input_dir)
        # Check that the input directory contains the required files
        self.validate_input()
        self.output_dir = _os.path.abspath(output_dir)
        self.equilibrated = equilibrated
        self.running = running
        self.jobid = jobid
        self.output_subdir = output_dir + "/lambda_" + f"{lam:.3f}" + "/run_" + str(run).zfill(2)

        # Create the output subdirectory
        _subprocess.call(["mkdir", "-p", self.output_subdir])
        # Create a soft link to the input dir to simplify running simulations
        _subprocess.call(["ln", "-s", self.input_dir, self.output_subdir + "/input"])

    
    def validate_input(self) -> None:
        """ Check that the required input files are present. """

        # Check that the input directory exists
        if not _os.path.isdir(self.input_dir):
            raise FileNotFoundError("Input directory does not exist.")

        # Check that the required input files are present
        required_files = ["run_somd.sh", "sim.cfg", "system.top", "system.crd", "morph.pert"]
        for file in required_files:
            if not _os.path.isfile(self.input_dir + "/" + file):
                raise FileNotFoundError("Required input file " + file + " not found.")


    def run(self, duration: float = 2.5) -> int:
        """
        Run a SOMD simulation.

        Parameters
        ----------
        duration : float, Optional, default: 2.5
            Duration of simulation, in ns.

        Returns
        -------
        jobid : int
            The SLURM job ID for the simulation.
        """

        # Run SOMD
        process = _subprocess.Popen(["sbatch", self.output_subdir + "/input/run_somd.sh", self.lam, duration],
                                    stin = _subprocess.PIPE, stdout = _subprocess.PIPE, stderr = _subprocess.STDOUT,
                                    close_fds=True)
        process_output = process.stdout.read()
        jobid = int((process_output.split()[-1]))

        return jobid


def get_lam_vals(input_dir: str = "./input", 
                 output_dir: str = "./output") -> _List[float]:
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
    lam_vals : List[float]
        List of lambda values for the simulations.
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

    return lam_vals


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
