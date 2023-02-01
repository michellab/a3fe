"""Functions for running free energy calculations with SOMD with automated
 equilibration detection based on an ensemble of simulations."""

import subprocess as _subprocess
import os as _os
import numpy as _np
from time import sleep as _sleep
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional

equil_fn = 1

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


def run_calc(block_size: float = 1, ensemble_size: int = 5, input_dir: str = "./input", 
             output_dir: str = "./output") -> None:
    """
    Run ensemble of SOMD free energy calculation with automated 
    equilibration detection.

    Parameters
    ----------
    block_size : float, Optional, default: 1
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
    lam_vals = get_lam_vals(input_dir)

    lam_windows = []
    for lam in lam_vals:
        lam_windows.append(LamWindow(lam, block_size, ensemble_size, input_dir, output_dir))

    # Run initial SOMD simulations
    for win in lam_windows:
        # Add 0.5 ns buffer to give chance for the equilibration to be detected.
        win.run(2*block_size + 0.5)

    with open("data.txt", "w") as file:
        # Periodically check the simulations and analyse/ resubmit as necessary
        running_wins = lam_windows
        while running_wins:
            _sleep(60 * 5) # Check every 5 minutes
            for win in running_wins:
                # Check if the window has finished
                if not win.running:
                    # Check if the simulation has equilibrated
                    if win.equilibrated:
                        file.write(f"Lambda: {win.lam:.3f} has equilibrated at {win.equil_time:.3f} ns \n")
                    else:
                        win.run(block_size) 

                    # If not, resubmit the simulation

                    # Else, analyse the simulation
    
        #running_windows = [win for win in running_wins if win.running]

    # Check the results of the simulations

    # Cycle of analysis/ resubmission as necessary

    # Save data and perform final analysis


class Simulation():
    """Class to store information about a single SOMD simulation."""
    
    def __init__(self, lam: float, run_no: int, input_dir: str = "./input",
                    output_dir: str = "./output") -> None:
        """
        Initialise a Simulation object.

        Parameters
        ----------
        lam : float
            Lambda value for the simulation.
        run_no : int
            Index of repeat for the simulation.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        None
        """

        self.lam = lam
        self.run_no = run_no
        self.input_dir = _os.path.abspath(input_dir)
        # Check that the input directory contains the required files
        self._validate_input()
        self.output_dir = _os.path.abspath(output_dir)
        self.job_id = None
        self.running = False
        self.output_subdir = output_dir + "/lambda_" + f"{lam:.3f}" + "/run_" + str(run_no).zfill(2)
        self.tot_simtime = 0 # ns
        # Now read useful parameters from the simulation file options
        self._add_attributes_from_simfile()

        # Create the output subdirectory
        _subprocess.call(["mkdir", "-p", self.output_subdir])
        # Create a soft link to the input dir to simplify running simulations
        _subprocess.call(["cp", "-r", self.input_dir, self.output_subdir + "/input"])


    @property
    def running(self) -> bool:
        """
        Check if the simulation is still running,
        and update the running attribute accordingly.

        Returns
        -------
        self._running : bool
            True if the simulation is still running, False otherwise.
        """
        # Check if the job is still running
        cmd = f"squeue -j -h {self.jobid}"
        process = _subprocess.Popen(cmd, shell=True, stdin = _subprocess.PIPE,
                                    stdout = _subprocess.PIPE, stderr = _subprocess.STDOUT,
                                    close_fds=True)
        process_output = process.stdout.read()
        if len(process_output) > 0:
            self._running = True
        else:
            self._running = False

        return self._running

    @running.setter
    def running(self, value: bool) -> None:
        self._running = value
    
    
    def _validate_input(self) -> None:
        """ Check that the required input files are present. """

        # Check that the input directory exists
        if not _os.path.isdir(self.input_dir):
            raise FileNotFoundError("Input directory does not exist.")

        # Check that the required input files are present
        required_files = ["run_somd.sh", "sim.cfg", "system.top", "system.crd", "morph.pert"]
        for file in required_files:
            if not _os.path.isfile(self.input_dir + "/" + file):
                raise FileNotFoundError("Required input file " + file + " not found.")


    def _add_attributes_from_simfile(self) -> None:
        """
        Read the SOMD simulation option file and
        add useful attributes to the Simulation object.
        
        Returns
        -------
        time_per_cycle : int
            Time per cycle, in ns.
        """
        
        timestep = None # ns
        nmoves = None # number of moves per cycle
        nrg_freq = None # number of timesteps between energy calculations
        with open(self.input_dir + "/sim.cfg", "r") as ifile:
            lines = ifile.readlines()
            for line in lines:
                if line.startswith("timestep ="):
                    timestep = float(line.split("=")[1].split()[0])
                if line.startswith("nmoves ="):
                    nmoves = float(line.split("=")[1])
                if line.startswith("energy frequency ="):
                    nrg_freq = float(line.split("=")[1])

        if None in [timestep, nmoves, nrg_freq]:
            raise ValueError("Could not find timestep or nmoves in sim.cfg.")

        self.timestep = timestep / 1_000_000 # fs to ns
        self.nrg_freq = nrg_freq
        self.time_per_cycle = timestep * nmoves / 1_000_000 # fs to ns


    def run(self, duration: float = 2.5) -> None:
        """
        Run a SOMD simulation.

        Parameters
        ----------
        duration : float, Optional, default: 2.5
            Duration of simulation, in ns.

        Returns
        -------
        None
        """
        # Need to modify the config file to set the correction n_cycles
        n_cycles = int(duration / self.time_per_cycle)
        self._set_n_cycles(n_cycles)

        # Run SOMD
        cmd = f"~/Documents/research/scripts/abfe/rbatch.sh --chdir {self.output_subdir} {self.output_subdir}/input/run_somd.sh {self.lam}"
        process = _subprocess.Popen(cmd, shell=True, stdin = _subprocess.PIPE,
                                    stdout = _subprocess.PIPE, stderr = _subprocess.STDOUT,
                                    close_fds=True)
        
        process_output = process.stdout.read()
        jobid = int((process_output.split()[-1]))
        self.running = True
        self.tot_simtime += duration
        self.jobid = jobid


    def _set_n_cycles(self, n_cycles: int) -> None:
        """
        Set the number of cycles in the SOMD config file.

        Parameters
        ----------
        n_cycles : int
            Number of cycles to set in the config file.

        Returns
        -------
        None
        """
        # Find the line with n_cycles and replace
        with open(self.output_subdir + "/input/sim.cfg", "r") as ifile:
            lines = ifile.readlines()
            for i, line in enumerate(lines):
                if line.startswith("ncycles ="):
                    lines[i] = "ncycles = " + str(n_cycles) + "\n"
                    break

        # Now write the new file
        with open(self.output_subdir + "/input/sim.cfg", "w+") as ofile:
            for line in lines:
                ofile.write(line)

    
    def read_gradients(self) -> _Tuple[_np.ndarray, _np.ndarray]:
        """
        Read the gradients from the output file.

        Returns
        -------
        times : np.ndarray
            Array of times, in ns.
        grads : np.ndarray
            Array of gradients, in kcal/mol.
        """
        # Read the output file
        with open(self.output_subdir + "/simfile.dat", "r") as ifile:
            lines = ifile.readlines()

        steps = []
        grads = []

        for line in lines:
            vals = line.split()
            if not line.startswith("#"):
                step = int(vals[0].strip())
                grad = float(vals[2].strip())
                steps.append(step)
                grads.append(grad)

        times = [x * self.timestep / 1_000_000 for x in steps] # Convert steps to time in ns

        times = _np.array(times)
        grads = _np.array(grads)

        return times, grads


class LamWindow():
    """A class to hold and manipulate a set of SOMD simulations at a given lambda value."""

    def __init__(self, lam: float, block_size: float = 1,
                ensemble_size: int = 5, input_dir: str = "./input",
                output_dir: str = "./output") -> None:
        """
        Initialise a LamWindow object.

        Parameters
        ----------
        lam : float
            Lambda value for the simulation.
        block_size : float, Optional, default: 1
            Size of the blocks to use for equilibration detection,
            in ns.
        ensemble_size : int, Optional, default: 5
            Number of simulations to run at this lambda value.
        input_dir : str, Optional, default: "./input"
            Path to the input directory.
        output_dir : str
            Path to the output directory.

        Returns
        -------
        None
        """
        self.lam = lam
        self.block_size = block_size
        self.ensemble_size = ensemble_size
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.equilibrated = False
        self.equil_time = None
        self.running = False
        self.tot_simtime = 0 # ns
        # Create the required simulations for this lambda value
        self.sims = []
        for i in range(1, ensemble_size + 1):
            sim = Simulation(lam, i, input_dir, output_dir)
            self.sims.append(sim)


    def run(self, duration: float = 2.5) -> None:
        """
        Run all simulations at the lambda value.

        Parameters
        ----------
        duration : float, Optional, default: 2.5
            Duration of simulation, in ns.

        Returns
        -------
        None
        """
        # Run the simulations
        for sim in self.sims:
            sim.run(duration)
            self.tot_simtime += duration


    @property
    def running(self) -> bool:
        """
        Check if all the simulations at the lambda window are still running
        and update the running attribute accordingly.

        Returns
        -------
        self._running : bool
            True if the simulation is still running, False otherwise.
        """
        all_finished = True
        for sim in self.sims:
            if sim.running:
                all_finished = False
                break
        self._running = not all_finished

        return self._running


    @running.setter
    def running(self, value: bool) -> None:
        self._running = value


    @property
    def equilibrated(self) -> bool:
        """
        Check if the ensemble of simulations at the lambda window is
        equilibrated, and update the equilibration time and status if
        so.

        Returns
        -------
        self._equilibrated : bool
            True if the simulation is equilibrated, False otherwise.
        """
        self.equilibrated, self.equil_time = self.check_equil()
        return self._equilibrated

        
    @equilibrated.setter
    def equilibrated(self, value: bool) -> None:
        self._equilibrated = value

    
    def check_equil(self) -> _Tuple[bool, _Optional[float]]:
        """
        Check if the ensemble of simulations at the lambda window is
        equilibrated.

        Returns
        -------
        equilibrated : bool
            True if the simulation is equilibrated, False otherwise.
        equil_time : float
            Time taken to equilibrate, in ns.
        """
        # Conversion between time and gradient indices. Minus one because no energies are recorded until
        # after the first nrg_freq steps
        time_to_ind = 1 / (self.sims[0].timestep * self.sims[0].nrg_freq) - 1
        idx_block_size = int(self.block_size * time_to_ind)

        # Read dh/dl data from all simulations and calculate the gradient of the
        # gradient, d_dh_dl
        d_dh_dls = []
        times, _ = self.sims[0].read_gradients()[0]
        equilibrated = False
        equil_time = None

        for sim in self.sims:
            _, dh_dl = sim.read_gradients() # Times should be the same for all sims
            # Create array of nan so that d_dh_dl has the same length as times irrespective of
            # the block size
            d_dh_dl = _np.full(len(dh_dl), _np.nan)
            for i in range(len(dh_dl)):
                if i < 2 * idx_block_size:
                    continue
                else:
                    later_block_av = _np.mean(dh_dl[i - idx_block_size: i])
                    earlier_block_av = _np.mean(dh_dl[i - 2 * idx_block_size: i - idx_block_size])
                    d_dh_dl[i] = (later_block_av - earlier_block_av) / self.block_size # Gradient of dh/dl in kcal mol-1 ns-1
            d_dh_dls.append(d_dh_dl)

        # Calculate the mean gradient
        d_dh_dls = _np.array(d_dh_dls)
        mean_d_dh_dl = _np.mean(d_dh_dls, axis=0)

        # Check if the mean gradient has been 0 at any point
        last_grad = mean_d_dh_dl[0]
        for i, grad in enumerate(mean_d_dh_dl):
            # Check if gradient has passed through 0
            if _np.sign(last_grad) != _np.sign(grad):
                equil_time = times[i]
                break
            last_grad = grad

        if equil_time:
            equilibrated = True 

        return equilibrated, equil_time


def get_lam_vals(input_dir: str = "./input") -> _List[float]:
    """
    Create required output directories.

    Parameters
    ----------
    input_dir : str
        Path to the input directory.

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
