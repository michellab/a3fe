"""Functions for running free energy calculations with SOMD with automated
 equilibration detection based on an ensemble of simulations."""

from cProfile import label
import subprocess as _subprocess
import os as _os
import threading as _threading
from matplotlib import pyplot as _plt
import numpy as _np
import pickle as _pkl
import scipy.stats as _stats
from time import sleep as _sleep
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional


class Ensemble():
    """
    Class to hold and manipulate an ensemble of SOMD simulations.
    """

    def __init__(self, block_size: float = 1, ensemble_size: int = 5,
                 input_dir: str = "./input",
                 output_dir: str = "./output") -> None:
        """ 
        Initialise an ensemble of SOMD simulations. If ensemble.pkl exists in the
        output directory, the ensemble will be loaded from this file and any arguments
        supplied will be overwritten.

        Parameters
        ----------
        block_size : float, Optional, default: 1
            Size of blocks to use for equilibration detection, in ns.
        ensemble_size : int, Optional, default: 5
            Number of simulations to run in the ensemble.
        input_dir : str, Optional, default: "./input"
            Path to directory containing input files for the simulations.
        output_dir : str, Optional, default: "./output"
            Path to directory to store output files from the simulations.

        Returns
        -------
        None
        """
        # Check if we are starting from a previous simulation
        if _os.path.isfile(f"{output_dir}/ensemble.pkl"):
            print("Loading previous ensemble. Any arguments will be overwritten...")
            with open(f"{output_dir}/ensemble.pkl", "rb") as file:
                self.__dict__ = _pkl.load(file)
            # TODO: Check if the simulations are still running and continue if so. Ensure that the
            # total simulation times are correct by checking the sim files.

        else: # No pkl file to resume from
            print("Creating new ensemble...")

            self.block_size = block_size
            self.ensemble_size = ensemble_size
            self.input_dir = input_dir
            self.output_dir = output_dir
            self._running: bool = False
            self.run_thread: _Optional[_threading.Thread] = None
            self.lam_vals: _List[float] = self._get_lam_vals()
            self.lam_windows: _List[LamWindow] = []
            self.running_wins: _List[LamWindow] = []

            # Creating lambda window objects sets up required input directories
            for lam in self.lam_vals:
                self.lam_windows.append(LamWindow(lam, self.block_size,
                                        self.ensemble_size, self.input_dir,
                                        output_dir))
            self._dump()

    def run(self) -> None:
        """Run the ensemble of simulations with adaptive equilibration detection,
        and perform analysis once finished."""
        # Run in the background with threading so that user can continuously check
        # the status of the Ensemble object
        self.run_thread = _threading.Thread(target=self._run_without_threading, name="Ensemble")
        self.run_thread.start()
        self.running = True

    @property
    def running(self) -> bool:
        """Return True if the ensemble is currently running."""
        if self.run_thread is not None:
            self._running = self.run_thread.is_alive()
        else:
            self._running = False
        return self._running

    @running.setter
    def running(self, value: bool) -> None:
        """Set the running attribute."""
        self._running = value

    def _run_without_threading(self) -> None:
        """ Run the ensemble of simulations with adaptive equilibration detection,
        and perform analysis once finished. This function is called by run() with threading,
        so that the function can be run in the background and the user can continuously
        check the status of the Ensemble object."""

        # Run initial SOMD simulations
        for win in self.lam_windows:
            # Add buffer of 1 block_size to give chance for the equilibration to be detected.
            win.run(3 * self.block_size)
            win._update_log()
            self._dump()

        # Periodically check the simulations and analyse/ resubmit as necessary
        self.running_wins = self.lam_windows
        self._dump()
        with open(f"{self.output_dir}/status.log", "w") as file:
            file.write("Starting equilibration detection \n")
            while self.running_wins:
                _sleep(20 * 1)  # Check 20 seconds
                for win in self.running_wins:
                    # Check if the window has now finished - calling win.running updates the win._running attribute
                    if not win.running:
                        # Check if the simulation has equilibrated and if not, resubmit
                        if win.equilibrated:
                            file.write(f"Lambda: {win.lam:.3f} has equilibrated at {win.equil_time:.3f} ns \n")
                        else:
                            win.run(self.block_size)

                        # Write status after checking for running and equilibration, as this updates the
                        # _running and _equilibrated attributes
                        win._update_log()
                        self._dump()

                self.running_wins = [win for win in self.running_wins if win.running]

        # All simulations are now finished, so perform final analysis
        self.analyse()

        # Save data and perform final analysis

    def _get_lam_vals(self) -> _List[float]:
        """
        Return list of lambda values for the simulations,
        based on the configuration file.

        Returns
        -------
        lam_vals : List[float]
            List of lambda values for the simulations.
        """

        # Read number of lambda windows from input file
        lam_vals_str = []
        with open(self.input_dir + "/sim.cfg", "r") as ifile:
            lines = ifile.readlines()
            for line in lines:
                if line.startswith("lambda array ="):
                    lam_vals_str = line.split("=")[1].split(",")
                    break
        lam_vals = [float(lam) for lam in lam_vals_str]

        return lam_vals

    def analyse(self) -> None:
        """ Analyse the results of the ensemble of simulations. Requires that 
        all lambda windows have equilibrated.  """

        # Check that all simulations have equilibrated
        for win in self.lam_windows:
            # Avoid checking win.equilibrated as this causes expensive equilibration detection to be run
            if not win._equilibrated:
                raise RuntimeError(f"Not all lambda windows have equilibrated. Analysis cannot be performed.")
                
        # Remove unequilibrated data from the equilibrated output directory
        for win in self.lam_windows:
            equil_time = win.equil_time
            equil_index = int(equil_time / (win.sims[0].timestep * win.sims[0].nrg_freq))
            for sim in win.sims:
                in_simfile = sim.output_subdir + "/simfile.dat"
                out_simfile = sim.output_subdir + "/simfile_equilibrated.dat"
                self._write_equilibrated_data(in_simfile, out_simfile, equil_index)

        # Analyse data with MBAR and compute uncertainties
        output_dir = self.output_dir
        ensemble_size = self.ensemble_size
        # This is nasty - assumes naming always the same
        for run in range(1, ensemble_size + 1):
            with open(f"{self.output_dir}/freenrg-MBAR-run_{run}.dat", "w") as ofile:
                _subprocess.run(["/home/finlayclark/sire.app/bin/analyse_freenrg",
                                "mbar", "-i", f"{output_dir}/lambda*/run_0{run}/simfile.dat",
                                 "-p", "100", "--overlap", "--temperature",
                                 "298.0"], stdout=ofile)

        # TODO: Make convergence plots (which should be flat)

    def _write_equilibrated_data(self, in_simfile: str,
                                 out_simfile: str,
                                 equil_index: int) -> None:
        """
        Remove unequilibrated data from a simulation file and write a new
        file containing only the equilibrated data.

        Parameters
        ----------
        in_simfile : str
            Path to simulation file.
        out_simfile : str
            Path to new simulation file, containing only the equilibrated data
            from the original file.
        equil_index : int
            Index of the first equilibrated frame, given by 
            equil_time / (timestep * nrg_freq)

        Returns
        -------
        None
        """
        with open(in_simfile, "r") as ifile:
            lines = ifile.readlines()

        # Figure out how many lines come before the data
        non_data_lines = 0
        for line in lines:
            if line.startswith("#"):
                non_data_lines += 1
            else:
                break

        # Overwrite the original file with one containing only the equilibrated data
        with open(out_simfile, "w") as ofile:
            for line in lines[equil_index + non_data_lines:]:
                ofile.write(line)

    def _update_log(self) -> None:
        """
        Update the status log file with the current status of the ensemble.

        Returns
        -------
        None
        """
        with open(f"{self.output_dir}/status.log", "a") as ofile:
            ofile.write("##############################################\n")
            for var in vars(self):
                ofile.write(f"{var}: {getattr(self, var)} \n")

    def _dump(self) -> None:
        """ Dump the current state of the ensemble to a pickle file. Specifically,
         pickle self.__dict__ with self.run_thread = None, as _thread_lock objects
         can't be pickled.   """
        temp_dict = {key:val for key,val in self.__dict__.items() if key != "run_thread"}
        temp_dict["run_thread"] = None
        with open(f"{self.output_dir}/ensemble.pkl", "wb") as ofile:
            _pkl.dump(temp_dict, ofile)


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
        self._equilibrated: bool = False
        self.equil_time: _Optional[float] = None
        self._running: bool = False
        self.tot_simtime: float = 0  # ns
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

        self._running = True

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
        self._equilibrated, self.equil_time = self.check_equil()
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
        # Conversion between time and gradient indices.
        time_to_ind = 1 / (self.sims[0].timestep * self.sims[0].nrg_freq)
        idx_block_size = int(self.block_size * time_to_ind)

        # Read dh/dl data from all simulations and calculate the gradient of the
        # gradient, d_dh_dl
        d_dh_dls = []
        dh_dls = []
        times, _ = self.sims[0].read_gradients()
        equilibrated = False
        equil_time = None

        for sim in self.sims:
            _, dh_dl = sim.read_gradients()  # Times should be the same for all sims
            dh_dls.append(dh_dl)
            # Create array of nan so that d_dh_dl has the same length as times irrespective of
            # the block size
            d_dh_dl = _np.full(len(dh_dl), _np.nan)
            # Compute rolling average with the block size
            rolling_av_dh_dl = self._get_rolling_average(dh_dl, idx_block_size)
            for i in range(len(dh_dl)):
                if i < 2 * idx_block_size:
                    continue
                else:
                    d_dh_dl[i] = (rolling_av_dh_dl[i] - rolling_av_dh_dl[i - idx_block_size]) / \
                        self.block_size  # Gradient of dh/dl in kcal mol-1 ns-1
            d_dh_dls.append(d_dh_dl)

        # Calculate the mean gradient
        mean_d_dh_dl = _np.mean(d_dh_dls, axis=0)

        # Check if the mean gradient has been 0 at any point, making
        # sure to exclude the initial nans
        last_grad = mean_d_dh_dl[2*idx_block_size]
        for i, grad in enumerate(mean_d_dh_dl[2*idx_block_size:]):
            # Check if gradient has passed through 0
            if _np.sign(last_grad) != _np.sign(grad):
                equil_time = times[i]
                break
            last_grad = grad

        if equil_time:
            equilibrated = True

        # Save plots of dh/dl and d_dh/dl
        plot(x_vals = times,
             y_vals= _np.array([self._get_rolling_average(dh_dl, idx_block_size) for dh_dl in dh_dls]),
             x_label= "Simulation Time per Window per Run / ns",
             y_label= r"$\frac{\mathrm{d}h}{\mathrm{d}\lambda}$ / kcal mol$^{-1}$", 
             outfile= f"{self.output_dir}/lambda_{self.lam:.3f}/dhdl",
            # Shift the equilibration time by 2 * block size to account for the
            # delay in the block average calculation.
             vline_val= equil_time + 1 * self.block_size if equil_time else None)

        plot(x_vals = times,
             y_vals= _np.array(d_dh_dls),
             x_label= "Simulation Time per Window per Run / ns",
             y_label= r"$\frac{\partial}{\partial t}\frac{\partial H}{\partial \lambda}$ / kcal mol$^{-1}$ ns$^{-1}$", 
             outfile= f"{self.output_dir}/lambda_{self.lam:.3f}/ddhdl",
             vline_val= equil_time + 2 * self.block_size if equil_time else None,
             hline_val=0)

        return equilibrated, equil_time

    def _get_rolling_average(self, data: _np.ndarray, idx_block_size: int) -> _np.ndarray:
        """
        Calculate the rolling average of a 1D array.

        Parameters
        ----------
        data : np.ndarray
            1D array of data to be block averaged.
        idx_block_size : int
            Index size of the blocks to be used in the block average.

        Returns
        -------
        block_av : np.ndarray
            1D array of block averages of the same length as data.
            Initial values (before there is sufficient data to calculate 
            a block average) are set to nan.
        """
        rolling_av = _np.full(len(data), _np.nan)

        for i in range(len(data)):
            if i < idx_block_size:
                continue
            else:
                block_av = _np.mean(data[i - idx_block_size: i])
                rolling_av[i] = block_av

        return rolling_av

    def _update_log(self) -> None:
        """ Write the status of the lambda window to a log file. """

        with open(f"{self.output_dir}/lambda_{self.lam:.3f}/status.log", "a") as ofile:
            ofile.write("##############################################\n")
            for var in vars(self):
                ofile.write(f"{var}: {getattr(self, var)} \n")

        for sim in self.sims:
            sim._update_log()


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
        self.job_id: _Optional[int] = None
        self._running: bool = False
        self.output_subdir: str = output_dir + "/lambda_" + f"{lam:.3f}" + "/run_" + str(run_no).zfill(2)
        self.tot_simtime: float = 0  # ns
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
        # Get job ids of currently running jobs
        cmd = "squeue -h -u $USER | awk '{print $1}' | paste -s -d, -"
        process = _subprocess.Popen(cmd, shell=True, stdin=_subprocess.PIPE,
                                    stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
                                    close_fds=True)
        output = process.communicate()[0]
        job_ids = [int(job_id) for job_id in output.decode('utf-8').strip().split(",") if job_id != ""]

        if self.job_id in job_ids:
            self._running = True
            print(f"Simulation {self.lam} {self.run_no} is still running.")

        else:
            self._running = False
            print(f"Simulation {self.lam} {self.run_no} is finished.")

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

        timestep = None  # ns
        nmoves = None  # number of moves per cycle
        nrg_freq = None  # number of timesteps between energy calculations
        with open(self.input_dir + "/sim.cfg", "r") as ifile:
            lines = ifile.readlines()
            for line in lines:
                if line.startswith("timestep ="):
                    timestep = float(line.split("=")[1].split()[0])
                if line.startswith("nmoves ="):
                    nmoves = float(line.split("=")[1])
                if line.startswith("energy frequency ="):
                    nrg_freq = float(line.split("=")[1])

        if timestep is None or nmoves is None or nrg_freq is None:
            raise ValueError("Could not find timestep or nmoves in sim.cfg.")

        self.timestep = timestep / 1_000_000  # fs to ns
        self.nrg_freq = nrg_freq
        self.time_per_cycle = timestep * nmoves / 1_000_000  # fs to ns

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
        # Need to make sure that duration is a multiple of the time per cycle
        # otherwise actual time could be quite different from requested duration
        if round(duration % self.time_per_cycle, 4) != 0:
            raise ValueError("Duration must be a multiple of the time per cycle.")
        # Need to modify the config file to set the correction n_cycles
        n_cycles = int(duration / self.time_per_cycle)
        self._set_n_cycles(n_cycles)

        # Run SOMD
        cmd = f"~/Documents/research/scripts/abfe/rbatch.sh --chdir {self.output_subdir} {self.output_subdir}/input/run_somd.sh {self.lam}"
        process = _subprocess.Popen(cmd, shell=True, stdin=_subprocess.PIPE,
                                    stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
                                    close_fds=True)
        if process.stdout is None:
            raise ValueError("Could not get stdout from process.")
        process_output = process.stdout.read()
        job_id = int((process_output.split()[-1]))
        self.running = True
        self.tot_simtime += duration
        self.job_id = job_id

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

        times = [x * self.timestep for x in steps]  # Timestep already in ns

        times_arr = _np.array(times)
        grads_arr = _np.array(grads)

        return times_arr, grads_arr

    def _update_log(self) -> None:
        """ Write the status of the simulation to a log file. """

        with open(f"{self.output_subdir}/status.log", "a") as ofile:
            ofile.write("##############################################\n")
            for var in vars(self):
                ofile.write(f"{var}: {getattr(self, var)} \n")


def plot(x_vals: _np.ndarray, y_vals: _np.ndarray, x_label: str, y_label: str, 
         outfile: str, vline_val: _Optional[float] = None, 
         hline_val: _Optional[float] = None) -> None:
    """ 
    Plot several sets of y_vals against one set of x vals, and show confidence
    intervals based on inter-y-set deviations (assuming normality).

    Parameters
    ----------
    x_vals : np.ndarray
        1D array of x values.
    y_vals : np.ndarray
        1 or 2D array of y values, with shape (n_sets, n_vals). Assumes that
        the sets of data are passed in the same order as the runs.
    x_label : str
        Label for the x axis.
    y_label : str
        Label for the y axis.
    outfile : str
        Name of the output file.
    vline_val : float, Optional
        x value to draw a vertical line at, for example the time taken for
        equilibration.
    hline_val : float, Optional
        y value to draw a horizontal line at.
    """
    y_avg = _np.mean(y_vals, axis=0)
    conf_int = _stats.t.interval(0.95, len(y_vals[:,0])-1, loc=y_avg, scale=_stats.sem(y_vals,axis=0)) # 95 % C.I.

    fig, ax = _plt.subplots(figsize=(8,6))
    ax.plot(x_vals, y_avg, label="Mean", linewidth=2)
    for i, entry in enumerate(y_vals):
        ax.plot(x_vals, entry, alpha=0.5, label=f"run {i+1}")
    if vline_val is not None:
        ax.axvline(x=vline_val, color='red', linestyle='dashed')
    if hline_val is not None:
        ax.axhline(y=hline_val, color='black', linestyle='dashed')
    # Add confidence intervals
    ax.fill_between(x_vals, conf_int[0], conf_int[1], alpha=0.5, facecolor='#ffa500')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    