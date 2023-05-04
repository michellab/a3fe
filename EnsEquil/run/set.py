"""Functionality for manipulating sets of Calculations"""

import logging as _logging
import numpy as _np
import os as _os
import scipy.stats as _stats
from typing import Dict as _Dict, List as _List, Tuple as _Tuple, Any as _Any, Optional as _Optional, Callable as _Callable

from ..analyse.analyse_set import compute_stats as _compute_stats
from .calculation import Calculation as _Calculation
from ..analyse.plot import plot_against_exp as _plt_against_exp
from ..read._read_exp_dgs import read_exp_dgs as _read_exp_dgs
from ._simulation_runner import SimulationRunner as _SimulationRunner


from ._utils import TmpWorkingDir as _TmpWorkingDir

class Set(_SimulationRunner):
    """ 
    Class to set up, run, and analyse sets of ABFE calculations
    (each represented by Calculation objects).
    """
    def __init__(self,
                 calc_paths: _Optional[_List] = None,
                 calc_args: _Optional[_Dict[str,_Dict]] = None,
                 base_dir: _Optional[str] = None,
                 input_dir: _Optional[str] = None,
                 output_dir: _Optional[str] = None,
                 stream_log_level: int = _logging.INFO,
                 update_paths: bool = True) -> None:
        """
        Instantiate a calculation based on files in the input dir. If calculation.pkl exists in the
        base directory, the calculation will be loaded from this file and any arguments
        supplied will be overwritten.

        Parameters
        ----------
        calc_paths: List, Optional, default: None
            List of paths to the Calculation base directories. If None, then all directories
            in the current directory will be assumed to be calculation base directories
        calc_args: Dict[str: _Dict], Optional, default: None
            Dictionary of arguments to pass to the Calculation objects in the form
            {"path_to_calc_base_dir": {keyword: arg, ...} ...}
        base_dir : str, Optional, default: None
            Path to the base directory which contains all the Calculations. If None,
            this is set to the current working directory.
        input_dir : str, Optional, default: None
            Path to directory containing input files for example experimental free 
            energy changes. If None, this is set to `current_working_directory/input`.
        output_dir : str, Optional, default: None
            Path to directory containing output files. If None, this
            is set to `current_working_directory/output`.
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            calculation object and its child objects.
        update_paths: bool, Optional, default: True
            If True, if the simulation runner is loaded by unpickling, then
            update_paths() is called.

        Returns
        -------
        None
        """
        super().__init__(base_dir=base_dir,
                         input_dir=input_dir,
                         output_dir=output_dir,
                         stream_log_level=stream_log_level,
                         update_paths=update_paths)
        
        if not self.loaded_from_pickle:
            # Load/ create the Calculations - temporarily shift to the Calculation base dir
            if not calc_paths:
                calc_paths = [direct for direct in _os.listdir() if _os.isdir(direct)]
            self.calc_paths = calc_paths
            # TODO: Find a better solution which doesn't open a stupid number of files
            #for calc_path in calc_paths:
                # Temporarily move to the calculation base directory
                #with _TmpWorkingDir(calc_path) as _:
                    #if calc_args:
                        #calc = _Calculation(**calc_args)
                    #else:
                        #calc = _Calculation()
                    #self.calcs.append(calc)

            # Save the state and update log
            self._update_log()
            self._dump()
    
    @property
    def calcs(self) -> _List[_Calculation]:
        return self._sub_sim_runners

    @calcs.setter
    def calcs(self, value) -> None:
        self._logger.info("Modifying/ creating Calculations")
        self._sub_sim_runners = value

    def analyse(self, exp_dgs_path: str, offset: bool) -> None:
        """
        Analyse all calculations in the set and plot the 
        free energy changes with respect to experiment.

        Parameters
        ----------
        exp_dgs_path : str
            The path to the file containing the experimental free energy
            changes. This must be a csv file with the columns:

            calc_base_dir, name, exp_dg, exp_err
        offset: bool
            If True, the calculated dGs will be offset to match the average
            experimental free energies.
        """
        # Read the experimental dGs into a pandas dataframe and add the extra 
        # columns needed for the calculated values
        all_dgs = _read_exp_dgs(exp_dgs_path)
        all_dgs["calc_dg"] = _np.nan
        all_dgs["calc_er"] = _np.nan

        # Get the calculated dGs
        for calc_path in self.calc_paths:
            # Calculate the 95 % C.I. based on the five replicate runs
            #dgs = calc.delta_g
            #dg = dgs.mean()
            #conf_int = _stats.t.interval(0.95,
                                        #len(dgs)-1,
                                        #dg,
                                        #scale=_stats.sem(dgs))[1] - dg  # 95 % C.I.
            # For now, just pull the results from the output files
            with open(_os.path.join(calc_path, "output", "overall_stats.dat"), "rt") as ifile:
                lines = ifile.readlines()
                dg = float(lines[1].split(" ")[3])
                conf_int = float(lines[1].split(" ")[-2])
                print(dg, conf_int)
            # Get the name of the ligand for the calculation and use this to add the results
            #name = all_dgs.index[all_dgs["calc_base_dir"] == calc.base_dir]
            abs_calc_path = _os.path.abspath(calc_path)
            name = all_dgs.index[all_dgs["calc_base_dir"] == abs_calc_path]
            all_dgs.loc[name, "calc_dg"] = dg
            all_dgs.loc[name, "calc_er"] = conf_int

        # Offset the calculated values with their corrections
        all_dgs["calc_dg"] += all_dgs["calc_cor"]

        # Save results including NaNs
        all_dgs.to_csv(_os.path.join(self.output_dir, "results_summary"), sep=",")

        # Exclude rows with NaN
        all_dgs.dropna(inplace=True)

        # Offset the results if required
        if offset:
            shift = all_dgs["exp_dg"].mean() - all_dgs["calc_dg"].mean()
            all_dgs["calc_dg"]+=shift

        # Calculate statistics and save results
        stats = _compute_stats(all_dgs)
        name = "overall_stats_offset.txt" if offset else "overall_stats.txt"
        with open(_os.path.join(self.output_dir, name), "wt") as ofile:
            for stat in stats:
                ofile.write(f"{stat}: {stats[stat][0]:.2f} ({stats[stat][1]:.2f}, {stats[stat][2]:.2f})\n")

        # Plot
        _plt_against_exp(all_results=all_dgs, 
                         output_dir=self.output_dir, 
                         offset=offset,
                         stats = stats)

