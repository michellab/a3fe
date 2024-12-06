"""Functionality for manipulating sets of Calculations"""

__all__ = ["CalcSet"]

import logging as _logging
import os as _os
from typing import Dict as _Dict
from typing import Iterable as _Iterable
from typing import List as _List
from typing import Optional as _Optional

import numpy as _np
from scipy import stats as _stats

from ..analyse.analyse_set import compute_stats as _compute_stats
from ..analyse.plot import plot_against_exp as _plt_against_exp
from ..read._read_exp_dgs import read_exp_dgs as _read_exp_dgs
from ._simulation_runner import SimulationRunner as _SimulationRunner
from ._utils import SimulationRunnerIterator as _SimulationRunnerIterator
from .calculation import Calculation as _Calculation
from ..configuration.system_preparation import (
    SystemPreparationConfig as _SystemPreparationConfig,
)


class CalcSet(_SimulationRunner):
    """
    Class to set up, run, and analyse sets of ABFE calculations
    (each represented by Calculation objects). This runs calculations
    sequentially to avoid overloading the system.
    """

    def __init__(
        self,
        calc_paths: _Optional[_List] = None,
        calc_args: _Dict[str, _Dict] = {},
        base_dir: _Optional[str] = None,
        input_dir: _Optional[str] = None,
        output_dir: _Optional[str] = None,
        stream_log_level: int = _logging.INFO,
        update_paths: bool = True,
    ) -> None:
        """
        Instantiate a calculation based on files in the input dir. If calculation.pkl exists in the
        base directory, the calculation will be loaded from this file and any arguments
        supplied will be overwritten.

        Parameters
        ----------
        calc_paths: List, Optional, default: None
            List of paths to the Calculation base directories. If None, then all directories
            in the current directory will be assumed to be calculation base directories
        calc_args: Dict[str: _Dict], Optional, default: {}
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
            set object and its child objects.
        update_paths: bool, Optional, default: True
            If True, if the simulation runner is loaded by unpickling, then
            update_paths() is called.

        Returns
        -------
        None
        """
        super().__init__(
            base_dir=base_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            stream_log_level=stream_log_level,
            update_paths=update_paths,
        )

        if not self.loaded_from_pickle:
            # Load/ create the Calculations - temporarily shift to the Calculation base dir
            if not calc_paths:  # If not supplied, assume that every sub-directory is a calculation base dir
                calc_paths = [
                    directory
                    for directory in _os.listdir()
                    if _os.path.isdir(directory)
                ]
            self.calc_paths = [_os.path.abspath(directory) for directory in calc_paths]
            self._calc_args = calc_args

            # Check that we can load all of the calculations
            for calc in self.calcs:
                # Having set them up according to _calc_args, save them
                calc._dump()

            # Save the state and update log
            self._update_log()
            self._dump()

    @property
    def _sub_sim_runners(self) -> _SimulationRunnerIterator[_Calculation]:
        return _SimulationRunnerIterator(
            getattr(self, "calc_paths", []),
            _Calculation,
            **getattr(self, "_calc_args", {}),
        )

    @_sub_sim_runners.setter
    def _sub_sim_runners(self, value: _Iterable) -> None:
        """
        Do nothing. This is required for compatibility with the parent class
         , which sets _sub_sim_runners to an empty list in __init__
        """

    @property
    def calcs(self) -> _SimulationRunnerIterator[_Calculation]:  # type: ignore
        return self._sub_sim_runners

    @calcs.setter
    def calcs(self, value: _Calculation) -> None:
        """Take the Calculation and add its base directory to the calc_paths list"""
        self._logger.info(
            f"Adding base directory {value.base_dir} to list of stored calculation base dirs"
        )
        self.calc_paths.append(value.base_dir)

    def setup(
        self,
        bound_leg_sysprep_config: _Optional[_SystemPreparationConfig] = None,
        free_leg_sysprep_config: _Optional[_SystemPreparationConfig] = None,
    ) -> None:
        """
        Set up all calculations sequentially.

        Parameters
        ----------
        bound_leg_sysprep_config: SystemPreparationConfig, opttional, default = None
            The system preparation configuration to use for the bound leg. If None, the default
            configuration is used.
        free_leg_sysprep_config: SystemPreparationConfig, opttional, default = None
            The system preparation configuration to use for the free leg. If None, the default
            configuration is used.
        """
        for calc in self.calcs:
            if calc.setup_complete:
                self._logger.info(
                    f"Calculation in {calc.base_dir} is already set up. Skipping..."
                )
                continue
            try:
                self._logger.info(f"Setting up calculation in {calc.base_dir}")
                calc.setup(
                    bound_leg_sysprep_config=bound_leg_sysprep_config,
                    free_leg_sysprep_config=free_leg_sysprep_config,
                )
                self._logger.info(f"Calculation in {calc.base_dir} successfully set up")

            except Exception as e:
                self._logger.error(
                    f"Error setting up calculation in {calc.base_dir}: {e}"
                )
                raise e

    def run(
        self,
        run_nos: _Optional[_List[int]] = None,
        adaptive: bool = True,
        runtime: _Optional[float] = None,
        runtime_constant: _Optional[float] = None,
        run_stages_parallel: bool = False,
    ) -> None:
        r"""
        Run all calculations. Analysis is not performed by default. If running adaptively,
        cycles of short runs then optimal runtime estimation are performed, where the optimal
        runtime is estimated according to

        .. math::

            t_{\\mathrm{Optimal, k}} = \\sqrt{\\frac{t_{\\mathrm{Current}, k}}{C}}\\sigma_{\\mathrm{Current}}(\\Delta \\widehat{F}_k)

        where:
        - :math:`t_{\\mathrm{Optimal, k}}` is the calculated optimal runtime for lambda window :math:`k`
        - :math:`t_{\\mathrm{Current}, k}` is the current runtime for lambda window :math:`k`
        - :math:`C` is the runtime constant
        - :math:`\sigma_{\\mathrm{Current}}(\\Delta \\widehat{F}_k)` is the current uncertainty in the free energy change contribution for lambda window :math:`k`. This is estimated from inter-run deviations.
        - :math:`\Delta \\widehat{F}_k` is the free energy change contribution for lambda window :math:`k`

        Parameters
        ----------
        run_nos : List[int], Optional, default: None
            List of run numbers to run. If None, all runs will be run.
        adaptive : bool, Optional, default: True
            If True, the stages will run until the simulations are equilibrated and perform analysis afterwards.
            If False, the stages will run for the specified runtime and analysis will not be performed.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this number of nanoseconds.
        runtime_constant: float, Optional, default: None
            The runtime_constant (kcal**2 mol**-2 ns*-1) only affects behaviour if running adaptively. This is used
            to calculate how long to run each simulation for based on the current uncertainty of the per-window
            free energy estimate.
        run_stages_parallel: bool, Optional, default: False
            If True, the stages for each individual calculation will be run in parallel. Can casuse issues with
            QOS limits on HPC clusters as each stage might try to submit jobs at the same time, resulting in
            oversubmission of jobs. Each calculation will still be run sequentially.

        Returns
        -------
        None
        """
        # Run each calculation sequentially
        for calc in self.calcs:
            if calc.is_complete:
                self._logger.info(
                    f"Calculation in {calc.base_dir} is already complete. Skipping..."
                )
                continue

            try:
                self._logger.info(f"Running calculation in {calc.base_dir}")
                calc.run(
                    run_nos=run_nos,
                    adaptive=adaptive,
                    runtime=runtime,
                    runtime_constant=runtime_constant,
                    parallel=run_stages_parallel,
                )
                calc.wait()
                self._logger.info(
                    f"Calculation in {calc.base_dir} completed successfully"
                )

            except Exception as e:
                self._logger.error(f"Error running calculation in {calc.base_dir}: {e}")
                raise e

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
        # base_dir = "/home/roy/a3fe/a3fe/data/example_calc_set"
        all_dgs = _read_exp_dgs(exp_dgs_path, self.base_dir)
        all_dgs["calc_dg"] = _np.nan
        all_dgs["calc_er"] = _np.nan

        # Get the calculated dGs
        for calc in self.calcs:
            # Get the name of the ligand for the calculation and use this to add the results
            # get the tail of the base dir name
            name = all_dgs.index[all_dgs["calc_base_dir"] == calc.base_dir]

            # Make sure that there is only one row with this name
            if not len(name) == 1:
                raise ValueError(
                    f"Found {len(name)} rows matching {calc.base_dir} in experimental dGs file"
                )

            # Carry out MBAR analysis if it has not been done already
            if calc._delta_g is None:
                calc.analyse()

            # Get the confidence interval
            mean_free_energy = _np.mean(calc._delta_g)
            conf_int = (
                _stats.t.interval(
                    0.95,
                    len(calc._delta_g) - 1,  # type: ignore
                    mean_free_energy,
                    scale=_stats.sem(calc._delta_g),
                )[1]
                - mean_free_energy
            )  # 95 % C.I.

            all_dgs.loc[name, "calc_dg"] = mean_free_energy
            all_dgs.loc[name, "calc_er"] = conf_int

        # Offset the calculated values with their corrections
        all_dgs["calc_dg"] += all_dgs["calc_cor"]

        # Save results including NaNs
        all_dgs.to_csv(_os.path.join(self.output_dir, "results_summary.txt"), sep=",")

        # Exclude rows with NaN
        all_dgs.dropna(inplace=True)

        # Offset the results if required
        if offset:
            shift = all_dgs["exp_dg"].mean() - all_dgs["calc_dg"].mean()
            all_dgs["calc_dg"] += shift

        # Calculate statistics and save results
        stats = _compute_stats(all_dgs)
        name = "overall_stats_offset.txt" if offset else "overall_stats.txt"
        with open(_os.path.join(self.output_dir, name), "wt") as ofile:
            for stat in stats:
                ofile.write(
                    f"{stat}: {stats[stat][0]:.2f} ({stats[stat][1]:.2f}, {stats[stat][2]:.2f})\n"
                )

        # Plot
        _plt_against_exp(
            all_results=all_dgs, output_dir=self.output_dir, offset=offset, stats=stats
        )
