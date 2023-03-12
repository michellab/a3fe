"""Abstract base class for simulation runners."""

from abc import ABC, abstractmethod
import pathlib as _pathlib
import pickle as _pkl
from typing import Optional as _Optional
import logging as _logging

class SimulationRunner(ABC):
    """An abstract base class for simulation runners."""
    
    def __init__(self,
                 base_dir: _Optional[str] = None,
                 input_dir: _Optional[str] = None,
                 output_dir: _Optional[str] = None,
                 stream_log_level: int = _logging.INFO) -> None:
        """
        base_dir : str, Optional, default: None
            Path to the base directory for the simulation runner.
            If None, this is set to the current working directory.
        input_dir : str, Optional, default: None
            Path to directory containing input files for the 
            simulation runner. If None, this is set to
            `base_directory/input`.
        output_dir : str, Optional, default: None
            Path to the output directory in which to store the
            output from the simulation. If None, this is set 
            to `base_directory/output`.
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers for the
            calculation object and its child objects.
        """
        # Set up the directories
        if base_dir is None:
            base_dir = str(_pathlib.Path.cwd())
        if not _pathlib.Path(base_dir).is_dir():
            _pathlib.Path(base_dir).mkdir(parents=True)
        if input_dir is None:
            input_dir = str(_pathlib.Path(base_dir, "input"))
        if output_dir is None:
            output_dir = str(_pathlib.Path(base_dir, "output"))

        # Check if we are starting from a previous simulation runner
        self.loaded_from_pickle = False
        if _pathlib.Path(f"{base_dir}/{self.__class__.__name__}.pkl").is_file():
            print(f"Loading previous {self.__class__.__name__}. Any arguments will be overwritten...")
            with open(f"{base_dir}/{self.__class__.__name__}.pkl", "rb") as file:
                self.__dict__ = _pkl.load(file)
            self.loaded_from_pickle = True

        else:
            # Set up the base dir, input dir, output dir, and logging
            self.base_dir = base_dir
            # Only create the input and output directories if they're called, using properties
            self._input_dir = input_dir
            self._output_dir = output_dir

            # Set up the logger
            # TODO: Debug - why is debug output no longer working
            self.stream_log_level = stream_log_level
            self._logger = _logging.getLogger(str(self))
            self._logger.setLevel(_logging.DEBUG)
            self._logger.propagate = False
            # For the file handler, we want to log everything
            file_handler = _logging.FileHandler(f"{base_dir}/{self.__class__.__name__}.log")
            file_handler.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            file_handler.setLevel(_logging.DEBUG)
            # For the stream handler, we want to log at the user-specified level
            stream_handler = _logging.StreamHandler()
            stream_handler.setFormatter(_logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
            stream_handler.setLevel(stream_log_level)
            # Add the handlers to the logger
            self._logger.addHandler(file_handler)
            self._logger.addHandler(stream_handler)
            #Set up the logger

            # Save state
            self._dump()

    @property
    def input_dir(self) -> str:
        """The input directory for the simulation runner."""
        if not _pathlib.Path(self._input_dir).is_dir():
            _pathlib.Path(self._input_dir).mkdir(parents=True)
        return self._input_dir

    @input_dir.setter
    def input_dir(self, value: str) -> None:
        """Set the input directory for the simulation runner."""
        self._input_dir = value

    @property
    def output_dir(self) -> str:
        """The output directory for the simulation runner."""
        if not _pathlib.Path(self._output_dir).is_dir():
            _pathlib.Path(self._output_dir).mkdir(parents=True)
        return self._output_dir
    
    @output_dir.setter
    def output_dir(self, value: str) -> None:
        """Set the output directory for the simulation runner."""
        self._output_dir = value

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def run(self) -> None:
        """Run the simulation runner"""

    def update_paths(self, old_sub_path: str, new_sub_path: str) -> None:
        """ 
        Replace the old sub-path with the new sub-path in the base, input, and output directory
        paths.

        Parameters
        ----------
        old_sub_path : str
            The old sub-path to replace.
        new_sub_path : str
            The new sub-path to replace the old sub-path with.
        """
        # Use private attributes to avoid triggering the property setters
        # which might cause issues by creating directories
        for attr in ["base_dir", "_input_dir", "_output_dir"]:
            setattr(self, attr, getattr(self, attr).replace(old_sub_path, new_sub_path))

    def _update_log(self) -> None:
        """ Update the status log file with the current status of the simulation runner."""
        self._logger.info("##############################################")
        for var in vars(self):
            self._logger.info(f"{var}: {getattr(self, var)}")
        self._logger.info("##############################################")

    def _dump(self) -> None:
        """ Dump the current state of the simulation object to a pickle file."""
        with open(f"{self.base_dir}/{self.__class__.__name__}.pkl", "wb") as ofile:
            _pkl.dump(self.__dict__, ofile)
