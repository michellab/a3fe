"""Abstract base class for engine configurations."""

from __future__ import annotations

import yaml as _yaml
from abc import ABC as _ABC, abstractmethod
from pydantic import BaseModel as _BaseModel
from typing import Any as _Any, Dict as _Dict


class EngineRunnerConfig(_BaseModel, _ABC):
    """Base class for engine runner configurations."""

    def get_config(self) -> _Dict[str, _Any]:
        """
        Get the configuration dictionary.
        """
        pass

    @abstractmethod
    def get_file_name(self) -> str:
        """
        Get the name of the configuration file.
        """
        pass
    
    def dump(self, save_dir: str) -> None:
        """
        Dump the configuration to a YAML file using `self.model_dump()`.

        Parameters
        ----------
        save_dir : str
            Directory to dump the configuration to.
        """
        model_dict = self.model_dump()

        save_path = save_dir + "/" + self.get_file_name()
        with open(save_path, "w") as f:
            _yaml.dump(model_dict, f, default_flow_style=False)


    @classmethod
    def load(cls, load_dir: str) -> EngineRunnerConfig:
        """
        Load a configuration from a YAML file.

        Parameters
        ----------
        load_dir : str
            Directory to load the configuration from.

        Returns
        -------
        config : EngineRunnerConfig
            The loaded configuration.
        """ 
        with open(load_dir + "/" + cls.get_file_name(), "r") as f:
            model_dict = _yaml.safe_load(f)
        return cls(**model_dict)

    @abstractmethod
    def write_config(self, simfile_path: str) -> None:
        """
        Write the configuration to a file.
        """
        pass

    @abstractmethod
    def get_run_cmd(self) -> str:
        """
        Get the command to run the simulation.
        """
        pass
