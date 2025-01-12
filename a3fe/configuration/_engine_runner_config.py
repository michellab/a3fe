"""Abstract base class for engine configurations."""

from __future__ import annotations

import yaml as _yaml
from pydantic import BaseModel as _BaseModel
from typing import Any as _Any
from typing import Dict as _Dict

class EngineRunnerConfig(_BaseModel):
    """Base class for engine runner configurations."""

    def get_config(self) -> _Dict[str, _Any]:
        """
        Get the configuration dictionary.
        """
        pass

    def get_file_name(self) -> str:
        """
        Get the name of the configuration file.
        """
        pass
    
    def dump(self, file_path: str) -> None:
        """
        Dump the configuration to a YAML file using `self.model_dump()`.

        Parameters
        ----------
        file_path : str
            Path to dump the configuration to.
        """
        try:
            config = self.model_dump()
            with open(file_path, "w") as f:
                _yaml.safe_dump(config, f)
        except Exception as e:
            print(f"Error dumping configuration: {e}")
            return
        print(f"Configuration dumped to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> EngineRunnerConfig:
        """
        Load a configuration from a YAML file.

        Parameters
        ----------
        file_path : str
            Path to load the configuration from.

        Returns
        -------
        config : EngineRunnerConfig
            The loaded configuration.
        """
        try: 
            with open(file_path, "r") as f:
                config_dict = _yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return
        return cls(**config_dict)
