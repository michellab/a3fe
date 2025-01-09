"""Abstract base class for engine configurations."""

from __future__ import annotations

import copy as _copy
import logging as _logging
import yaml as _yaml
from abc import ABC, abstractmethod
from typing import Any as _Any
from typing import Dict as _Dict

from ..run._logging_formatters import _A3feStreamFormatter


class EngineRunnerConfig(ABC):
    """An abstract base class for engine configurations."""

    def __init__(
        self,
        stream_log_level: int = _logging.INFO,
    ) -> None:
        """
        Initialize the engine configuration.

        Parameters
        ----------
        stream_log_level : int, Optional, default: logging.INFO
            Logging level to use for the steam file handlers.
        """
        # Set up logging
        self._stream_log_level = stream_log_level
        self._set_up_logging()

    def _set_up_logging(self) -> None:
        """Set up logging for the configuration."""
        # If logger exists, remove it and start again
        if hasattr(self, "_logger"):
            handlers = self._logger.handlers[:]
            for handler in handlers:
                self._logger.removeHandler(handler)
                handler.close()
            del self._logger

        # Create a new logger
        self._logger = _logging.getLogger(f"{self.__class__.__name__}")
        self._logger.propagate = False
        self._logger.setLevel(_logging.DEBUG)

        # For the stream handler, we want to log at the user-specified level
        stream_handler = _logging.StreamHandler()
        stream_handler.setFormatter(_A3feStreamFormatter())
        stream_handler.setLevel(self._stream_log_level)
        self._logger.addHandler(stream_handler)

    @abstractmethod
    def get_config(self) -> _Dict[str, _Any]:
        """
        Get the configuration dictionary.

        Returns
        -------
        config : Dict[str, Any]
            The configuration dictionary.
        """
        pass

    def dump(self, file_path: str) -> None:
        """
        Dump the configuration to a YAML file.

        Parameters
        ----------
        file_path : str
            Path to dump the configuration to.
        """
        config = self.get_config()
        with open(file_path, "w") as f:
            _yaml.safe_dump(config, f, default_flow_style=False)
        self._logger.info(f"Configuration dumped to {file_path}")

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
        with open(file_path, "r") as f:
            config_dict = _yaml.safe_load(f)
        return cls(**config_dict)

    @abstractmethod
    def get_file_name(self) -> str:
        """
        Get the name of the configuration file.

        Returns
        -------
        file_name : str
            The name of the configuration file.
        """
        pass

    def __eq__(self, other: object) -> bool:
        """
        Check if two configurations are equal.

        Parameters
        ----------
        other : object
            The other configuration to compare with.

        Returns
        -------
        equal : bool
            Whether the configurations are equal.
        """
        if not isinstance(other, EngineRunnerConfig):
            return NotImplemented
        return self.get_config() == other.get_config()

    def copy(self) -> EngineRunnerConfig:
        """
        Create a deep copy of the configuration.

        Returns
        -------
        config : EngineRunnerConfig
            A deep copy of the configuration.
        """
        return _copy.deepcopy(self)
