"""Simulation engine backends."""

from ..configuration import EngineType
from ._engine import EngineBackend
from .gromacs import GromacsBackend
from .somd import SomdBackend

engine_backend_registry = {
    EngineType.SOMD: SomdBackend(),
    EngineType.GROMACS: GromacsBackend(),
}
