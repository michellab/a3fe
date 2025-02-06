"""
Automated Adaptive Absolute alchemical Free Energy calculator. A package for running
adaptive alchemical absolute binding free energy calculations with SOMD
(distributed within sire(https://sire.openbiosim.org/)) using adaptive
protocols based on an ensemble of simulations. This requires the SLURM scheduling
system. Please see the documentation (https://a3fe.readthedocs.io/en/latest/?badge=latest).
"""

# This package was previously named "EnsEquil". To allow objects pickled with the
# old name to be unpickled with the new name, we set EnsEquil to point to a3fe.
import resource as _resource
import sys as _sys
import warnings as _warnings

from ._version import __version__
from .run import (
    CalcSet,
    Calculation,
    JobStatus,
    LamWindow,
    Leg,
    LegType,
    PreparationStage,
    Simulation,
    Stage,
    StageType,
    EngineType,
    enums,
)

from .configuration import SomdSystemPreparationConfig, SlurmConfig, SomdConfig

_sys.modules["EnsEquil"] = _sys.modules["a3fe"]

# A3FE can open many files due to the use of multiprocessing and
# threading with MBAR. This can cause a "Too many open files" error.
# The following code increases the number of open files allowed to the
# hard limit.
_nofile_soft, _nofile_hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
if _nofile_soft < _nofile_hard:
    _warnings.warn(
        f"Soft limit for number of open files ({_nofile_soft}) is less than the hard limit ({_nofile_hard})."
        " Increasing the soft limit to the hard limit."
    )
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (_nofile_hard, _nofile_hard))
