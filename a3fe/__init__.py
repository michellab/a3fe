"""A package for running free energy calculations with SOMD with automated equilibration detection based on an ensemble of simulations."""

# This package was previously named "EnsEquil". To allow objects pickled with the
# old name to be unpickled with the new name, we set EnsEquil to point to a3fe.
import sys as _sys

_sys.modules["EnsEquil"] = _sys.modules["a3fe"]


# Run imports
from ._version import __version__

# Pdb helper import
from .read import *
from .run import *
