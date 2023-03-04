"""A package for running free energy calculations with SOMD with automated equilibration detection based on an ensemble of simulations."""

# Add imports here
from .calculation import *
from .leg import *
from .stage import *
from .lambda_window import *
from .simulation import *

from ._version import __version__
