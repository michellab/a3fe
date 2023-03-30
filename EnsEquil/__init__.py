"""A package for running free energy calculations with SOMD with automated equilibration detection based on an ensemble of simulations."""

# Add imports here
from .run import calculation, leg, stage, lambda_window, simulation
from .run.calculation import *
from .run.leg import *
from .run.stage import *
from .run.lambda_window import *
from .run.simulation import *

from ._version import __version__
