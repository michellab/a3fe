"""
Unit and regression test for the EnsEquil package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import run


def test_run_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "run" in sys.modules
