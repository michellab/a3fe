"""
Unit and regression test for the EnsEquil package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import EnsEquil


def test_EnsEquil_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "EnsEquil" in sys.modules
