"""Basic test on import of a3fe"""

import sys

import a3fe


def test_a3fe_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "a3fe" in sys.modules