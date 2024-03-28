""""Unit tests for the _restraint dataclass."""

from pytest import approx


def test_create_a3fe_restraint(a3fe_restraint):
    """Test that an A3feRestraint object works as expected."""
    assert (
        a3fe_restraint.toString()
        == 'boresch restraints dictionary = {"anchor_points":{"r1":803, "r2":801, "r3":818, "l1":28, "l2":6, "l3":21}, "equilibrium_values":{"r0":9.67, "thetaA0":1.46, "thetaB0":1.33,"phiA0":-1.81, "phiB0":2.25, "phiC0":2.44}, "force_constants":{"kr":8.11, "kthetaA":54.75, "kthetaB":115.30, "kphiA":159.43, "kphiB":167.55, "kphiC":83.78}}'
    )
    assert a3fe_restraint.getCorrection(method="analytical").value() == approx(
        -10.753345568838798, abs=1e-6
    )
    assert a3fe_restraint.getCorrection(method="numerical").value() == approx(
        -10.755515978356776, abs=1e-6
    )
