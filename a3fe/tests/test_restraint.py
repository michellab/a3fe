""""Unit tests for the _restraint dataclass."""

from pytest import approx

from a3fe.run._restraint import A3feRestraint

from .fixtures import bss_restraint


def test_create_a3fe_restraint(bss_restraint):
    """Test that an A3feRestraint object can be created from a BioSimSpace restraint object."""
    a3fe_restraint = A3feRestraint(bss_restraint)
    assert isinstance(a3fe_restraint, A3feRestraint)
    assert (
        a3fe_restraint.somd_restr_string
        == 'boresch restraints dictionary = {"anchor_points":{"r1":1330, "r2":1318, "r3":1332, "l1":28, "l2":6, "l3":21}, "equilibrium_values":{"r0":7.05, "thetaA0":1.32, "thetaB0":1.43,"phiA0":-2.78, "phiB0":1.21, "phiC0":0.91}, "force_constants":{"kr":3.91, "kthetaA":31.06, "kthetaB":70.95, "kphiA":26.92, "kphiB":66.33, "kphiC":99.93}}'
    )
    assert a3fe_restraint.getCorrection(method="analytical").value() == approx(
        -9.854938427174691, abs=1e-6
    )
    assert a3fe_restraint.getCorrection(method="numerical").value() == approx(
        -9.858136706200451, abs=1e-6
    )
