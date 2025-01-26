"""Unit and regression tests for the SomdConfig class."""

from tempfile import TemporaryDirectory
import os
import pytest

from a3fe import SomdConfig

def test_config_yaml_save_and_load():
    """Test that the config can be saved to and loaded from YAML."""
    with TemporaryDirectory() as dirname:
        config = SomdConfig(runtime=1,leg_type="BOUND")
        config.dump(dirname)
        config2 = SomdConfig.load(dirname)
        assert config.leg_type == config2.leg_type

def test_get_somd_config():
    """Test that the SOMD configuration file is generated correctly."""
    with TemporaryDirectory() as dirname:
        config = SomdConfig(
            integrator="langevinmiddle",
            timestep=4.0,
            runtime=1,  # Integer runtime
            cutoff_type="PME",
            thermostat=False
            )
        config_path = config.get_somd_config(run_dir=dirname)
        assert config_path == os.path.join(dirname, "somd.cfg")

        with open(config_path, "r") as f:
            config_content = f.read()

        assert "integrator = langevinmiddle" in config_content
        assert "cutoff type = PME" in config_content
        assert "thermostat = False" in config_content

@pytest.mark.parametrize("integrator,thermostat,should_pass", [
    ("langevinmiddle", False, True),
    ("langevinmiddle", True, False),
    ("leapfrogverlet", True, True),
    ("leapfrogverlet", False, False),
])
def test_integrator_thermostat_validation(integrator, thermostat, should_pass):
    """Test integrator and thermostat combination validation."""
    if should_pass:
        config = SomdConfig(integrator=integrator, thermostat=thermostat, runtime=1)
        assert config.integrator == integrator
        assert config.thermostat == thermostat
    else:
        with pytest.raises(ValueError):
            SomdConfig(integrator=integrator, thermostat=thermostat, runtime=1)

@pytest.mark.parametrize("charge,cutoff,should_pass", [
    (0, "cutoffperiodic", True),
    (0, "PME", True),
    (1, "PME", True),
    (-1, "PME", True),
    (1, "cutoffperiodic", False),
    (-1, "cutoffperiodic", False),
])
def test_charge_cutoff_validation(charge, cutoff, should_pass):
    """
    Test ligand charge & cutoff type combination validation:
    if ligand_charge!=0 => must use PME.
    """
    if should_pass:
        config = SomdConfig(ligand_charge=charge, cutoff_type=cutoff, runtime=1)
        assert config.ligand_charge == charge
        assert config.cutoff_type == cutoff
    else:
        with pytest.raises(ValueError):
            SomdConfig(ligand_charge=charge, cutoff_type=cutoff, runtime=1)

def test_ligand_charge_validation():
    """Test that ligand charge validation works correctly."""

    #test ligand_charge=0, any cutoff_type
    valid_config_cutoff = SomdConfig(
        ligand_charge=0,
        cutoff_type="cutoffperiodic",
        runtime=1
    )
    assert valid_config_cutoff.ligand_charge == 0
    assert valid_config_cutoff.cutoff_type == "cutoffperiodic"


    valid_config_charge = SomdConfig(
        ligand_charge=1,
        cutoff_type="PME",
        runtime=1
    )
    assert valid_config_charge.ligand_charge == 1
    assert valid_config_charge.cutoff_type == "PME"

    with pytest.raises(ValueError):
        SomdConfig(
            ligand_charge=1,
            cutoff_type="cutoffperiodic",
            runtime=1
        )


def test_get_somd_config_with_extra_options():
    """
    Test SOMD config generation with some extra_options.
    """
    with TemporaryDirectory() as dirname:
        config = SomdConfig(
            integrator="langevinmiddle",
            runtime=1,
            cutoff_type="PME",
            thermostat=False,
            extra_options={"custom_option": "value"}
        )
        path = config.get_somd_config(run_dir=dirname)
        with open(path, "r") as f:
            content = f.read()
        assert "### Extra Options ###" in content
        assert "custom_option = value" in content

def test_compare_with_reference_config():
    """Test that we can generate a config file that matches a reference config."""
    reference_lines = [
        "timestep = 4.0 * femtosecond",
        "constraint = hbonds",
        "hydrogen mass repartitioning factor = 3.0",
        "integrator = langevinmiddle",
        "inverse friction = 1.0 * picosecond",
        "temperature = 25.0 * celsius",
        "thermostat = False",
        "barostat = True",
        "pressure = 1.0 * atm",
        "cutoff type = cutoffperiodic",
        "cutoff distance = 12.0 * angstrom",
        "reaction field dielectric = 78.3",
        "buffered coordinates frequency = 5000",
        "center solute = True",
        "minimise = True",
        "use boresch restraints = True",
        "turn on receptor-ligand restraints mode = True",
        "perturbed residue number = 1",
        "energy frequency = 200",
        "lambda array = 0.0, 0.125, 0.25, 0.375, 0.5, 1.0"
    ]
    with TemporaryDirectory() as dirname:
        config = SomdConfig(
            runtime=1,
            constraint="hbonds",
            hydrogen_mass_factor=3.0,
            integrator="langevinmiddle",
            inverse_friction=1.0,
            temperature=25.0,
            thermostat=False,
            barostat=True,
            pressure=1.0,
            cutoff_type="cutoffperiodic",
            cutoff_distance=12.0,
            reaction_field_dielectric=78.3,
            buffered_coords_freq=5000,
            center_solute=True,
            minimise=True,
            use_boresch_restraints=True,
            turn_on_receptor_ligand_restraints=True,
            perturbed_residue_number=1,
            energy_frequency=200,
        )
        cfg_path = config.get_somd_config(run_dir=dirname, lambda_array=[0.0, 0.125, 0.25, 0.375, 0.5, 1.0])
        with open(cfg_path, "r") as f:
            cfg_content = f.read()
        for line in reference_lines:
            assert line in cfg_content, f"Expected '{line}' in generated config."

def test_copy_from_existing_config():
    """Test that we can copy from an existing somd.cfg file."""
    reference_config = "/home/roy/software/deve/a3fe/a3fe/data/example_run_dir/bound/discharge/output/lambda_0.000/run_01/somd.cfg"
    if not os.path.isfile(reference_config):
        pytest.skip("Reference config not found, skipping test.")
    c = SomdConfig._from_config_file(reference_config)

    assert c.use_boresch_restraints is True
    assert c.turn_on_receptor_ligand_restraints is False
    assert c.topfile.endswith("somd.prm7")
    expected_lambda = [0.0, 0.068, 0.137, 0.199, 0.261, 0.317, 0.368, 0.419, 0.472,
                      0.524, 0.577, 0.627, 0.677, 0.727, 0.775, 0.824, 0.877, 0.938, 1.0]
    assert c.lambda_array == expected_lambda    
    # Boresch restraints dictionary
    expected_boresch_dict = (
        '{"anchor_points":{"r1":4900, "r2":4888, "r3":4902, "l1":3, "l2":5, "l3":11}, '
        '"equilibrium_values":{"r0":7.67, "thetaA0":2.55, "thetaB0":1.48,"phiA0":-0.74, '
        '"phiB0":-1.53, "phiC0":3.09}, "force_constants":{"kr":3.74, "kthetaA":28.06, '
        '"kthetaB":9.98, "kphiA":16.70, "kphiB":24.63, "kphiC":5.52}}'
    )
    assert c.boresch_restraints_dictionary == expected_boresch_dict
