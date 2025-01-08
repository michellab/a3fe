"""Unit and regression tests for the SomdConfig class."""

from tempfile import TemporaryDirectory
import os

from a3fe import SomdConfig

def create_test_input_dir():
    """Create a temporary directory with a mock somd.cfg file."""
    temp_dir = TemporaryDirectory()
    with open(os.path.join(temp_dir.name, "somd.cfg"), "w") as f:
        f.write("ncycles = 1000\n")
    return temp_dir

def test_create_config():
    """Test that the config can be created."""
    with create_test_input_dir() as input_dir:
        # Test with integer runtime
        config = SomdConfig(
            runtime=1,  # Integer runtime
            input_dir=input_dir
        )
        assert isinstance(config, SomdConfig)
        
        # Test with float runtime
        config = SomdConfig(
            runtime=0.3,  # Float runtime
            input_dir=input_dir
        )
        assert isinstance(config, SomdConfig)

def test_config_pickle_and_load():
    """Test that the config can be pickled and loaded."""
    with create_test_input_dir() as input_dir:
        with TemporaryDirectory() as dirname:
            config = SomdConfig(
                runtime=1,  # Integer runtime
                input_dir=input_dir
            )
            config.dump(dirname)
            config2 = SomdConfig.load(dirname)
            assert config == config2

def test_get_somd_config():
    """Test that the SOMD configuration file is generated correctly."""
    with create_test_input_dir() as input_dir:
        with TemporaryDirectory() as dirname:
            config = SomdConfig(
                integrator="langevinmiddle",
                nmoves=25000,
                timestep=4.0,
                runtime=1,  # Integer runtime
                input_dir=input_dir,
                cutoff_type="PME",
                thermostat=False
            )
            config_path = config.get_somd_config(
                run_dir=dirname,
                config_name="test"
            )
            assert config_path == os.path.join(dirname, "test.cfg")

            expected_config = (
                "### Integrator ###\n"
                "nmoves = 25000\n"
                "timestep = 4.0 * femtosecond\n"
                "constraint = hbonds\n"
                "hydrogen mass repartitioning factor = 3.0\n"
                "integrator = langevinmiddle\n"
                "inverse friction = 1.0 * picosecond\n"
                "temperature = 25.0 * celsius\n"
                "thermostat = False\n"
                "\n"
                "### Barostat ###\n"
                "barostat = True\n"
                "pressure = 1.0 * atm\n"
                "\n"
                "### Non-Bonded Interactions ###\n"
                "cutoff type = PME\n"
                "cutoff distance = 10.0 * angstrom\n"
                "\n"
                "### Trajectory ###\n"
                "buffered coordinates frequency = 5000\n"
                "center solute = True\n"
                "\n"
                "### Minimisation ###\n"
                "minimise = True\n"
                "\n"
                "### Alchemistry ###\n"
                "perturbed residue number = 1\n"
                "energy frequency = 200\n"
            )

            with open(config_path, "r") as f:
                config_content = f.read()

            assert config_content == expected_config

def test_constraint_validation():
    """Test constraint type validation."""
    with create_test_input_dir() as input_dir:
        config = SomdConfig(
            constraint="hbonds",
            runtime=1,  # Integer runtime
            input_dir=input_dir
        )
        assert config.constraint == "hbonds"
        
        config = SomdConfig(
            constraint="all-bonds",
            runtime=0.3,  # Float runtime
            input_dir=input_dir
        )
        assert config.constraint == "all-bonds"

        try:
            SomdConfig(
                constraint="invalid",
                runtime=1,  # Integer runtime
                input_dir=input_dir
            )
            assert False, "Should raise ValueError"
        except ValueError:
            pass

def test_cutoff_type_validation():
    """Test cutoff type validation."""
    with create_test_input_dir() as input_dir:
        # Test PME cutoff type
        config = SomdConfig(
            cutoff_type="PME",
            runtime=1,  # Integer runtime
            input_dir=input_dir
        )
        assert config.cutoff_type == "PME"
        assert config.cutoff_distance == 10.0
        assert config.reaction_field_dielectric is None

        # Test cutoffperiodic type
        config = SomdConfig(
            cutoff_type="cutoffperiodic",
            runtime=0.3,  # Float runtime
            input_dir=input_dir
        )
        assert config.cutoff_type == "cutoffperiodic"
        assert config.cutoff_distance == 12.0
        assert config.reaction_field_dielectric == 78.3

def test_integrator_thermostat_validation():
    """Test integrator and thermostat validation."""
    with create_test_input_dir() as input_dir:
        # Valid configuration
        config = SomdConfig(
            integrator="langevinmiddle",
            thermostat=False,
            runtime=1,  # Integer runtime
            input_dir=input_dir
        )
        assert config.integrator == "langevinmiddle"
        assert config.thermostat is False

        # Invalid configuration
        try:
            SomdConfig(
                integrator="langevinmiddle",
                thermostat=True,
                runtime=1,  # Integer runtime
                input_dir=input_dir
            )
            assert False, "Should raise ValueError"
        except ValueError:
            pass

def test_lambda_validation():
    """Test lambda array and value validation."""
    with create_test_input_dir() as input_dir:
        config = SomdConfig(
            lambda_array=[0.0, 0.5, 1.0],
            lambda_val=0.5,
            runtime=1,  # Integer runtime
            input_dir=input_dir
        )
        assert config.lambda_array == [0.0, 0.5, 1.0]
        assert config.lambda_val == 0.5

def test_restraints_configuration():
    """Test restraints configuration options."""
    with create_test_input_dir() as input_dir:
        config = SomdConfig(
            use_boresch_restraints=True,
            receptor_ligand_restraints=True,
            runtime=1,  # Integer runtime
            input_dir=input_dir
        )
        assert config.use_boresch_restraints is True
        assert config.receptor_ligand_restraints is True

def test_alchemical_files():
    """Test alchemical transformation file paths."""
    with create_test_input_dir() as input_dir:
        config = SomdConfig(
            morphfile="/path/to/morph.pert",
            topfile="/path/to/system.top",
            crdfile="/path/to/system.crd",
            runtime=1,  # Integer runtime
            input_dir=input_dir
        )
        assert config.morphfile == "/path/to/morph.pert"
        assert config.topfile == "/path/to/system.top"
        assert config.crdfile == "/path/to/system.crd"

def test_get_somd_config_with_extra_options():
    """Test SOMD configuration generation with extra options."""
    with create_test_input_dir() as input_dir:
        with TemporaryDirectory() as dirname:
            config = SomdConfig(
                integrator="langevinmiddle",
                nmoves=25000,
                timestep=4.0,
                runtime=1,  # Integer runtime
                input_dir=input_dir,
                cutoff_type="PME",
                thermostat=False,
                extra_options={"custom_option": "value"}
            )
            config_path = config.get_somd_config(
                run_dir=dirname,
                config_name="test_extra"
            )
            
            with open(config_path, "r") as f:
                config_content = f.read()
                
            assert "### Extra Options ###" in config_content
            assert "custom_option = value" in config_content