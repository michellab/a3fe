"""Unit and regression tests for the SlurmConfig class."""

from tempfile import TemporaryDirectory


from unittest.mock import patch

from a3fe import SlurmConfig

import os


def test_create_default_config():
    """Test that the default config is created correctly."""
    config = SlurmConfig()
    assert config.partition == "default"


def test_modify_config():
    """Check that we can modify the config."""
    config = SlurmConfig()
    config.partition = "test_partition"


def test_config_pickle_and_load():
    """Test that the config can be pickled and loaded."""
    with TemporaryDirectory() as dirname:
        config = SlurmConfig()
        config.dump(dirname)
        config2 = SlurmConfig.load(dirname)
        assert config == config2


def test_get_submission_cmds():
    """
    Test that the submission commands are generated correctly
    and that the script is written correctly.
    """
    # Tmpdir to store the script
    with TemporaryDirectory() as dirname:
        config = SlurmConfig(
            partition="test_partition",
            time="24:00:00",
            gres="gpu:1",
            nodes=1,
            ntasks_per_node=1,
            output="test.out",
        )
        submission_cmds = config.get_submission_cmds(
            cmd="echo 'Test'", run_dir=dirname, script_name="test"
        )
        script_path = os.path.join(dirname, "test.sh")
        assert submission_cmds == ["sbatch", f"--chdir={dirname}", script_path]

        expected_script = (
            "#!/bin/bash\n"
            "#SBATCH --partition=test_partition\n"
            "#SBATCH --time=24:00:00\n"
            "#SBATCH --gres=gpu:1\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --output=test.out\n"
            "\necho 'Test'\n"
        )

        with open(script_path, "r") as f:
            script = f.read()

        assert script == expected_script


def test_get_submission_cmds_extra_options():
    """
    Test that the submission commands are generated correctly
    and that the script is written correctly with extra options.
    """
    # Tmpdir to store the script
    with TemporaryDirectory() as dirname:
        config = SlurmConfig(
            partition="test_partition",
            time="24:00:00",
            gres="gpu:1",
            nodes=1,
            ntasks_per_node=1,
            output="test.out",
        )
        config.extra_options = {"mem": "10G", "exclude": "node1"}
        _ = config.get_submission_cmds(
            cmd="echo 'Test'", run_dir=dirname, script_name="test"
        )
        script_path = os.path.join(dirname, "test.sh")

        with open(script_path, "r") as f:
            script = f.read()

        assert "#SBATCH --mem=10G\n" in script
        assert "#SBATCH --exclude=node1\n" in script


def test_get_output_file_base():
    """
    Test that the output file base is generated correctly.
    """
    config = SlurmConfig(output="slurm%j.out")
    output_file_base = config.get_slurm_output_file_base(run_dir="test_dir")
    assert output_file_base == "test_dir/slurm"


def test_get_default_partition():
    """
    Test that the default partition is set correctly.
    """
    with patch("subprocess.run") as mocked_run:
        # Define the mock return value
        mocked_run.return_value.stdout = (
            "serial*\nRTX4060\nGTX980\nRTX3080\ntest\nlong\n"
        )
        assert SlurmConfig.get_default_partition() == "serial"
