"""Configuration classes for SLURM configuration."""

__all__ = [
    "SlurmConfig",
]

import yaml as _yaml
import subprocess as _subprocess
import re as _re

from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field
from pydantic import ConfigDict as _ConfigDict

import os as _os

from typing import List as _List, Dict as _Dict


class SlurmConfig(_BaseModel):
    """
    Pydantic model for holding a SLURM configuration.
    """

    partition: str = _Field("default", description="SLURM partition to submit to.")
    time: str = _Field("24:00:00", description="Time limit for the SLURM job.")
    gres: str = _Field("gpu:1", description="Resources to request - normally one GPU.")
    nodes: int = _Field(1, ge=1)
    ntasks_per_node: int = _Field(1, ge=1)
    output: str = _Field(
        "slurm-%A.%a.out", description="Output file for the SLURM job."
    )
    extra_options: _Dict[str, str] = _Field(
        {}, description="Extra options to pass to SLURM. For example, {'account': 'qt'}"
    )

    model_config = _ConfigDict(validate_assignment=True)

    def get_submission_cmds(
        self, cmd: str, run_dir: str, script_name: str = "a3fe"
    ) -> _List[str]:
        """
        Generates the SLURM submission commands list based on the configuration.

        Parameters
        ----------
        cmd : str
            Command to run during the SLURM job.

        run_dir : str
            Directory to submit the SLURM job from.

        script_name : str, optional, default="a3fe"
            Name of the script file to write. Note that when running many jobs from the
            same directory, this should be unique to avoid overwriting the script file.

        Returns
        -------
        List[str]
            The list of SLURM arguments.
        """
        # First, write the script to a file
        script_path = _os.path.join(run_dir, f"{script_name}.sh")

        script = (
            "#!/bin/bash\n"
            f"#SBATCH --partition={self.partition}\n"
            f"#SBATCH --time={self.time}\n"
            f"#SBATCH --gres={self.gres}\n"
            f"#SBATCH --nodes={self.nodes}\n"
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}\n"
            f"#SBATCH --output={self.output}\n"
        )

        for key, value in self.extra_options.items():
            script += f"#SBATCH --{key}={value}\n"

        script += f"\n{cmd}\n"

        with open(script_path, "w") as f:
            f.write(script)

        return ["sbatch", f"--chdir={run_dir}", script_path]

    def get_slurm_output_file_base(self, run_dir: str) -> str:
        """
        Get the base name of the SLURM output file.

        Parameters
        ----------
        run_dir : str
            Directory the job was submitted from.

        Returns
        -------
        str
            The base name of the SLURM output file.
        """
        return run_dir + "/" + self.output.split("%")[0]

    @classmethod
    def get_default_partition(cls) -> "str":
        """Get the default SLURM partition."""
        sinfo = _subprocess.run(
            ["sinfo", "-o", "%P", "-h"], stdout=_subprocess.PIPE, text=True
        )
        # Search for the default queue (marked with "*", then throw away the "*")
        return _re.search(r"([^\s]+)(?=\*)", sinfo.stdout).group(1)

    def dump(self, save_dir: str) -> None:
        """
        Dumps the configuration to a YAML file.

        Parameters
        ----------
        save_dir : str
            Directory to save the YAML file to.
        """
        model_dict = self.model_dump()

        save_path = save_dir + "/" + self.get_file_name()
        with open(save_path, "w") as f:
            _yaml.dump(model_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, load_dir: str) -> "SlurmConfig":
        """
        Loads the configuration from a YAML file.

        Parameters
        ----------
        load_dir : str
            Directory to load the YAML file from.

        Returns
        -------
        SlurmConfig
            The loaded configuration.
        """
        with open(load_dir + "/" + cls.get_file_name(), "r") as f:
            model_dict = _yaml.safe_load(f)
        return cls(**model_dict)

    @staticmethod
    def get_file_name() -> str:
        """
        Get the name of the SLURM configuration file.
        """
        return "slurm_config.yaml"
