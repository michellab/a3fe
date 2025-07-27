"""
SLURM script generator for A3FE calculations.

This module provides Python classes to generate SLURM submit scripts
for A3FE preparation jobs and SOMD simulations.
"""

from pathlib import Path
from typing import Optional, Union, Any
from pydantic import BaseModel, Field



class A3feSlurmParameters(BaseModel):
    """Pydantic BaseModel for A3FE SLURM submit script parameters with validation."""
    
    # Basic SLURM parameters
    account: str = Field(default="def-mkoz", description="SLURM account")
    job_name: str = Field(default="a3fe_job", description="Job name")
    cpus_per_task: int = Field(default=1, ge=1, description="CPUs per task")
    gres: str = Field(default="", description="Generic resources (GPUs), e.g. 'gpu:v100:1'")
    time: str = Field(default="00:30:00", description="Wall time limit")
    mem: str = Field(default="4G", description="Memory requirement")
    
    # Output files
    output_file: str = Field(default="job-%A.%a.out", description="Standard output file")
    error_file: str = Field(default="job-%A.%a.err", description="Standard error file")
    
    # Module system
    purge_modules: bool = Field(default=True, description="Whether to purge modules first")
    modules_to_load: list[str] = Field(
        default=["StdEnv/2020", "gcc/9.3.0", "cuda/11.8.0", "openmpi/4.0.3", "gromacs/2023"],
        description="Modules to load"
    )
    
    # Conda environment
    conda_init_script: str = Field(
        default="~/miniconda3/etc/profile.d/conda.sh",
        description="Path to conda initialization script"
    )
    conda_env: str = Field(default="a3fe_gra", description="Conda environment name")
    
    # Environment variables
    setup_cuda_env: bool = Field(default=True, description="Whether to set up CUDA environment")
    somd_platform: str = Field(default="CUDA", description="SOMD platform (CUDA or CPU)")

    # pre/post command sequences
    pre_commands: list[str] = Field(
        default_factory=list,
        description="Commands to run before main execution"
    )
    post_commands: list[str] = Field(
        default_factory=list,
        description="Commands to run after main execution"
    )
    
    # Additional SLURM directives
    custom_directives: dict[str, str] = Field(
        default_factory=dict, 
        description="Additional SLURM directives"
    )

    class Config:
        extra = "forbid"
        validate_assignment = True


class A3feSlurmGenerator:
    """Generates SLURM submit scripts for A3FE calculations."""
    
    def __init__(self, base_params: Optional[A3feSlurmParameters] = None) -> None:
        """
        Initialize A3FE SLURM submit script generator.
        
        Parameters
        ----------
        base_params : A3feSlurmParameters, optional
            Base SLURM parameters. If None, uses default parameters.
        """
        self.base_params = base_params or A3feSlurmParameters()
    
    def generate_prep_script(
        self,
        job_name: str,
        python_command: str,
        custom_overrides: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generate SLURM script for A3FE preparation jobs.
        
        Parameters
        ----------
        job_name : str
            Name of the job
        python_command : str
            Python command to execute
        custom_overrides : dict, optional
            Custom parameter overrides
        pre_commands : list, optional
            Additional commands to run before main command
        post_commands : list, optional
            Additional commands to run after main command
            
        Returns
        -------
        str
            SLURM script content
        """
        # Create a copy of base parameters
        params = self.base_params.copy(deep=True)
        
        # Set job-specific parameters
        params.job_name = job_name
        params.output_file = f"{job_name}-%A.%a.out"
        params.error_file = f"{job_name}-%A.%a.err"
        
        # Apply custom overrides
        if custom_overrides:
            for key, value in custom_overrides.items():
                if hasattr(params, key):
                    setattr(params, key, value)
                else:
                    params.custom_directives[key] = value
        
        return self._format_prep_script(
            params=params,
            python_command=python_command,
        )
    
    def generate_somd_script(
        self,
        job_name: str,
        custom_overrides: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generate SLURM script for SOMD simulations.
        
        Parameters
        ----------
        job_name : str
            Name of the job
        custom_overrides : dict, optional
            Custom parameter overrides
        pre_commands : list, optional
            Additional commands to run before main command
        post_commands : list, optional
            Additional commands to run after main command
            
        Returns
        -------
        str
            SLURM script content
        """
        # Create a copy of base parameters
        params = self.base_params.copy(deep=True)
        
        # Set job-specific parameters for SOMD
        params.job_name = job_name
        params.output_file = f"{job_name}-%A.%a.out"
        params.error_file = f"{job_name}-%A.%a.err"
        params.time = "24:00:00"  # Longer time for simulations
        params.mem = "4G"  # More memory for simulations
        
        # Apply custom overrides
        if custom_overrides:
            for key, value in custom_overrides.items():
                if hasattr(params, key):
                    setattr(params, key, value)
                else:
                    params.custom_directives[key] = value
        
        return self._format_somd_script(
            params=params,
        )
    
    def _format_prep_script(
        self,
        params: A3feSlurmParameters,
        python_command: str,
    ) -> str:
        """Format parameters into preparation script content."""
        lines = [
            "#!/bin/bash",
            "",
            # SLURM directives
            f"#SBATCH --account={params.account}",
            f"#SBATCH --job-name={params.job_name}",
            f"#SBATCH --cpus-per-task={params.cpus_per_task}",
        ]
        
        # Only add gres directive if it's not empty
        if params.gres.strip():
            lines.append(f"#SBATCH --gres={params.gres}")
        
        lines.extend([
            f"#SBATCH --time={params.time}",
            f"#SBATCH --error={params.error_file}",
            f"#SBATCH --output={params.output_file}",
            f"#SBATCH --mem={params.mem}",
        ])
        
        # Add custom SLURM directives
        for key, value in params.custom_directives.items():
            if value:  # Only add if value is not empty
                lines.append(f"#SBATCH --{key}={value}")
            else:  # For flags without values
                lines.append(f"#SBATCH --{key}")
        
        lines.extend(["", "", ""])
        
        # Module loading
        if params.purge_modules:
            lines.append("module --force purge")
        
        if params.modules_to_load:
            # Filter out CUDA module if not using CUDA
            modules = params.modules_to_load.copy()
            if not params.setup_cuda_env and params.somd_platform.upper() == "CPU":
                # Remove CUDA-related modules
                modules = [mod for mod in modules if "cuda" not in mod.lower()]
            
            if modules:  # Only add if there are modules to load
                module_line = "module load " + "  ".join(modules)
                lines.append(module_line)
        
        lines.extend(["", ""])
        
        # Conda setup
        lines.extend([
            "# initialize and activate conda",
            f". {params.conda_init_script}",
            f"conda activate {params.conda_env}",
            "",
            ""
        ])
        
        # Environment setup
        # TODO: this is needed for running GPU in Graham
        if params.setup_cuda_env:
            lines.extend([
                "unset LD_LIBRARY_PATH",
                'export LD_LIBRARY_PATH="$CUDA_HOME/lib64"',
                "",
                ""
            ])
        
        # Pre-commands
        if params.pre_commands:
            lines.extend([
                "# Pre-execution commands",
                *params.pre_commands,
                "",
            ])
        
        # Main command
        lines.extend([
            "# Main execution",
            python_command,
            ""
        ])
        
        # Post-commands
        if params.post_commands:
            lines.extend([
                "# Post-execution commands",
                *params.post_commands,
                "",
            ])
        
        return "\n".join(lines) + "\n"
    
    # TODO: consolidate this with _format_prep_script()
    def _format_somd_script(
        self,
        params: A3feSlurmParameters,
        # pre_commands: Optional[list[str]] = None,
        # post_commands: Optional[list[str]] = None,
    ) -> str:
        """Format parameters into SOMD script content."""
        lines = [
            "#!/bin/bash",
            "",
            # SLURM directives
            f"#SBATCH --account={params.account}",
            f"#SBATCH --job-name={params.job_name}",
            f"#SBATCH --cpus-per-task={params.cpus_per_task}",
        ]
        
        # Only add gres directive if it's not empty
        if params.gres.strip():
            lines.append(f"#SBATCH --gres={params.gres}")
        
        lines.extend([
            f"#SBATCH --time={params.time}",
            f"#SBATCH --error={params.error_file}",
            f"#SBATCH --output={params.output_file}",
            f"#SBATCH --mem={params.mem}",
        ])
        
        # Add custom SLURM directives
        for key, value in params.custom_directives.items():
            if value:  # Only add if value is not empty
                lines.append(f"#SBATCH --{key}={value}")
            else:  # For flags without values
                lines.append(f"#SBATCH --{key}")
        
        lines.extend(["", "", ""])
        
        # Module loading
        if params.purge_modules:
            lines.append("module --force purge")
        
        if params.modules_to_load:
            module_line = "module load " + "  ".join(params.modules_to_load)
            lines.append(module_line)
        
        lines.extend(["", ""])
        
        # Conda setup
        lines.extend([
            "# initialize and activate conda",
            f". {params.conda_init_script}",
            f"conda activate {params.conda_env}",
            "",
            ""
        ])
        
        # Environment setup (SOMD always needs CUDA)
        if params.setup_cuda_env and params.somd_platform.upper() == "CUDA":
            lines.extend([
                "unset LD_LIBRARY_PATH",
                'export LD_LIBRARY_PATH="$CUDA_HOME/lib64"',
                "",
                ""
            ])
        
        # Pre-commands
        if params.pre_commands:
            lines.extend([
                "# Pre-execution commands",
                *params.pre_commands,
                "",
            ])
        
        # SOMD execution
        lines.extend([
            "lam=$1",
            'echo "lambda is: $lam"',
            f"srun somd-freenrg -C somd.cfg -l $lam -p {params.somd_platform.upper()}",
            ""
        ])
        
        # Post-commands
        if params.post_commands:
            lines.extend([
                "# Post-execution commands",
                *params.post_commands,
                "",
            ])
        
        return "\n".join(lines) + "\n"
    
    def write_script(
        self,
        output_path: Union[str, Path],
        script_content: str,
    ) -> None:
        """
        Write script to disk and make it executable.
        
        Parameters
        ----------
        output_path : str or Path
            Path to write the script
        script_content : str
            Content of the script
        """
        output_file = Path(output_path)
        output_file.write_text(script_content)
        output_file.chmod(0o755)  # Make executable


def create_default_a3fe_generator() -> A3feSlurmGenerator:
    """Create A3FE SLURM generator with default parameters."""
    return A3feSlurmGenerator()


def create_custom_a3fe_generator(**kwargs) -> A3feSlurmGenerator:
    """Create A3FE SLURM generator with custom parameters."""
    params = A3feSlurmParameters(**kwargs)
    return A3feSlurmGenerator(params)