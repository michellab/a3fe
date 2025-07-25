"""
SLURM configurations for A3FE preparation steps.

This module contains default SLURM configurations for different preparation steps,
separate from the Leg class for better organization and reusability.
"""
from .slurm_script_generator import A3feSlurmParameters


# Default SLURM configurations for different preparation steps
DEFAULT_PREP_STEP_CONFIGS = {
    "parameterise": A3feSlurmParameters(
        job_name="param",
        cpus_per_task=1,
        gres="",  # CPU-only for parameterization
        time="00:30:00",
        mem="2G",
        modules_to_load=["StdEnv/2020", "gcc/9.3.0", "openmpi/4.0.3", "gromacs/2023"],
        setup_cuda_env=False
    ),
    "solvate": A3feSlurmParameters(
        job_name="solvate",
        cpus_per_task=1,
        gres="",  # CPU-only for solvation
        time="00:30:00", 
        mem="2G",
        modules_to_load=["StdEnv/2020", "gcc/9.3.0", "openmpi/4.0.3", "gromacs/2023"],
        setup_cuda_env=False
    ),
    "minimise": A3feSlurmParameters(
        job_name="minimise",
        cpus_per_task=2,
        gres="",  # CPU-only for GROMACS minimization
        time="00:60:00",
        mem="2G", 
        modules_to_load=["StdEnv/2020", "gcc/9.3.0", "cuda/11.8.0", "openmpi/4.0.3", "gromacs/2023"],
        setup_cuda_env=True
    ),
    "heat_preequil": A3feSlurmParameters(
        job_name="heat_preequil",
        cpus_per_task=4,
        gres="gpu:v100:1",  # GPU for GROMACS heating/equilibration
        time="03:00:00",
        mem="2G",
        modules_to_load=["StdEnv/2020", "gcc/9.3.0", "cuda/11.8.0", "openmpi/4.0.3", "gromacs/2023"],
        setup_cuda_env=True
    ),
    "ensemble_equil": A3feSlurmParameters(
        job_name="ensemble_equil",
        cpus_per_task=4,
        gres="gpu:v100:1",  # GPU for SOMD ensemble equilibration
        time="12:00:00",
        mem="2G",
        modules_to_load=["StdEnv/2020", "gcc/9.3.0", "cuda/11.8.0", "openmpi/4.0.3", "gromacs/2023"],
        setup_cuda_env=True
    ),
    "somd_production": A3feSlurmParameters(
        job_name="somd",
        cpus_per_task=4,
        gres="gpu:v100:1",  # GPU for SOMD production runs
        time="24:00:00", 
        mem="4G",
        modules_to_load=["StdEnv/2020", "gcc/9.3.0", "cuda/11.8.0", "openmpi/4.0.3", "gromacs/2023"],
        setup_cuda_env=True
    )
}


class SimulationSlurmConfigs:
    """
    Manager class for SLURM configurations.
    
    This class provides a centralized way to manage and customize SLURM configurations
    for different preparation steps, with support for site-specific customizations.
    """
    
    def __init__(self, custom_configs: dict[str, A3feSlurmParameters] = None):
        """
        Initialize with optional custom configurations.
        
        Parameters
        ----------
        custom_configs : Dict[str, A3feSlurmParameters], optional
            Custom configurations that override defaults
        """
        # Start with default configurations
        self._configs = {
            step: config.copy(deep=True) 
            for step, config in DEFAULT_PREP_STEP_CONFIGS.items()
        }
        
        # Apply any custom configurations
        if custom_configs:
            for step, config in custom_configs.items():
                if step in self._configs:
                    self._configs[step] = config.copy(deep=True)
                else:
                    raise ValueError(f"Unknown preparation step: {step}")
    
    def get_config(self, step_type: str) -> A3feSlurmParameters:
        """
        Get configuration for a specific preparation step.
        
        Parameters
        ----------
        step_type : str
            The preparation step type
            
        Returns
        -------
        A3feSlurmParameters
            Copy of the configuration for the specified step
        """
        if step_type not in self._configs:
            raise ValueError(f"Unknown step type: {step_type}. Available: {list(self._configs.keys())}")
        return self._configs[step_type].copy(deep=True)
    
    def update_config(self, step_type: str, **kwargs) -> None:
        """
        Update configuration for a specific step type.
        
        Parameters
        ----------
        step_type : str
            The preparation step type to update
        **kwargs
            Parameters to update in the configuration
        """
        if step_type not in self._configs:
            raise ValueError(f"Unknown step type: {step_type}")
        
        config = self._configs[step_type]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.custom_directives[key] = str(value)
    
    def get_all_configs(self) -> dict[str, A3feSlurmParameters]:
        """Get all current configurations."""
        return {
            step: config.copy(deep=True) 
            for step, config in self._configs.items()
        }
    
    def list_steps(self) -> list:
        """List all available preparation steps."""
        return list(self._configs.keys())
    
    def reset_to_defaults(self, step_type: str = None) -> None:
        """
        Reset configuration(s) to defaults.
        
        Parameters
        ----------
        step_type : str, optional
            Specific step to reset. If None, reset all steps.
        """
        if step_type:
            if step_type not in self._configs:
                raise ValueError(f"Unknown step type: {step_type}")
            self._configs[step_type] = DEFAULT_PREP_STEP_CONFIGS[step_type].copy(deep=True)
        else:
            self._configs = {
                step: config.copy(deep=True) 
                for step, config in DEFAULT_PREP_STEP_CONFIGS.items()
            }
    
    def create_site_specific_configs(
        self, 
        account: str,
        modules: list = None,
        conda_env: str = None,
        base_gres: str = "gpu:v100:1"
    ) -> None:
        """
        Apply site-specific settings to all configurations.
        
        Parameters
        ----------
        account : str
            SLURM account to use
        modules : list, optional
            List of modules to load for all steps
        conda_env : str, optional
            Conda environment name
        base_gres : str, optional
            Default GPU specification for GPU steps
        """
        updates = {"account": account}
        
        if modules:
            updates["modules_to_load"] = modules
        if conda_env:
            updates["conda_env"] = conda_env
        
        # Apply to all steps
        for step_type in self._configs:
            self.update_config(step_type, **updates)
            
            # Update GPU specifications for GPU steps, but preserve CPU-only steps
            if self._configs[step_type].gres.strip():  # Only if step uses GPU
                self.update_config(step_type, gres=base_gres)


# Global instance for easy access (can be customized)
default_slurm_configs = SimulationSlurmConfigs()


# Convenience functions for backward compatibility and ease of use
def get_slurm_config(step_type: str) -> A3feSlurmParameters:
    """Get SLURM configuration for a preparation step."""
    return default_slurm_configs.get_config(step_type)


def update_slurm_config(step_type: str, **kwargs) -> None:
    """Update SLURM configuration for a preparation step."""
    default_slurm_configs.update_config(step_type, **kwargs)


def reset_slurm_configs(step_type: str = None) -> None:
    """Reset SLURM configurations to defaults."""
    default_slurm_configs.reset_to_defaults(step_type)


def setup_site_configs(account: str, **kwargs) -> None:
    """Set up site-specific SLURM configurations."""
    default_slurm_configs.create_site_specific_configs(account, **kwargs)


# TODO: Predefined site configurations; needs to be updated
def setup_compute_canada_configs(account: str, cluster: str = "cedar") -> None:
    """
    Set up configurations for Compute Canada clusters.
    
    Parameters
    ----------
    account : str
        Your Compute Canada account (e.g., "def-username")
    cluster : str
        Cluster name ("cedar", "graham", "beluga", "narval")
    """
    cluster_configs = {
        "cedar": {
            "modules": ["StdEnv/2020", "gcc/9.3.0", "cuda/11.4", "openmpi/4.0.3", "gromacs/2021.3"],
            "base_gres": "gpu:v100l:1"
        },
        "graham": {
            "modules": ["StdEnv/2020", "gcc/9.3.0", "cuda/11.4", "openmpi/4.0.3", "gromacs/2021.3"],
            "base_gres": "gpu:t4:1"
        },
        "beluga": {
            "modules": ["StdEnv/2020", "gcc/9.3.0", "cuda/11.4", "openmpi/4.0.3", "gromacs/2021.3"],
            "base_gres": "gpu:v100:1"
        },
        "narval": {
            "modules": ["StdEnv/2023", "gcc/12.3", "cuda/12.2", "openmpi/4.1.5", "gromacs/2024"],
            "base_gres": "gpu:a100:1"
        }
    }
    
    if cluster not in cluster_configs:
        raise ValueError(f"Unknown cluster: {cluster}. Available: {list(cluster_configs.keys())}")
    
    config = cluster_configs[cluster]
    setup_site_configs(
        account=account,
        modules=config["modules"],
        base_gres=config["base_gres"]
    )