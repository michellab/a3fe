from pydantic import BaseModel as _BaseModel


class SlurmConfig(_BaseModel):
    """
    Pydantic model for holding SLURM configuration.
    """

    partition: str = "default"
    time: str = "24:00:00"
    nodes: int = 1
    ntasks_per_node: int = 1
    output: str = "slurm.out"
    error: str = "slurm.err"

    def to_slurm_header(self) -> str:
        """
        Generates the SLURM header string based on the configuration.
        """
        return (
            f"#SBATCH --partition={self.partition}\n"
            f"#SBATCH --time={self.time}\n"
            f"#SBATCH --nodes={self.nodes}\n"
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}\n"
            f"#SBATCH --output={self.output}\n"
            f"#SBATCH --error={self.error}\n"
        )
