# Data for Examples and Testing

- `example_exp_dgs`: Example of how experimental data should be structured for use with the `calc_set` class.
- `example_gromacs_discharge_stage`: Example of a complete charged-ligand GROMACS discharge stage. Trajectory and restart data have been removed. The independent GROMACS 2025.0 BAR references for runs 1 and 2 are 169.1505 and 168.6743 kcal mol^-1, respectively. They were calculated over 20--100 ps using `gmx bar -f output/lambda_*/run_01/prod/prod.xvg -b 20 -e 100` (and equivalently for run 2), then converted from kJ mol^-1 by dividing by 4.184.
- `gromacs_integration_input`: Minimal neutral free-leg discharge input for GROMACS SLURM integration testing.
- `example_output`: Example SOMD output data for testing.
- `example_restraint_stage`: Example of a complete restraint stage. Trajectory data has been removed.
- `example_run_dir`: Example of an almost complete calculation (although only with the `discharge` stages) which has not yet been run.
