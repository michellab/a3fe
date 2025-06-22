===============
Change Log
===============

0.4.0
====================

**Note**: 
This is a breaking change! Old pickle files are not compatible with this version. 
If you need to load old pickle files, please revert to version 0.3.x.

- Decoupled SOMD engine from a3fe calculation and removed run_somd.sh and template_config.cfg, replacing them with a3fe.configuration.slurm_config.SlurmConfig and a3fe.configuration.engine_config.SomdConfig. 
- SystemPreparationConfigs saved as readable yaml file, rather than as a pickle. 
- Separated system preparation configuration from system preparation and configured SOMD with a3fe.configuration.system_prep_config.SomdSystemPreparationConfig.

**Migration Guide**:
In previous versions, users needed to configure run_somd.sh and template_config.cfg files to run calculations. 
However, with version 0.4.0, these files are no longer needed. Instead, configuration objects (SlurmConfig, SomdConfig, and SomdSystemPreparationConfig) are passed directly to the Calculation object.

For examples and detailed usage of these configuration objects, see the `Customising Calculations <guides.html#customising-calculations>`_ section in the documentation.

0.3.0
====================

- Improved CalcSet so that 1) get_optimal_lam_vals can be called directly on the CalcSet object, and 2) analyse can be called directly without requiring experimental results (by specifying compare_to_exp=False).
- Documentation on PME vs RF slightly improved.

0.2.1
====================

- Fixed issue https://github.com/michellab/a3fe/issues/14 where all jobs in the slurm queue were assumed to be
  pending or running. This caused problems when completed jobs remained in the queue.

0.2.0
====================

- Added ability to run charged compounds using the co-alchemical ion approach. Ligand net charge is detected from the input files and the config is checked to ensure that PME is used for electrostatics.
- Added parameterisation tests.
- Fixed bug arising from incorrect removal of "is_equilibrated" method (37de921)
- Made detection of username more robust (see https://github.com/michellab/a3fe/issues/8)

0.1.3
====================

- Ensured that config options are written correctly when the template config file does not terminate in a new line. Previously, new options would be appended to the last line of the file if it did not end with \n, which could happen if the user manually edited the file.

0.1.2
====================

- Fixed bug in ``get_slurm_file_base`` which caused the function to fail to read the output name if an "=" instead of a space was used to separate the argument and value.

v0.1.1
====================

- Ensured that ``simulation_runner`` objects only get pickled once they've been fully initialised. This avoids issues where an error occurs during initialisation and the object is pickled in an incomplete state. Addresses https://github.com/michellab/a3fe/issues/1.

v0.1.0
====================

- Initial release


