===============
Change Log
===============

0.3.2
====================
- Fix bug which caused somd.rst7 files in the ensemble equilibration directories to be incorrectly numbered in some cases.
- Fix bug which caused the output directory to be incorrectly replaced with "output" in some cases.

0.3.1
====================
- Modified the SimulationRunnerIterator to ensure that the state of sub-simulation runners is saved before they are torn down. This means that setting the equilibration time at the CalcSet level will now work as expected (previously the state of the calculations was not saved, causing the analysis to fail).
- Updated read_overlap_mat so that it works when "#Overlap" is not printed to a new line (but is included at the end of the previous line). This seems to happen when MBAR requires many iterations to converge.

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


