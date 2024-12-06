===============
Change Log
===============

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


