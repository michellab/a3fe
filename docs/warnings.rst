Warnings
========

 - We do no recommend running ABFE calculations with membrane proteins using the current version of ``a3fe``. This is because SOMD (Sire/ OpenMM Molecular Dynamics, part of Sire) is used for the free energy calculations (after setup with GROMACS), and currently SOMD uses an isotropic barostat. We plan to implement support for ABFE with GROMACS in the future.