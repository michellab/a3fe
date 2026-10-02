Warnings
========

 - We do not recommend running ABFE calculations with membrane proteins using the current version of ``a3fe``. This is because SOMD (Sire/ OpenMM Molecular Dynamics, part of Sire) is used for the free energy calculations (after setup with GROMACS), and currently SOMD uses an isotropic barostat.
 - Adaptive simulations are currently only supported with SOMD. GROMACS calculations should be run non-adaptively by setting ``adaptive=False`` and supplying a fixed runtime.
