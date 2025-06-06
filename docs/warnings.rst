Warnings
========

 - ABFE calculations with membrane proteins are supported via GROMACS for minimization, NVT, NPT, and ensemble equilibration. 
   However, the configuration file (gromacs.mdp) used for these preparation steps is general and non-specialized. 
   It is currently recommended that users perform the pre-equilibration procedures externally. 
   Nevertheless, future releases will include support for these steps and we also plan to implement support ABFE calculations with GROMACS in the future.