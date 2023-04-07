EnsEquil
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/fjclark/EnsEquil/workflows/CI/badge.svg)](https://github.com/fjclark/EnsEquil/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fjclark/EnsEquil/branch/main/graph/badge.svg?token=UMH0OUSUJY)](https://codecov.io/gh/fjclark/EnsEquil)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/ensequil/badge/?version=latest)](https://ensequil.readthedocs.io/en/latest/?badge=latest)
      


A package for running free energy calculations with SOMD with automated equilibration detection based on an ensemble of simulations. This requires the SLURM scheduling system.

### Installation

Assuming that you already have SLURM installed, we first need to install the [BioSimSpace](https://biosimspace.openbiosim.org/) dependencies using mamba (or conda). SOMD, which is used to run the simulations, is contained within Sire, which will be installed as a dependency of BioSimSpace.
```bash
mamba create -n ensequil -c conda-forge -c openbiosim/label/dev biosimspace --only-deps
mamba activate ensequil
mamba install -c conda-forge ambertools pytest watchdog
```
Now download BioSimSpace, install, and test:
```bash
git clone https://github.com/fjclark/BioSimSpace.git
cd biosimspace/python
git checkout feature_abfe_somd
BSS_SKIP_DEPDENCIES=1 python setup.py develop
cd ..
pytest test
cd ..
```
Finally, download EnsEquil, install, and test:
 ```bash
 git clone https://github.com/fjclark/EnsEquil.git
 cd EnsEquil
 pip install .
 pytest EnsEquil
 ```
 
### Examples

Create a run directory and copy `EnsEquil/EnsEquil/data/example_input` to <your run directory>/input. Replace the `system.*` and `somd.pert` files as you would for a standard SOMD simulation (these can be generated using [BioSimSpace](https://biosimspace.openbiosim.org/)), modify `somd.cfg` to change the simulation settings, and change the SLURM options in `run_somd.sh` to fit your SLURM installation. EnsEquil can then be run through ipython:

```
ipython
import EnsEquil as ee
ens = ee.Stage(block_size=1) # Use a block size of 1 ns to detect equilibration
ens.run()
```
Creating the `ens` ensemble object will result in the creation of all required output directories. After starting the run with `ens.run()`, the ensemble will run in the background (and will be killed if you exit ipython) and you will be able to query the state of, and interact with, the ensemble object.

### Copyright

Copyright (c) 2023, Finlay Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
