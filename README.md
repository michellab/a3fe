EnsEquil
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/fjclark/EnsEquil/workflows/CI/badge.svg)](https://github.com/fjclark/EnsEquil/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fjclark/EnsEquil/branch/main/graph/badge.svg?token=UMH0OUSUJY)](https://codecov.io/gh/fjclark/EnsEquil)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A package for running free energy calculations with SOMD with automated equilibration detection based on an ensemble of simulations. This requires the SLURM scheduling system.

### Installation

Assuming that you already have SLURM installed, install [BioSimSpace](https://biosimspace.openbiosim.org/) using mamba (or conda). SOMD, which is used to run the simulations, is contained within Sire, which will be installed as a dependency:

```bash
 mamba create -n ensequil "python<3.10"
 mamba activate ensequil
 mamba install -c openbiosim biosimspace
 ```
 
 Now download EnsEquil, install, and test:
 ```bash
 git clone https://github.com/fjclark/EnsEquil.git
 cd EnsEquil
 pip install .
 pytest EnsEquil
 ```
 
### Examples

Create a run directory and copy `EnsEquil/EnsEquil/data/example_input` to <your run directory>/input. Replace the `system.*` and `morph.pert` files as you would for a standard SOMD simulation (these can be generated using [BioSimSpace](https://biosimspace.openbiosim.org/)), modify `sim.cfg` to change the simulation settings, and change the SLURM options in `run_somd.sh` to fit your SLURM installation. EnsEquil can then be run through ipython:

```
ipython
import EnsEquil as ee
ens = ee.Ensemble(block_size=1) # Use a block size of 1 ns to detect equilibration
ens.run()
```
Creating the `ens` ensemble object will result in the creation of all required output directories. After starting the run with `ens.run()`, the ensemble will run in the background (and will be killed if you exit ipython) and you will be able to query the state of, and interact with, the ensemble object.

### Copyright

Copyright (c) 2023, Finlay Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
