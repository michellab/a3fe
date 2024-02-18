a3fe
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/fjclark/a3fe/workflows/CI/badge.svg)](https://github.com/fjclark/a3fe/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fjclark/a3fe/branch/main/graph/badge.svg?token=UMH0OUSUJY)](https://codecov.io/gh/fjclark/a3fe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/a3fe/badge/?version=latest)](https://a3fe.readthedocs.io/en/latest/?badge=latest)
      
A package for running adaptive alchemical absolute binding free energy calculations with SOMD based on an ensemble of simulations. This requires the SLURM scheduling system.

### Installation

a3fe depends on SLURM for scheduling jobs, and on GROMACS for running initial equilibration simulations. Please ensure that your have sourced your GMXRC or loaded your GROMACS module before proceeding with the installation.

We first need to install the [BioSimSpace](https://biosimspace.openbiosim.org/) dependencies using mamba (or conda). SOMD, which is used to run the simulations, is contained within Sire, which will be installed as a dependency of BioSimSpace.
```bash
mamba create -n a3fe -c conda-forge -c openbiosim/label/dev biosimspace
mamba activate a3fe
```
Now download a3fe, install, and test:
 ```bash
 git clone https://github.com/fjclark/a3fe.git
 cd a3fe
 pip install .
 pytest a3fe
 ```
 
### Quick Start

- Activate your a3fe conda environment 
- Create a base directory for the calculation and create an directory called `input` within this
- Move your input files into the the input directory. For example, if you have parameterised AMBER-format input files, name these bound_param.rst7, bound_param.prm7, free_param.rst7, and free_param.prm7. For more details see the documentation.
- Copy run somd.sh and template_config.sh from a3fe/a3fe/data/example_run_dir to your `input` directory, making sure to the SLURM options in run_somd.sh so that the jobs will run on your cluster
- In the calculation base directory, run the following python code, either through ipython or as a python script (you will likely want to run the script with `nohup`or use ipython through tmux to ensure that the calculation is not killed when you lose connection)

```python
import a3fe as a3 
calc = a3.Calculation()
calc.setup()
calc.get_optimal_lam_vals()
calc.run()
calc.wait()
calc.analyse()
```

- Check the results in the ``output`` directories (separate output directories are created for the Calculation, Legs, and Stages)

### Copyright

Copyright (c) 2023, Finlay Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
