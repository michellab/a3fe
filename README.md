a3fe
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/michellab/a3fe/workflows/CI/badge.svg)](https://github.com/fjclark/a3fe/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/michellab/a3fe/graph/badge.svg?token=5IGO8SCRRQ)](https://codecov.io/gh/michellab/a3fe)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/a3fe/badge/?version=latest)](https://a3fe.readthedocs.io/en/latest/?badge=latest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
      
<img src="./a3fe_logo.png" alt="Alt text" style="width: 50%; height: 50%;">

**A**utomated **A**daptive **A**bsolute alchemical **F**ree **E**nergy calculator. A package for running adaptive alchemical absolute binding free energy calculations with SOMD (distributed within [sire](https://sire.openbiosim.org/)) using adaptive protocols based on an ensemble of simulations. This requires the SLURM scheduling system. Please see the [**documentation**](https://a3fe.readthedocs.io/en/latest/?badge=latest).

For details of the algorithms and testing, please see the assocated prerint: 

**Clark, F.; Robb, G.; Cole, D.; Michel, J. Automated Adaptive Absolute Binding Free Energy Calculations. ChemRxiv June 18, 2024. https://doi.org/10.26434/chemrxiv-2024-3ft7f.**

### Installation

a3fe depends on SLURM for scheduling jobs, and on GROMACS for running initial equilibration simulations. Please ensure that your have sourced your GMXRC or loaded your GROMACS module before proceeding with the installation. While we recommend installing with [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), you can substitute `mamba` with `conda` in the following commands.

Now, download and install a3fe:
```bash
git clone https://github.com/michellab/a3fe.git
cd a3fe
mamba env create -f environment.yaml
python -m pip install --no-deps .
```

### Quick Start

- Activate your a3fe conda environment 
- Create a base directory for the calculation and create an directory called `input` within this
- Move your input files into the the input directory. For example, if you have parameterised AMBER-format input files, name these bound_param.rst7, bound_param.prm7, free_param.rst7, and free_param.prm7. For more details see the documentation. Alternatively, copy the example input files from a3fe/a3fe/data/example_run_dir to your input directory.
- Copy run somd.sh and template_config.sh from a3fe/a3fe/data/example_run_dir to your `input` directory, making sure to the SLURM options in run_somd.sh so that the jobs will run on your cluster
- In the calculation base directory, run the following python code, either through ipython or as a python script (you will likely want to run the script with `nohup`or use ipython through tmux to ensure that the calculation is not killed when you lose connection)

```python
import a3fe as a3 
calc = a3.Calculation(ensemble_size=5)
calc.setup()
calc.get_optimal_lam_vals()
calc.run(adaptive=False, runtime = 5) # Run non-adaptively for 5 ns per replicate
calc.wait()
calc.set_equilibration_time(1) # Discard the first ns of simulation time
calc.analyse()
calc.save()
```

- Check the results in the ``output`` directories (separate output directories are created for the Calculation, Legs, and Stages)

### Copyright

Copyright (c) 2023, Finlay Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
