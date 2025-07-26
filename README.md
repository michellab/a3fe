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

For details of the algorithms and testing, please see the assocated paper:

**Clark, F.; Robb, G. R.; Cole, D. J.; Michel, J. Automated Adaptive Absolute Binding Free Energy Calculations. J. Chem. Theory Comput. 2024, 20 (18), 7806–7828. https://doi.org/10.1021/acs.jctc.4c00806.**

### Installation

a3fe depends on SLURM for scheduling jobs, and on GROMACS for running initial equilibration simulations. Please ensure that your have sourced your GMXRC or loaded your GROMACS module before proceeding with the installation. While we recommend installing with [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), you can substitute `mamba` with `conda` in the following commands.

Now, download and install a3fe:
```bash
git clone https://github.com/michellab/a3fe.git
cd a3fe
mamba env create -f environment.yaml
mamba activate a3fe
python -m pip install --no-deps .
```

### How to run on different clusters with GPU (DRAC | Compute Canada)
- According to [creating and using a virtual environment](https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment), it's strongly recommended against using Conda or Anaconda in DRAC HPC. This will cause issues with setting CUDA because Conda or Anaconda will pull in its own compilers, CUDA runtimes, and library paths, and mess up with Modules in DRAC HPC cluster. This is also why we need to use `export LD_LIBRARY_PATH="$CUDA_HOME/lib64"` on Graham to get around this issue. However, this quick fix is not recommended, so better to use `mamba env create -f environment.yaml`!
- On Graham, we should use the following environment.yaml file: 
    ```
    name: a3fe_gra
    channels:
    - conda-forge
    - openbiosim

    dependencies:
        # Base depends
    - arviz
    - pandas
    - python>=3.9
    - pip
    - openmm<8.3.0
    - numpy
    - matplotlib
    - scipy
    - ipython
    - pymbar<4
    - ambertools
    - biosimspace<=2024.4.1
    - pydantic
    - loguru
    ```
  we should add these modules/headers into sbatch script for Graham:
    ```
    module --force purge
    module load StdEnv/2020  gcc/9.3.0  cuda/11.8.0  openmpi/4.0.3
    module load gromacs/2023

    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate a3fe_gra

    unset LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
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
- a up-to-date run script can be found here: `a3fe_jh/a3fe/data/example_run_jh/run_calc.py`
    - in the same folder, we can use `protein.pdb` and `ligand.sdf` as the input to quickly test the run

- Check the results in the ``output`` directories (separate output directories are created for the Calculation, Legs, and Stages)

### Some notes for solving runtime errors
- if `ensemble_equilibration_*` runs faild, we can simply remove the entire folder and re-run the calculation
- reloading molecules via `_BSS.IO.readMolecules()` leads to different molecule number (MolNum). This may cause issues when we resume a previously-stopped run
  - as a result, when using `skip_preparation=True` in Leg.setup(), we have to ensure that restraints are generated in the same run as pre-equilibrated system is loaded. In other words, `restraints.pkl` and `Leg.pkl` must be created in the same run
- pay attention to calculation.pkl (leg.pkl or stage.pkl) files when re-running a previously stopped calculation because the calculation will load these pickle file by default. These pickle files are use to load the previously saved `Calculation`, `Leg` and `Stage` objects.
### Copyright

Copyright (c) 2023, Finlay Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
