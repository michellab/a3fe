Getting Started
===============
EnsEquil is a package for running alchemical absolute binding free energy calculations with SOMD (Sire / OpenMM Molecular Dynamics) through SLURM. 
It is based on Sire(https://sire.openbiosim.org/) and also uses BioSimSpace(https://biosimspace.openbiosim.org/) during the set-up stages.

Installation
************
Please see the instructions on the github repo (https://github.com/fjclark/EnsEquil).

Quick Start
***********
- Activate your EnsEquil conda environment 
- Create a base directory for the calculation and create an directory called ``input`` within this
- Move your input files into the the input directory. For example, if you have parameterised AMBER-format input files, name these bound_param.rst7, bound_param.prm7, free_param.rst7, and free_param.prm7. For more details see Preparing Input for EnsEquil
- Copy run somd.sh and template_config.sh from EnsEquil/EnsEquil/data/example_run_dir to your ``input`` directory, making sure to the SLURM options in run_somd.sh so that the jobs will run on your cluster
- In the calculation base directory, run the following python code, either through ipython or as a python script (you will likely want to run this with ``nohup``/ through tmux to ensure that the calculation is not killed when you lose connection)

.. code-block:: python

    import EnsEquil as ee 
    calc = ee.Calculation()
    calc.setup()
    calc.run()
    calc.wait()
    calc.analyse()

- Check the results in the ``output`` directories (separate output directories are created for the Calculation, Legs, and Stages)

Some handy commands and code snippets, assuming that you have set up the calculation following the code above:

Terminate all the SLURM jobs:
``calc.kill()``
Delete all the output (but not input files), ready to run again:
``calc.clean()``
Return any failed simulations and find out where to look for output:
.. code-block:: python

    failed_sims = calc.failed_simulations
    for failed_sim in failed_sims:
        print(failed_sim.base_dir)

Check how many GPU hours your calculation has cost
.. code-block:: python

    print(f"Total GPU hours: {calc.tot_gpu_time:0.f}")
    print("#"*10)
    for leg in calc.legs:
        print(f"Total GPU hours for leg {leg.leg_type}: {leg.tot_gpu_time:.0f}")

EnsEquil Design
***************
EnsEquil stores and manipulates simulations based on a heirarchy of "simulation runners". Each simulation runner
is responsible for manipulating a set of "sub simulation runners". For example, ``Calculation`` objects hold and
manipulate two ``Leg`` objects (bound and free), and ``Leg`` objects hold and manipulate ``Stage`` objects. The ``Stage``
objects for first leg can be accessed through some Calculation instance ``calc`` with ``calc.legs[0].stages``. All simulation
runners are derived from the abstract base class ``SimulationRunner``, where as much as possible of the functionality
is defined. As a result, all simulation runners have similar interfaces e.g. they all share the run() and kill() methods. The heirarchy
of simulation runners is:

- Calculation
- Leg
- Stage
- LamWindow
- Simulation

Calling calc.kill(), for example, recursively kills all sub simulation runners and their sub simulation runners.

Each simulation runner logs to a file in its base directory named according to the class name (e.g. Calculation.log). In addition,
each simulation runner saves a pickled version of itself to a file named in the same way (e.g. Calculation.pkl), which
allows them to be restarted at any point. The pickle files are automatically detected and used to load the Simulation
runners when they are present in the base directory. For example, running ``calc = ee.Calculation()`` in the base directory of
an pre-prepared calculation will load the previous calculation, overwriting any arguments supplied to Calculation().

EnsEquil is designed to be easily adaptable to any SLURM cluster. The SLURM submission settings can be tailored by modifying
the header of ``template_config.cfg`` in the input directory.

EnsEquil aims to run ABFE as efficiently as possible, while generating robust estimates of uncertainty. A user-specified number of 
replicate simulations (this is specified when the simulation runner is created, e.g. ``calc = ee.Calculation(ensemble_size=5)``)
are run (the default is 5). This allows:

- A reasonablly robust estimate of the uncertainty from the inter-run differences
- More robust adaptive equilibration detection from the average gradient of dH/dlam with respect to simulation time

The adaptive equilibration algorithm decides if a set of repeat simulations at a given lambda window have equilibrated by averaging the
individual dH/dlam over all repeats, then smoothing the result with block averaging (the block size can be specified when creating
simulation runners, e.g. ``calc = ee.Calculation(block_size=1)``). If when the gradient of dH/dlam with respect to time falls below
some threshold, the calculation is taken to be equilibrated, and the simulations are terminated. Otherwise, EnsEquil automatically
submits new SLURM jobs. This algorithm will be refined in future.

If the input is not parameterised, EnsEquil will parameterise your input with ff14SB, OFF 2.0.0, and TIP3P. See 
"Preparing Input for EnsEquil". EnsEquil will solvate your system in a rhombic dodecahedral box with 150 mM NaCl
and perform a standard minimisation, heating, and pre-equilibration routine.

At present, EnsEquil uses GROMACS to run all set-up jobs, so please ensure that you have loaded the required CUDA and
GROMACS modules, or sourced GMXRC. These GROMACS jobs are also submitted through SLURM, and a unique 5 ns "ensemble
equilibration" simulation is run for each of the ``ensemble_size`` repeats. For the bound leg, these are used to extract
different restraints for each replicate simulation using the in-built BioSimSpace algorithm (see
https://github.com/fjclark/BioSimSpace/blob/01dba53b01386a3851e277874f9080c316c4632e/python/BioSimSpace/Sandpit/Exscientia/FreeEnergy/_restraint_search.py#L902).

EnsEquil can use a default spacing of lambda windows which should work reasonably for most systems with the default SOMD
settings. However, to optimise the lambda schedule by running short (200 ps default) simulations and generating a new spacing
according to the integrated variance of the gradients, run ``calc.get_optimal_lam_vals()``.

Preparing Input for EnsEquil
****************************
EnsEquil accepts either PDB files of the protein and crystallographic waters, along with an sdf file for the ligand,
or parameterised AMBER-format input files for the free and bound legs. The preparation stage will be detected
by EnsEquil when you instantiate a Calculation, and only the required preparation steps will be carried out for each
leg. 

To find out which input files are required for a given preparation stage for a given leg, run:

.. code-block:: python

    # Minimised parameterised structures for the free leg
    ee.PreparationStage.MINIMISED.get_simulation_input_files(ee.LegType.FREE)

.. list-table:: Preparation stage types and required input files
   :widths: 25 25 25 50
   :header-rows: 1
   
   * - PreparationStage
     - LegType
     - Required Input Files
     - Description
   * - STRUCTURES_ONLY
     - BOUND
     - protien.pdb, ligand.sdf (water.pdb)
     - The ligand free-protein structure, the ligand, and (optionally) the crystallographic waters
   * - STRUCTURES_ONLY
     - FREE
     - ligand.sdf
     - The ligand
   * - PARAMETERISED
     - BOUND
     - bound_param.prm7, bound_param.rst7
     - The AMBER parm7 and restart files for the complex, including crystallographic waters
   * - PARAMETERISED
     - FREE
     - free_param.prm7, free_param.rst7
     - The AMBER parm7 and restart files for the ligand
   * - SOLVATED
     - BOUND
     - bound_solv.prm7, bound_solv.rst7
     - The solvated complex with 150 mM NaCl
   * - SOLVATED
     - FREE
     - free_solv.prm7, free_solv.rst7
     - The solvated ligand with 150 mM NaCl
   * - MINIMISED
     - BOUND
     - bound_min.prm7, bound_min.rst7
     - The solvated complex after minimisation
   * - MINIMISED
     - FREE
     - free_min.prm7, free_min.rst7
     - The solvated ligand after minimisation
   * - PREEQUILIBRATED
     - BOUND
     - bound_preequil.prm7, bound_preequil.rst7
     - The solvated complex after heating and short initial equilibration steps
   * - PREEQUILIBRATED
     - FREE
     - free_preequil.prm7, free_preequil.rst7
     - The solvated ligand after heating and short initial equilibration steps

In addition, for every preparation stage, **run_somd.sh and template_config.cfg must be present in the input
directory.**

Please note that if you are suppling parameterised input files, **the ligand must be the first molecule in the system
and the ligand must be named "LIG"**. The former can be achieved by reordering the system with BioSimSpace, and the latter
by simply editing the ligand name in the prm7 files.

Running Simulations
*******************
Following the "Quick Start" guide will result in 5 repeat calculations being run for every lambda window. These will be adaptively
checked for equilibration and resubmitted if equilibration has not been achieved, as described in "EnsEquil Design". The defaults
can be modified when creating the Calculation, for example:

.. code-block:: python

    calc = ee.Calculation(block_size=3, threshold=0.5, ensemble_size=4)

EnsEquil is designed to be run adaptively, but can be run non-adaptively:

.. code-block:: python

    # Run all windows for 5 ns
    calc.run(adaptive=False, duration=5)
    calc.wait()
    # Check if we have equilibrated and analyse if so
    if calc.equilibrated:
        calc.analyse()
    else:
        print("Calculation not yet equilibrated")

If you want to run different stages for different amounts of time, this can be done:

.. code-block:: python

    # Alternatively, run the bound vanish leg for 8 ns and all other stages for 6 ns
    for leg in calc.legs: 
        for stage in leg.stages: 
            if leg.leg_type.value == 1 and stage.stage_type.value == 3: # Bound (1) vanish (3) 
                stage.run(adaptive=False, runtime=8) 
            else: 
                stage.run(adaptive=False, runtime=6)
    calc.wait()

    # Bypass the equliibration detection and set the equilibration time to 1 ns unless bound vanish, 
    # in which case 3 ns
    for leg in calc.legs: 
        for stage in leg.stages: 
            if leg.leg_type.value == 1 and stage.stage_type.value == 3: # Bound (1) vanish (3)
                for lam in stage.lam_windows: 
                    lam._equilibrated = True 
                    lam._equil_time=3
            else:
                for lam in stage.lam_windows: 
                    lam._equilibrated = True 
                    lam._equil_time=1

    res, err = calc.analyse()


To stop and restart a currently running calculation:

.. code-block:: python

    calc.kill()
    calc.clean()
    calc.run()

To run using optimal lambda window spacing based on short test simulations:

.. code-block:: python

    calc.get_optimal_lam_vals()
    calc.run()

Analysis
********
To analyse the free energy changes and create a variety of plots to aid analysis, run ``calc.analyse()``
and check the  ``output`` directories for the calculations, legs, stages, and lambda windows.

Convergence analysis involves repeatedly calculating the free energy changes with different subsets of the 
data, and is computationally intensive. Hence, it is implemented in a different function. To run convergence
analysis, enter ``calc.analyse_convergence()``. Plots of the free energy change against total simulation time
will be created in each output directory.

Note that **analysis is not performed through SLURM jobs and is CPU intensive**, so you may wish to switch to
e.g. an interactive session on a compute node before performing analysis.

Some useful initial checks on the output are:

- Is the calculation converged? See the plots of free energy change against total simulation time. Often, the bound vanish stage shows the poorest convergence
- Are there large discrepancies between runs? The overall 95 % confidence interval for the free energy change is typically around 1 kcal / mol for an intermediate-sized ligand in a reasonably behaved system with 5 replicates. If the uncertainty is much larger, identify which leg and stage it originates from by checking the free energy changes for each, and inspect the potential of mean force and histograms of the gradients to get an idea of which lambda windows are problematic. Inspecting the trajectories for these lambda windows is often helpful. Note that with different restraints, the results for the bound leg stages are not directly comparable (but the overall results for the leg should be the same), but checking for large discrepancies may still be informative.
- Are the free energy changes for the bound restraining stage (where the receptor-ligand restraints are introduced) reasonable? As a result of the restraint selection algorithm, these changes should all be around 1.2 kcal/ mol. If they are not, check the plots of the Boresch degrees of freedom in the ensemble equilibration direcoties. Discontinous jumps can indicate a change in binding modules

