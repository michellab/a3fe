Getting Started
===============
a3fe is a package for running alchemical absolute binding free energy calculations with SOMD (Sire / OpenMM Molecular Dynamics) through SLURM. 
It is based on Sire(https://sire.openbiosim.org/) and also uses BioSimSpace(https://biosimspace.openbiosim.org/) during the set-up stages. For a
discussion of the algorithms used, please see [the preprint](https://doi.org/10.26434/chemrxiv-2024-3ft7f).

Installation
************
Please see the instructions in the github repo README (https://github.com/michellab/a3fe).

Quick Start
***********
- Activate your a3fe conda environment 
- Create a base directory for the calculation and create an directory called ``input`` within this
- Move your input files into the the input directory. For example, if you have parameterised AMBER-format input files, name these bound_param.rst7, bound_param.prm7, free_param.rst7, and free_param.prm7. **Ensure that the ligand is named LIG and is the first molecule in the system.** For more details see :ref:`Preparing Input for a3fe`. Alternatively, copy the pre-provided input from ``a3fe/a3fe/data/example_run_dir/input`` to your input directory.
- In the calculation base directory, run the following python code, either through ipython or as a python script (you will likely want to run this with ``nohup``/ through tmux to ensure that the calculation is not killed when you lose connection). Running though ipython will let you interact with the calculation while it's running.

.. code-block:: python

    import a3fe as a3 
    calc = a3.Calculation(
        ensemble_size=5, # Use 5 (independently equilibrated) replicate runs
        slurm_config=a3.SlurmConfig(partition="<desired partition>"),  # Set your desired partition!
        engine_config=a3.SomdConfig(),
    )
    calc.setup()
    calc.get_optimal_lam_vals()
    calc.run(adaptive=False, runtime = 5) # Run non-adaptively for 5 ns per replicate
    calc.wait()
    calc.set_equilibration_time(1) # Discard the first ns of simulation time
    calc.analyse()
    calc.save()

- Check the results in the ``output`` directories (separate output directories are created for the Calculation, Legs, and Stages)

Quick Tips
***********

Some handy commands and code snippets, assuming that you have set up the calculation following the code above:

**Terminate all the SLURM jobs**: ``calc.kill()``

**Delete all the output** (but not input files), ready to run again: ``calc.clean()``

**Delete the large trajectory and restart files** ``calc.lighten()``

**Return any failed simulations** and find out where to look for output:

.. code-block:: python

    for failed_sim in calc.failed_simulations:
        print(failed_sim.base_dir)

**Check how many GPU hours your calculation has cost**:

.. code-block:: python

    print(f"Total GPU hours: {calc.tot_gpu_time:0.f}")
    print("#"*10)
    for leg in calc.legs:
        print(f"Total GPU hours for leg {leg.leg_type}: {leg.tot_gpu_time:.0f}")

**Get a detailed summary of free energy components as a pandas dataframe**:

.. code-block:: python

    df = calc.get_summary_df()
    print(df)

**Save the current state of the calculation**:

.. code-block:: python

    calc.save()
