Guides
=======

.. _preparing-input:

Preparing Input for a3fe
****************************
The most basic input that a3fe accepts is PDB files of the protein and crystallographic waters, along with an sdf
file for the ligand. Pre-parameterised inputs in AMBER-format are also accepted. The table below details the combinations
of input files that can be supplied to a3fe, and the names that they must be given (files for both the free and bound leg must
be provided). The preparation stage will be detected by a3fe when you instantiate a Calculation, and only the required preparation
steps will be carried out for each leg. 

You can also find out which input files are required for a given preparation stage for a given leg programmatically, e.g:

.. code-block:: python

    # Minimised parameterised structures for the free leg
    a3.PreparationStage.MINIMISED.get_simulation_input_files(a3.LegType.FREE)

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

**The parameterisation step is liable to failure if your pdb is not formatted for tleap**, becuase BioSimSpace is very fussy and will raise an error if tleap is not
completely happy with the input files.

.. tip::

    If you are having trouble parameterising your protein pdb, the recommended workflow is:

   * Clean your unsanitised pdb using pdb4amber, e.g. ``pdb4amber -i protein.pdb -o protein_sanitised.pdb``
   * Attempt to run a3fe again (renaming ``protein_sanitised.pdb`` to ``protein.pdb``)
   * If a3fe still raises an error, attempt parameterisation directly with ``tleap`` to get more detailed error messages. E.g., type ``tleap``, then

   .. code-block:: bash

      source leaprc.protein.ff14SB
      source leaprc.water.tip3p
      # Loading an unsanitised pdb will likely raise an error
      prot = loadpdb protein_sanitised.pdb
      saveamberparm prot protein.parm7 protein.rst7
      savepdb prot protein_fully_sanitised.pdb

   * If the above fails, this is often due to residue/ atom names which do not match the templates. Read the errors to find out which residues / atoms are causing the issues, then check the expected names in library which was loaded after typing ``source leaprc.protein.ff14SB`` e.g. ``cat $AMBERHOME/dat/leap/lib/amino12.lib``.  Rename the offending atoms/ residues and repeat the above step.
   * Finally, rename ``protein_fully_sanitised.pdb`` to ``protein.pdb`` and run a3fe again.

Alternatively, if you have a parameterised protein (protein.rst7 and protein.prm7), and ligand.sdf (and optionally waters.prm7 and waters.rst7), you can use the 
notebook supplied in a3fe/a3fe/data/example_run_dir/parameterise_and_assemble_input.ipynb to parameterise the ligand and create the parameterised input files required by a3fe.

Running Standard Non-Adaptive Calculations
*******************************************

Once you have the required files in `input` as described above, you can run a standard non-adaptive ABFE calculation. To run 5 replicates with 5 ns of sampling per window 
(discarding 1 ns of this to equilibration):

.. code-block:: python

    import a3fe as a3
    calc = a3.Calculation(ensemble_size = 5)
    calc.setup()
    calc.run(adaptive=False, runtime=5) # Run for 5 ns per window per replicate
    calc.wait() # Wait for the simulations to finish
    calc.set_equilibration_time(1) # Discard the first ns of simulation time
    calc.analyse() # Fast analyses
    calc.analyse_convergence() # Slower convergence analysis
    calc.save()

We suggest running this through ipython (so that you can interact with the calculation while it is running) in a tmux session (so that the process
is not killed when you log out).

Customising Calculations
*************************

Calculation setup options, including the force fields, lambda schedules, and length of the equilibration steps, can be customised using :class:`a3fe.run.system_prep.SystemPreparationConfig`.
For example, to use GAFF2 instead of OFF2 for the small molecule, set this in the config object and pass this to ``calc.setup()``:

.. code-block:: python

    config = a3.SystemPreparationConfig()
    cfg.forcefields["ligand"] = "gaff2"
    calc_set.setup(bound_leg_sysprep_config = cfg, free_leg_sysprep_config = cfg)

To customise the specifics of how each lambda window is run (e.g. timestep), you can use the ``set_simfile_option`` method. For example, to set the timestep to 2 fs, run
``calc.set_simfile_option("timestep", "2 * femtosecond")``. This will change parameters from the defaults given in ``template_config.cfg`` in the ``input`` directory, and warn
you if you are trying to set a parameter that is not present in the template config file. To see a list of available options, run ``somd-freenrg --help-config``. Note that if you
want to change any slurm options in ``run_somd.sh``, you should modify ``run_somd.sh`` in the the calculation ``input`` directory then run ``calc.update_run_somd()`` to update all
``run_somd.sh`` files in the calculation.

Running Fast Non-Adaptive Calculations
***************************************

By modifying the ``SystemPreparationConfig`` object as described above, we can now try running a very fast non-adaptive calculation with just
three replicates. Note that this is expected to produce an erroneously favourable free energy of binding.

.. code-block:: python

  import a3fe as a3
  # Shorten several of the initial equilibration stages.
  # This should still be stable.
  cfg = a3.SystemPreparationConfig()
  cfg.runtime_npt_unrestrained = 50 # ps
  cfg.runtime_npt = 50 # ps
  cfg.ensemble_equilibration_time = 100 # ps
  calc = a3.Calculation(ensemble_size = 3)
  calc_set.setup(bound_leg_sysprep_config = cfg, free_leg_sysprep_config = cfg)
  calc_set.run(adaptive = False, runtime=0.1) # ns
  calc.wait() # Wait for the simulations to finish
  calc.set_equilibration_time(1) # Discard the first ns of simulation time
  calc.analyse() # Fast analyses
  calc.save()

Running Adaptive Calculations
******************************

You can also take advantage of the adaptive algorithms available with a3fe. The code below uses the automated lambda window selection,
simulation time allocation, and equilibration detection algorithms.

.. code-block:: python

    import a3fe as a3
    calc = a3.Calculation(ensemble_size = 5)
    calc.setup()
    # Get optimised lambda schedule with thermodynamic speed
    # of 2 kcal mol-1
    calc.get_optimal_lam_vals(delta_er = 2)
    # Run adaptively with a runtime constant of 0.0005 kcal**2 mol-2 ns**-1
    # Note that automatic equilibration detection with the paired t-test 
    # method will also be carried out.
    calc.run(adaptive=True, runtime_constant = 0.0005)
    calc.wait()
    calc.analyse()
    calc.save()

.. note::

    It is recommended to run the ``calc.get_optimal_lam_vals()`` step before proceeding with ``calc.run(adaptive=True)``. This is becuase
    the relative simulation costs for the bound and free leg are determined during the optimisation step, and are used to calculate the simulation
    time allocations during the adaptive run. If you do not run the optimisation step, you can set the required attributes manually with e.g.
    ``calc.legs[0].recursively_set_attr("relative_simulation_cost", 1, force=True)`` and ``calc.legs[1].recursively_set_attr("relative_simulation_cost", 0.2, force=True)``

.. note::

    During the adaptive allocation of simulation time, the allocated runtime is computed taking into account the relative simulation cost. To obtain
    comparable total simulation times to those described in the manuscript, you should set the reference simulation time to the cost (hr / ns) of the bound leg of the
    MIF/ MIF180 complex ([input files here](https://github.com/michellab/Automated-ABFE-Paper/tree/main/simulations/initial_systems/mif/input)). The cost can be obtained 
    by running a short simulation for the leg and checking the cost with e.g. ``ref_cost = calc.legs[0].tot_gpu_time / calc.legs[0].tot_simtime``. This should then be passed when
    optimising the lambda schedule with e.g. ``calc.get_optimal_lam_vals(delta_er = 2, reference_sim_cost = ref_cost)``.

Analysis
********

Analysis can be performed with:

.. code-block:: python

    # Calculate the free energy changes using MBAR and 
    # generate a variety of plots to aid analysis.
    # Run through SLURM as MBAR can be computationally intensive.
    # Avoid costly RMSD analysis.
    calc.analyse(slurm=True, plot_rmsds=False)
    # Run longer analysis to check how the estimate is changing with
    # simulation time
    calc.analyse_convergence()

.. note::

    Your simulations must be equilibrated before analysis can be performed. Practically, this means that all lambda windows must be set as equilibrated.
    This will be done automatically by the adaptive equilibration detection algorithms, but can also be done manually with e.g. ``calc.set_equilibration_time(1)``.

``calc.analyse()`` generates a variety of outputs in the ``output`` directories of the calculation, leg, and stage directories. The most detailed
information is given in the stage output directories. You can get a detailed breakdown of the results as a pandas dataframe by running ``calc.get_results_df()``.

Convergence analysis involves repeatedly calculating the free energy changes with different subsets of the 
data, and is computationally intensive. Hence, it is implemented in a different function. To run convergence
analysis, enter ``calc.analyse_convergence()``. Plots of the free energy change against total simulation time
will be created in each output directory.

Some useful initial checks on the output are:

- Is the calculation equilibrated, or is the estimated free energy strongly dependent on the total simulation time? See the plots of free energy change against total simulation time. Often, the bound vanish stage shows the slowest equilibration
- Are there large discrepancies between runs? The overall 95 % confidence interval for the free energy change is typically around 1 kcal / mol for an intermediate-sized ligand in a reasonably behaved system with 5 replicates. If the uncertainty is much larger, identify which leg and stage it originates from by checking the free energy changes for each, and inspect the potential of mean force and histograms of the gradients to get an idea of which lambda windows are problematic. Inspecting the trajectories for these lambda windows is often helpful. Checking for Gelman-Rubin :math:`\hat{R} > 1.1` (indicative of substantial discrepancies between runs)(stage output directory) can also be informative.
- Are the free energy changes for the bound restraining stage (where the receptor-ligand restraints are introduced) reasonable? As a result of the restraint selection algorithm, these changes should all be around 1.2 kcal/ mol. If they are not, check the plots of the Boresch degrees of freedom in the ensemble equilibration direcoties. Discontinous jumps can indicate a change in binding modules

Running Sets of Calculations
*****************************

You can run sets of calculations using the :class:`a3fe.run.CalcSet` class. To do so:

- Create a directory containing subdirectories for each calculation, each containing the required input files as described above
- Create an ``input`` directory containing an ``exp_dgs.csv`` file with the experimental free energy changes formatted as below, where ``calc_base_dir`` is the name of the subdirectory containing the calculation input files, ``name`` is your desired name for the calculation, ``exp_dg`` is the experimental free energy change, ``exp_er`` is the experimental uncertainty, and ``calc_cor`` is a correction to be applied to the calculated free energy change (for example a symmetry correction).

.. code-block:: csv

    calc_base_dir,name,exp_dg,exp_er,calc_cor
    t4l,t4l,-9.06,0.5,0
    mdm2_pip2_short,mdm2_pip2_short,-2.93,0.5,0

.. note::

    If you do not have experimental data, you can either omit the ``exp_dgs.csv``, or supply it but leave the ``exp_dg`` and ``exp_er`` columns blank. The advantage of still supplying it is that you can still provide symmetry corrections in the ``calc_cor`` column. In both cases, you should set ``compare_to_exp = False`` in the ``calc_set.analyse()`` call.

- Create, run, and analyse the set of calculations, for example for a set of non-adaptive calculations for "t4l" and "mdm2_pip2_short":

.. code-block:: python

    import a3fe as a3
    calc_set = a3.CalcSet(calc_paths = ["t4l", "mdm2_pip2_short"])
    calc_set.setup()
    calc_set.run(adaptive=False, runtime=5)
    calc_set.wait()
    calc_set.set_equilibration_time(1)
    calc_set.analyse(exp_dgs_path = "input/exp_dgs.csv", offset = False)
    calc_set.save()

ABFE with Charged Ligands
*************************

Since A3FE 0.2.0, ABFE calculations with charged ligands are supported using a co-alchemical ion approach. The charge of the ligand will be automatically detected, assuming that this is correctly specified in the input sdf. The only change in the input required is that the use of PME, rather than reaction field electrostatics, should be specified in ``template_config.cfg`` as e.g.:

.. code-block:: bash

    ### Non-Bonded Interactions ###
    cutoff type = PME
    cutoff distance = 10 * angstrom

