a3fe Design
============

Software Design
****************

a3fe stores and manipulates simulations based on a heirarchy of "simulation runners". Each simulation runner
is responsible for manipulating a set of "sub simulation runners". For example, :class:`a3fe.Calculation` instances hold and
manipulate two :class:`a3fe.Leg` instances (bound and free), and :class:`a3fe.Leg` objects hold and manipulate :class:`a3fe.Stage` instances.
The :class:`a3fe.Stage` objects for the first leg can be accessed through some Calculation instance ``calc`` with ``calc.legs[0].stages`` (whether the leg is bound
or free can be quieried with ``calc.legs[0].leg_type``). All simulation runners are derived from the abstract base class :class:`a3fe.run._simulation_runner.SimulationRunner`, 
where as much as possible of the functionality is defined. As a result, all simulation runners have similar interfaces e.g. they all share the run() and kill() methods.
The heirarchy of simulation runners is:

- :class:`a3fe.CalcSet`
- :class:`a3fe.Calculation`
- :class:`a3fe.Leg`
- :class:`a3fe.Stage`
- :class:`a3fe.LamWindow`
- :class:`a3fe.Simulation`

Calling calc.kill(), for example, recursively kills all sub simulation runners and their sub simulation runners. You can recursively set
or get attributes for all sub simulation runners in the heirarchy with e.g. ``calc.recursively_set_attr("_equilibrated", True)`` or
``calc.recursively_get_attr("_equilibrated")``.

Each simulation runner logs to a file in its base directory named according to the class name (e.g. Calculation.log). In addition,
each simulation runner saves a pickled version of itself to a file named in the same way (e.g. Calculation.pkl), which
allows them to be restarted at any point. The pickle files are automatically detected and used to load the Simulation
runners when they are present in the base directory. For example, running ``calc = a3.Calculation()`` in the base directory of
an pre-prepared calculation will load the previous calculation, overwriting any arguments supplied to Calculation().
The current state of a simulation runner can be written to the pickle file with the save() method, e.g. ``calc.save()``.

Algorithms
***********

a3fe aims to run ABFE as efficiently as possible, while generating robust estimates of uncertainty. A user-specified number of 
replicate simulations (this is specified when the simulation runner is created, e.g. ``calc = a3.Calculation(ensemble_size=5)``)
are run (the default is 5). Running several replicates in parallel allows:

- A reasonablly robust estimate of the uncertainty from the inter-run differences
- More robust ensemble-based adaptive equilibration detection

a3fe implements agorithms for:

- Automatic determination of the optimal lambda spacing (e.g. ``calc.get_optimal_lam_vals()``)
- Adaptive allocation of simulation time to minimise the inter-run uncertainty (e.g. ``calc.run(adaptive=True)``)
- Adaptive equilibration detection (see :func:`a3fe.analyse.detect_equil.check_equil_multiwindow_paired_t`, used when ``calc.run(adaptive=True)`` is specified)

For more details of the algorithms, please see [the preprint](https://doi.org/10.26434/chemrxiv-2024-3ft7f).

Some Notes on the Implementation
*********************************

a3fe is designed to be easily adaptable to any SLURM cluster. The SLURM submission settings can be tailored by modifying 
the :class:`a3fe.SlurmConfig` of your calculation (or other simulation runner). For example, to change the partition:

.. code-block:: python

    calc.slurm_config.partition = "my-cluster-gpu-partition"

If you don't supply a partition to the SlurmConfig, a3fe will use the default partition.

If the input is not parameterised, a3fe will parameterise your input with ff14SB, OFF 2.0.0, and TIP3P by default. See 
:ref:`preparing input<preparing-input>`. By default, a3fe will solvate your system in a rhombic dodecahedral box with 150 mM NaCl
and perform a standard minimisation, heating, and pre-equilibration routine.

At present, a3fe uses GROMACS to run all set-up jobs, so please ensure that you have loaded the required CUDA and
GROMACS modules, or sourced GMXRC. These GROMACS jobs are also submitted through SLURM, and a unique 5 ns "ensemble
equilibration" simulation is run for each of the ``ensemble_size`` repeats. For the bound leg, these are used to extract
different Boresch restraints for each replicate simulation using the in-built BioSimSpace algorithm (see
`the BioSimSpace restraint selection code <https://github.com/fjclark/BioSimSpace/blob/01dba53b01386a3851e277874f9080c316c4632e/python/BioSimSpace/Sandpit/Exscientia/FreeEnergy/_restraint_search.py#L902>`_).
This fits force constants of the Boresch restraints according to the fluctuations observed during the fitting simulations, and scores candidate restraints accorinding 
to how severly they restrict the configurational space accessible to the ligand (more restriction is better as it indicates that the restraints are mimicking a 
stronger native interaction).

a3fe can use a default spacing of lambda windows which should work reasonably for most systems with the default SOMD
settings. However, to optimise the lambda schedule by running short (100 ps default) simulations and generating a new spacing
according to the integrated variance of the gradients, run ``calc.get_optimal_lam_vals()``.

One weakness of a3fe is that the molecular dynamics engine used for production simulations (SOMD) does not support enhanced sampling; HREX is not available. However,
this does mean that all individual SOMD simulations can be run in parallel. 

Units
******

+-------------------+----------+
| Quantity          | Unit     |
+===================+==========+
| Simulation Time   | ns       |
+-------------------+----------+
| Computer Time     | hr       |
+-------------------+----------+
| Energy            | kcal/mol |
+-------------------+----------+

Note that when specifying the run-time of a calculation, this is per-window, per-replicate. For example, if you specify
``calc.run(adaptive=False, runtime=1)`` and ``calc.ensemble_size==5``, then the total run-time for each window will be 5 ns. However,
when you query the total simulation time with ``calc.tot_simtime``, this is the cumulative total for every simulation in the calculation.
