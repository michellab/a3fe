## Common errors at runtime
* [ERROR 1] We need to set `cutoff type = PME` for charged molecules. otherwise, we get the error below. 
    ```
    ValueError: The ligand has a non-zero charge (2), so SOMD must use PME for the 
    electrostatics. Please set the 'cutoff type' option in the somd.cfg file to 
    'PME'.
    ```
* [ERROR 2] we need to firt check the protein-ligand system to see if they look right. This error is likely due to 1. ligand is too flexible or mobile, 2. insufficient sampling in the ensemble equilibraiton, 3. poor ligand-protein contacts in the binding pose, 4. maybe the restraint method is too strict? try using mdanalysis-based method? 
    ```
    INFO - 2025-08-15 23:57:14,823 - Leg (type = BOUND)_1 - Loading trajectory for run 2...
    INFO - 2025-08-15 23:57:23,586 - Leg (type = BOUND)_1 - Selecting restraints for run 2...
    INFO: The attribute(s) types, masses have already been read from the topology file. The guesser will only guess empty values for this attribute, if any exists. To overwrite it by completely guessed values, you can pass the attribute to the force_guess parameter instead of the to_guess one
    INFO: There is no empty types values. Guesser did not guess any new values for types attribute
    INFO: There is no empty masses values. Guesser did not guess any new values for masses attribute
    Searching for low variance pairs. Frame no: 100%|██████████| 2501/2501 [00:41<00:00, 59.92it/s]
    Scoring candidate Boresch anchor points. Anchor set no: 100%|██████████| 50/50 [03:50<00:00,  4.62s/it]
    ╭───────────────────── Traceback (most recent call last) ──────────────────────╮
    │ /scratch/jjhuang/jjhuang/fep_workflows/system_12/run_calc.py:1542 in         │
    │ <module>                                                                     │
    │                                                                              │
    │   1539 │   calc = a3.Calculation(base_dir="/home/jjhuang/scratch/jjhuang/fep │
    │   1540 │   │   │   │   │   │    input_dir="/home/jjhuang/scratch/jjhuang/fep │
    │   1541 │                                                                     │
    │ ❱ 1542 │   calc.setup(                                                       │
    │   1543 │   │   bound_leg_sysprep_config=sysprep_cfg,                         │
    │   1544 │   │   free_leg_sysprep_config=sysprep_cfg,                          │
    │   1545 │   )                                                                 │
    │                                                                              │
    │ /project/6097686/jjhuang/fep_workflows/a3fe_jh/a3fe/run/calculation.py:221   │
    │ in setup                                                                     │
    │                                                                              │
    │   218 │   │   │   │   slurm_configs=self.slurm_configs,                      │
    │   219 │   │   │   )                                                          │
    │   220 │   │   │   self.legs.append(leg)                                      │
    │ ❱ 221 │   │   │   leg.setup(configs[leg_type], skip_preparation=skip_prepara │
    │   222 │   │                                                                  │
    │   223 │   │   # Save the state                                               │
    │   224 │   │   self.setup_complete = True                                     │
    │                                                                              │
    │ /project/6097686/jjhuang/fep_workflows/a3fe_jh/a3fe/run/leg.py:257 in setup  │
    │                                                                              │
    │    254 │   │   │   │   # Run separate equilibration simulations for each of  │
    │    255 │   │   │   │   # extract the final structures to give a diverse ense │
    │    256 │   │   │   │   # conformations. For the bound leg, this also extract │
    │ ❱  257 │   │   │   │   system = self.run_ensemble_equilibration(sysprep_conf │
    │    258 │   │   │   │   # note that in run_ensemble_equilibration(), we also  │
    │    259 │   │   else:                                                         │
    │    260 │   │   │   # TODO: this part is USELESS because we cannot dump and l │
    │                                                                              │
    │ /project/6097686/jjhuang/fep_workflows/a3fe_jh/a3fe/run/leg.py:834 in        │
    │ run_ensemble_equilibration                                                   │
    │                                                                              │
    │    831 │   │   │   │   │   system=pre_equilibrated_system,                   │
    │    832 │   │   │   │   )                                                     │
    │    833 │   │   │   │   self._logger.info(f"Selecting restraints for run {i + │
    │ ❱  834 │   │   │   │   restraint = _BSS.FreeEnergy.RestraintSearch.analyse(  │
    │    835 │   │   │   │   │   method="BSS",                                     │
    │    836 │   │   │   │   │   system=pre_equilibrated_system,                   │
    │    837 │   │   │   │   │   traj=traj,                                        │
    │                                                                              │
    │ /home/jjhuang/miniconda3/envs/a3fe_gra/lib/python3.12/site-packages/BioSimSp │
    │ ace/Sandpit/Exscientia/FreeEnergy/_restraint_search.py:696 in analyse        │
    │                                                                              │
    │    693 │   │   │   ligand_selection_str += append_to_ligand_selection        │
    │    694 │   │                                                                 │
    │    695 │   │   if restraint_type.lower() == "boresch":                       │
    │ ❱  696 │   │   │   return RestraintSearch._boresch_restraint(                │
    │    697 │   │   │   │   u,                                                    │
    │    698 │   │   │   │   system,                                               │
    │    699 │   │   │   │   temperature,                                          │
    │                                                                              │
    │ /home/jjhuang/miniconda3/envs/a3fe_gra/lib/python3.12/site-packages/BioSimSp │
    │ ace/Sandpit/Exscientia/FreeEnergy/_restraint_search.py:817 in                │
    │ _boresch_restraint                                                           │
    │                                                                              │
    │    814 │   │   │   │   )                                                     │
    │    815 │   │                                                                 │
    │    816 │   │   elif method == "BSS":                                         │
    │ ❱  817 │   │   │   return RestraintSearch._boresch_restraint_BSS(            │
    │    818 │   │   │   │   u,                                                    │
    │    819 │   │   │   │   system,                                               │
    │    820 │   │   │   │   temperature,                                          │
    │                                                                              │
    │ /home/jjhuang/miniconda3/envs/a3fe_gra/lib/python3.12/site-packages/BioSimSp │
    │ ace/Sandpit/Exscientia/FreeEnergy/_restraint_search.py:1778 in               │
    │ _boresch_restraint_BSS                                                       │
    │                                                                              │
    │   1775 │   │   )                                                             │
    │   1776 │   │                                                                 │
    │   1777 │   │   # Convert to Boresch anchors, order by correction, and filter │
    │ ❱ 1778 │   │   pairs_ordered_boresch, boresch_dof_data = _findOrderedBoresch │
    │   1779 │   │   │   u,                                                        │
    │   1780 │   │   │   ligand_selection_str,                                     │
    │   1781 │   │   │   receptor_selection_str,                                   │
    │                                                                              │
    │ /home/jjhuang/miniconda3/envs/a3fe_gra/lib/python3.12/site-packages/BioSimSp │
    │ ace/Sandpit/Exscientia/FreeEnergy/_restraint_search.py:1599 in               │
    │ _findOrderedBoresch                                                          │
    │                                                                              │
    │   1596 │   │   │   │   │   pairs_ordered_boresch.append(pair)                │
    │   1597 │   │   │                                                             │
    │   1598 │   │   │   if len(pairs_ordered_boresch) == 0:                       │
    │ ❱ 1599 │   │   │   │   raise _AnalysisError(                                 │
    │   1600 │   │   │   │   │   "No candidate sets of Boresch restraints are suit │
    │   1601 │   │   │   │   │   "search criteria or increase force constants."    │
    │   1602 │   │   │   │   )                                                     │
    ╰──────────────────────────────────────────────────────────────────────────────╯
    AnalysisError: No candidate sets of Boresch restraints are suitable. Please 
    expand search criteria or increase force constants.
    ```
* [ERROR 3] The most tricky issue is about convergence. For some lambda in some system, the predicted runtime increases as 
the actual runtime increases. so why?
  *   I took a look of one such lambda (I will check more later) and I found that 
  ```
    ================================================================================
    FINAL SUMMARY: Predicted Maximum Efficiency Runtime vs Time
    ================================================================================
    Time/Rep(ns) Total(ns)  Actual(ns)  Predicted(ns)  Inter-run_SEM  Norm SEM      
    --------------------------------------------------------------------------------
    0.2          1.0        1.000       3.692          1.889          0.152046    
    0.4          2.0        2.000       5.721          2.069          0.235568    
    0.8          4.0        4.000       8.187          2.094          0.337114    
    1.6          8.0        8.000       11.407         2.063          0.469720 
  ```
  In a3fe, `predicted runtime` by default is computed as following:
  ```
    total_times_all_runs = np.array(self.total_times) * len(self.run_nos)  
    sems_inter_delta_g *= np.sqrt(total_times_all_runs) -> this is the normalised SEM

    predicted_run_time_max_eff = (1 / np.sqrt(runtime_constant * relative_simulation_cost)) * normalised_sem_dg
    # for this specific lambda,  predicted_run_time_max_eff ~ 24.28 * normalised_sem_dg
  ```
  as indicated by the table, when extending the simulation, `normalised SEM` is increasing: 0.152 → 0.235 → 0.337 → 0.470. 
  if 
