"""
This script provides functions to update attributes of a leg object stored in a pickle file.

It's messy, but I need to constantly move and run simulations on different machines (HPC and local) 
so I need to update the paths in the object. 

We may also use the following code (not tested):

    old_base = "/project/6097686/jjhuang/fep_workflows/run_in_one_node1"
    new_base = "/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2"
    
    calc.update_paths(old_base, new_base)
"""
import pickle
import os
from pathlib import Path

def update_leg_object(pickle_path, output_path=None, **kwargs):
    """
    Load a leg object from pickle, update specified attributes, and save it back.
    
    Args:
        pickle_path (str): Path to the input pickle file
        output_path (str, optional): Path to save updated pickle. If None, overwrites original.
        **kwargs: Keyword arguments for attributes to update
        
    Returns:
        dict: The updated leg object
        
    Example:
        # Update base_dir and ensemble_size
        updated_leg = update_leg_object(
            "/path/to/Leg.pkl",
            base_dir="/new/base/directory",
            ensemble_size=5,
            runtime_constant=0.002
        )
        
        # Save to different location
        updated_leg = update_leg_object(
            "/path/to/Leg.pkl",
            output_path="/path/to/Updated_Leg.pkl",
            base_dir="/new/base/directory"
        )
    """
    print(f"Loading leg object from: {pickle_path}")
    with open(pickle_path, "rb") as f:
        leg = pickle.load(f)
    
    print(f"Original leg object loaded with {len(leg)} attributes")
    
    updates_made = []
    
    # Update attributes if provided
    for attr_name, new_value in kwargs.items():
        if attr_name in leg:
            old_value = leg[attr_name]
            leg[attr_name] = new_value
            updates_made.append((attr_name, old_value, new_value))
            print(f"Updated '{attr_name}': {old_value} -> {new_value}")
        else:
            # Add new attribute if it doesn't exist
            leg[attr_name] = new_value
            updates_made.append((attr_name, "NEW", new_value))
            print(f"Added new attribute '{attr_name}': {new_value}")
    
    # Determine output path
    if output_path is None:
        output_path = pickle_path
        print(f"Will overwrite original file: {output_path}")
    else:
        print(f"Will save to new file: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(leg, f)
    
    print(f"\nSummary:")
    print(f"- Total attributes in leg object: {len(leg)}")
    print(f"- Attributes updated/added: {len(updates_made)}")
    print(f"- Saved to: {output_path}")
    
    if updates_made:
        print(f"\nChanges made:")
        for attr, old_val, new_val in updates_made:
            if old_val == "NEW":
                print(f"  + {attr}: {new_val}")
            else:
                print(f"  ~ {attr}: {old_val} -> {new_val}")
    
    return leg

def inspect_leg_object(pickle_path):
    """
    Load and display all attributes of a leg object without modifying it.
    
    Args:
        pickle_path (str): Path to the pickle file
        
    Returns:
        dict: The leg object
    """
    
    with open(pickle_path, "rb") as f:
        leg = pickle.load(f)
    
    print(f"Leg object inspection for: {pickle_path}")
    print("=" * 60)
    print(f"Total attributes: {len(leg)}")
    print("-" * 60)
    
    for key, value in leg.items():
        # Truncate long values for display
        if isinstance(value, (list, dict)) and len(str(value)) > 100:
            value_str = f"{type(value).__name__} with {len(value)} items"
        elif len(str(value)) > 80:
            value_str = str(value)[:77] + "..."
        else:
            value_str = str(value)
        
        print(f"{key:<25}: {value_str}")
    
    return leg


def update_all_paths_leg(pickle_path, old_base, new_base):
    """Update all path-related attributes in the leg object AND its stages."""
    
    with open(pickle_path, "rb") as f:
        leg = pickle.load(f)
    
    # Update leg direct path attributes
    if 'base_dir' in leg:
        leg['base_dir'] = leg['base_dir'].replace(old_base, new_base)
    
    if '_input_dir' in leg:
        leg['_input_dir'] = leg['_input_dir'].replace(old_base, new_base)
    
    if '_output_dir' in leg:
        leg['_output_dir'] = leg['_output_dir'].replace(old_base, new_base)
    
    # Update stage_input_dirs
    if 'stage_input_dirs' in leg:
        new_stage_dirs = {}
        for stage_type, old_path in leg['stage_input_dirs'].items():
            new_stage_dirs[stage_type] = old_path.replace(old_base, new_base)
        leg['stage_input_dirs'] = new_stage_dirs
    
    # Update virtual_queue.log_dir
    if 'virtual_queue' in leg:
        leg['virtual_queue'].log_dir = leg['virtual_queue'].log_dir.replace(old_base, new_base)
    
    # THIS IS THE KEY PART: Update paths in all Stage objects
    if '_sub_sim_runners' in leg:
        for stage in leg['_sub_sim_runners']:
            # Update Stage paths
            stage.base_dir = stage.base_dir.replace(old_base, new_base)
            stage._input_dir = stage._input_dir.replace(old_base, new_base)  
            stage._output_dir = stage._output_dir.replace(old_base, new_base)
            
            # Update Stage's virtual queue
            if hasattr(stage, 'virtual_queue'):
                stage.virtual_queue.log_dir = stage.virtual_queue.log_dir.replace(old_base, new_base)
            
            # Update any lambda windows in the stage
            if hasattr(stage, '_sub_sim_runners'):
                for lam_window in stage._sub_sim_runners:
                    lam_window.base_dir = lam_window.base_dir.replace(old_base, new_base)
                    lam_window._input_dir = lam_window._input_dir.replace(old_base, new_base)
                    lam_window._output_dir = lam_window._output_dir.replace(old_base, new_base)
                    
                    # Update simulations within lambda windows
                    if hasattr(lam_window, '_sub_sim_runners'):
                        for sim in lam_window._sub_sim_runners:
                            sim.base_dir = sim.base_dir.replace(old_base, new_base)
                            sim._input_dir = sim._input_dir.replace(old_base, new_base)
                            sim._output_dir = sim._output_dir.replace(old_base, new_base)
                            if hasattr(sim, 'slurm_file_base') and sim.slurm_file_base:
                                sim.slurm_file_base = sim.slurm_file_base.replace(old_base, new_base)
    
    with open(pickle_path, "wb") as f:
        pickle.dump(leg, f)
    
    print(f"Updated all paths from '{old_base}' to '{new_base}' including all nested objects")
    return leg


def update_all_paths_stage(pickle_path, old_base, new_base):
    """Update all path-related attributes in the stage object."""
    
    with open(pickle_path, "rb") as f:
        stage = pickle.load(f)
    
    # Update direct path attributes
    if 'base_dir' in stage:
        stage['base_dir'] = stage['base_dir'].replace(old_base, new_base)
    
    if '_input_dir' in stage:
        stage['_input_dir'] = stage['_input_dir'].replace(old_base, new_base)
    
    if '_output_dir' in stage:
        stage['_output_dir'] = stage['_output_dir'].replace(old_base, new_base)
    
    
    # Update virtual_queue.log_dir
    stage['virtual_queue'].log_dir = stage['virtual_queue'].log_dir.replace(old_base, new_base)


    # THIS IS THE KEY PART: Update paths in all LamWindow objects
    if '_sub_sim_runners' in stage:
        for lamwin in stage['_sub_sim_runners']:
            # Update Stage paths
            lamwin.base_dir = lamwin.base_dir.replace(old_base, new_base)
            lamwin._input_dir = lamwin._input_dir.replace(old_base, new_base)  
            lamwin._output_dir = lamwin._output_dir.replace(old_base, new_base)
            
            # Update Stage's virtual queue
            if hasattr(lamwin, 'virtual_queue'):
                lamwin.virtual_queue.log_dir = lamwin.virtual_queue.log_dir.replace(old_base, new_base)
            
            # Update any lambda windows in the stage
            # note that for lambda windows, _sub_sim_runners contains Simulation objects
            if hasattr(lamwin, '_sub_sim_runners'):
                for sim in lamwin._sub_sim_runners:
                    sim.base_dir = sim.base_dir.replace(old_base, new_base)
                    sim._input_dir = sim._input_dir.replace(old_base, new_base)
                    sim._output_dir = sim._output_dir.replace(old_base, new_base)
                    
                    # Update simulations within lambda windows
                    if hasattr(sim, '_sub_sim_runners'):
                        for sim_run in sim._sub_sim_runners:
                            sim_run.base_dir = sim_run.base_dir.replace(old_base, new_base)
                            sim_run._input_dir = sim_run._input_dir.replace(old_base, new_base)
                            sim_run._output_dir = sim_run._output_dir.replace(old_base, new_base)
                            if hasattr(sim, 'slurm_file_base') and sim.slurm_file_base:
                                sim.slurm_file_base = sim.slurm_file_base.replace(old_base, new_base)
    
    with open(pickle_path, "wb") as f:
        pickle.dump(stage, f)
    
    print(f"Updated all paths from '{old_base}' to '{new_base}'")
    return stage


# Example usage functions
if __name__ == "__main__":
    # Example 1: Update a single leg object
    from a3fe.run.enums import StageType
    # new_stage_dirs = {
    # StageType.RESTRAIN: '/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2/bound/restrain/input',
    # StageType.DISCHARGE: '/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2/bound/discharge/input', 
    # StageType.VANISH: '/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2/bound/vanish/input'
    # }
    updated_leg = update_all_paths_leg(
        pickle_path="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2/bound/Leg.pkl",
        old_base="/project/6097686/jjhuang/fep_workflows/run_in_one_node1",
        new_base="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2",
    )

    for stage_name in ['restrain', 'discharge', 'vanish']:
        stage_path = f"/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2/bound/{stage_name}/Stage.pkl"
        updated_stage = update_all_paths_stage(
            stage_path,
            old_base="/project/6097686/jjhuang/fep_workflows/run_in_one_node1",
            new_base="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2"
        )

    print("Leg updater functions loaded. Use the functions above to update your leg objects.")