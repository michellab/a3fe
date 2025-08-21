"""
This script updates attributes in A3FE pickle files to ensure they have the correct
types and values for `leg_type` and `stage_type`. It can handle both dictionary
representations and actual object instances. The script recursively processes nested
structures, converting string representations to enum types where necessary.

we can extend this script to modify/add more attributes to the objects
"""

import pickle
import glob
import os
import shutil
from typing import Any, Dict, List, Union

from a3fe.run.enums import LegType, StageType


def infer_leg_type_from_path(base_dir: str) -> Union[Any, str]:
    """Infer leg type from directory path and return the appropriate enum."""
    if not base_dir:
        return 'unknown'
    path_lower = str(base_dir).lower()
    
    if 'bound' in path_lower:
        return LegType.BOUND
    elif 'free' in path_lower:
        return LegType.FREE
    
    return 'unknown'


def infer_stage_type_from_path(base_dir: str) -> Union[Any, str]:
    """Infer stage type from directory path and return the appropriate enum."""
    if not base_dir:
        return 'unknown'
    path_lower = str(base_dir).lower()
    path_parts = path_lower.split('/')
    
    if 'discharge' in path_parts:
        return StageType.DISCHARGE
    elif 'vanish' in path_parts:
        return StageType.VANISH
    elif 'restrain' in path_parts:
        return StageType.RESTRAIN
    
    return 'unknown'


def string_to_leg_type(value: str) -> Union[Any, str]:
    """Convert string to LegType enum if possible."""
    if LegType is None:
        return value
    
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower == 'bound':
            return LegType.BOUND
        elif value_lower == 'free':
            return LegType.FREE
    
    return value


def string_to_stage_type(value: str) -> Union[Any, str]:
    """Convert string to StageType enum if possible."""
    if StageType is None:
        return value
    
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower == 'restrain':
            return StageType.RESTRAIN
        elif value_lower == 'discharge':
            return StageType.DISCHARGE
        elif value_lower == 'vanish':
            return StageType.VANISH
    
    return value


def identify_object_type(data: Union[Dict, Any]) -> str:
    """Identify what type of A3FE object this represents."""
    if isinstance(data, dict):
        # Look for characteristic attributes
        if 'lam' in data and 'run_no' in data:
            return 'Simulation'
        elif 'lam' in data and '_sub_sim_runners' in data:
            return 'LamWindow'
        elif 'lam' in data and 'sims' in data:
            return 'LamWindow'
        elif 'lam_vals' in data and ('stage_type' in data or 'lam_windows' in data or '_sub_sim_runners' in data):
            return 'Stage'
        elif ('leg_type' in data and 'stage_types' in data) or ('leg_type' in data and '_sub_sim_runners' in data):
            return 'Leg'
        elif 'setup_complete' in data and ('legs' in data or '_sub_sim_runners' in data):
            return 'Calculation'
    else:
        # It's an actual object instance
        return data.__class__.__name__
    
    return 'Unknown'


def migrate_data_structure(data: Union[Dict, Any], level: int = 0) -> bool:
    """
    Migrate a data structure (dict or object) and all nested structures.
    Returns True if any modifications were made.
    """
    total_modified = False
    indent = "  " * level
    
    # Handle dictionary representations
    if isinstance(data, dict):
        obj_type = identify_object_type(data)
        base_dir = data.get('base_dir', '')
        
        if obj_type != 'Unknown':
            print(f"{indent}Processing {obj_type} dict at {base_dir}")
            
            # Migrate this object
            if migrate_single_dict(data, obj_type):
                total_modified = True
        
        # Recursively process nested structures
        for key, value in data.items():
            if key == '_sub_sim_runners' and isinstance(value, list):
                for i, item in enumerate(value):
                    if migrate_data_structure(item, level + 1):
                        total_modified = True
            elif isinstance(value, (dict, list)):
                if migrate_data_structure(value, level + 1):
                    total_modified = True
    
    # Handle lists
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if migrate_data_structure(item, level + 1):
                total_modified = True
    
    # Handle object instances
    else:
        obj_type = data.__class__.__name__
        if obj_type in ['Calculation', 'Leg', 'Stage', 'LamWindow', 'Simulation']:
            print(f"{indent}Processing {obj_type} object")
            
            # For object instances, work with their __dict__
            if hasattr(data, '__dict__'):
                # First migrate the object's dict
                if migrate_data_structure(data.__dict__, level + 1):
                    total_modified = True
                
                # Then handle object-specific attribute migration
                if migrate_object_attributes(data):
                    total_modified = True
    
    return total_modified


def migrate_object_attributes(obj: Any) -> bool:
    """
    Migrate object-specific attributes that might be missing.
    Returns True if any modifications were made.
    """
    modified = False
    obj_type = obj.__class__.__name__
    
    if obj_type == 'LamWindow':
        # Check if LamWindow is missing leg_type or has string leg_type
        if not hasattr(obj, 'leg_type') or obj.leg_type is None:
            leg_type = infer_leg_type_from_path(getattr(obj, 'base_dir', ''))
            obj.leg_type = leg_type
            modified = True
            print(f"      Added leg_type='{leg_type}' to LamWindow object")
        elif isinstance(obj.leg_type, str):
            old_value = obj.leg_type
            obj.leg_type = string_to_leg_type(obj.leg_type)
            modified = True
            print(f"      Converted leg_type from string '{old_value}' to enum '{obj.leg_type}' in LamWindow object")
        
        # Check if LamWindow is missing stage_type or has string stage_type
        if not hasattr(obj, 'stage_type') or obj.stage_type is None:
            stage_type = infer_stage_type_from_path(getattr(obj, 'base_dir', ''))
            obj.stage_type = stage_type
            modified = True
            print(f"      Added stage_type='{stage_type}' to LamWindow object")
        elif isinstance(obj.stage_type, str):
            old_value = obj.stage_type
            obj.stage_type = string_to_stage_type(obj.stage_type)
            modified = True
            print(f"      Converted stage_type from string '{old_value}' to enum '{obj.stage_type}' in LamWindow object")
    
    elif obj_type == 'Simulation':
        # Check if Simulation is missing leg_type or has string leg_type
        if not hasattr(obj, 'leg_type') or obj.leg_type is None:
            leg_type = infer_leg_type_from_path(getattr(obj, 'base_dir', ''))
            obj.leg_type = leg_type
            modified = True
            print(f"      Added leg_type='{leg_type}' to Simulation object")
        elif isinstance(obj.leg_type, str):
            old_value = obj.leg_type
            obj.leg_type = string_to_leg_type(obj.leg_type)
            modified = True
            print(f"      Converted leg_type from string '{old_value}' to enum '{obj.leg_type}' in Simulation object")
        
        # Check if Simulation is missing stage_type or has string stage_type
        if not hasattr(obj, 'stage_type') or obj.stage_type is None:
            stage_type = infer_stage_type_from_path(getattr(obj, 'base_dir', ''))
            obj.stage_type = stage_type
            modified = True
            print(f"      Added stage_type='{stage_type}' to Simulation object")
        elif isinstance(obj.stage_type, str):
            old_value = obj.stage_type
            obj.stage_type = string_to_stage_type(obj.stage_type)
            modified = True
            print(f"      Converted stage_type from string '{old_value}' to enum '{obj.stage_type}' in Simulation object")
    
    elif obj_type == 'Stage':
        # For Stage objects, check if stage_type needs conversion to enum
        if hasattr(obj, 'stage_type') and isinstance(obj.stage_type, str):
            old_value = obj.stage_type
            obj.stage_type = string_to_stage_type(obj.stage_type)
            modified = True
            print(f"      Converted stage_type from string '{old_value}' to enum '{obj.stage_type}' in Stage object")
        
        # Check if leg_type is missing or needs conversion to enum
        if not hasattr(obj, 'leg_type') or obj.leg_type is None:
            leg_type = infer_leg_type_from_path(getattr(obj, 'base_dir', ''))
            obj.leg_type = leg_type
            modified = True
            print(f"      Added leg_type='{leg_type}' to Stage object")
        elif isinstance(obj.leg_type, str):
            old_value = obj.leg_type
            obj.leg_type = string_to_leg_type(obj.leg_type)
            modified = True
            print(f"      Converted leg_type from string '{old_value}' to enum '{obj.leg_type}' in Stage object")
    
    elif obj_type == 'Leg':
        # For Leg objects, check if leg_type needs conversion to enum
        if hasattr(obj, 'leg_type') and isinstance(obj.leg_type, str):
            old_value = obj.leg_type
            obj.leg_type = string_to_leg_type(obj.leg_type)
            modified = True
            print(f"      Converted leg_type from string '{old_value}' to enum '{obj.leg_type}' in Leg object")
    
    return modified


def migrate_single_dict(data_dict: Dict, obj_type: str) -> bool:
    """
    Migrate a single dictionary representation of an object.
    Returns True if modified.
    """
    modified = False
    base_dir = data_dict.get('base_dir', '')
    
    if obj_type == 'Stage':
        # Add leg_type if missing, or convert string to enum
        if 'leg_type' not in data_dict:
            leg_type = infer_leg_type_from_path(base_dir)
            data_dict['leg_type'] = leg_type
            modified = True
            print(f"      Added leg_type='{leg_type}' to Stage")
        elif isinstance(data_dict['leg_type'], str):
            old_value = data_dict['leg_type']
            data_dict['leg_type'] = string_to_leg_type(data_dict['leg_type'])
            modified = True
            print(f"      Converted leg_type from string '{old_value}' to enum '{data_dict['leg_type']}' in Stage")
        
        # Convert string stage_type to enum if needed
        if 'stage_type' in data_dict and isinstance(data_dict['stage_type'], str):
            old_value = data_dict['stage_type']
            data_dict['stage_type'] = string_to_stage_type(data_dict['stage_type'])
            modified = True
            print(f"      Converted stage_type from string '{old_value}' to enum '{data_dict['stage_type']}' in Stage")
    
    elif obj_type == 'LamWindow':
        # Add stage_type if missing, or convert string to enum
        if 'stage_type' not in data_dict:
            stage_type = infer_stage_type_from_path(base_dir)
            data_dict['stage_type'] = stage_type
            modified = True
            print(f"      Added stage_type='{stage_type}' to LamWindow")
        elif isinstance(data_dict['stage_type'], str):
            old_value = data_dict['stage_type']
            data_dict['stage_type'] = string_to_stage_type(data_dict['stage_type'])
            modified = True
            print(f"      Converted stage_type from string '{old_value}' to enum '{data_dict['stage_type']}' in LamWindow")
        
        # Add leg_type if missing, or convert string to enum
        if 'leg_type' not in data_dict:
            leg_type = infer_leg_type_from_path(base_dir)
            data_dict['leg_type'] = leg_type
            modified = True
            print(f"      Added leg_type='{leg_type}' to LamWindow")
        elif isinstance(data_dict['leg_type'], str):
            old_value = data_dict['leg_type']
            data_dict['leg_type'] = string_to_leg_type(data_dict['leg_type'])
            modified = True
            print(f"      Converted leg_type from string '{old_value}' to enum '{data_dict['leg_type']}' in LamWindow")
    
    elif obj_type == 'Simulation':
        # Add stage_type if missing, or convert string to enum
        if 'stage_type' not in data_dict:
            stage_type = infer_stage_type_from_path(base_dir)
            data_dict['stage_type'] = stage_type
            modified = True
            print(f"      Added stage_type='{stage_type}' to Simulation")
        elif isinstance(data_dict['stage_type'], str):
            old_value = data_dict['stage_type']
            data_dict['stage_type'] = string_to_stage_type(data_dict['stage_type'])
            modified = True
            print(f"      Converted stage_type from string '{old_value}' to enum '{data_dict['stage_type']}' in Simulation")
        
        # Add leg_type if missing, or convert string to enum
        if 'leg_type' not in data_dict:
            leg_type = infer_leg_type_from_path(base_dir)
            data_dict['leg_type'] = leg_type
            modified = True
            print(f"      Added leg_type='{leg_type}' to Simulation")
        elif isinstance(data_dict['leg_type'], str):
            old_value = data_dict['leg_type']
            data_dict['leg_type'] = string_to_leg_type(data_dict['leg_type'])
            modified = True
            print(f"      Converted leg_type from string '{old_value}' to enum '{data_dict['leg_type']}' in Simulation")
    
    elif obj_type == 'Leg':
        # Convert string leg_type to enum if needed
        if 'leg_type' in data_dict and isinstance(data_dict['leg_type'], str):
            old_value = data_dict['leg_type']
            data_dict['leg_type'] = string_to_leg_type(data_dict['leg_type'])
            modified = True
            print(f"      Converted leg_type from string '{old_value}' to enum '{data_dict['leg_type']}' in Leg")
    
    return modified


def migrate_pickle_file(pickle_path: str, dry_run: bool = False) -> bool:
    """Migrate a pickle file containing mixed data structures."""
    try:
        print(f"\nProcessing: {pickle_path}")
        
        # Load the pickle
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded data type: {type(data)}")
        if hasattr(data, '__class__'):
            print(f"Object class: {data.__class__.__name__}")
        
        # Migrate the data structure
        modified = migrate_data_structure(data)
        
        if modified and not dry_run:
            # Create backup
            backup_path = pickle_path + '.backup'
            if not os.path.exists(backup_path):
                shutil.copy2(pickle_path, backup_path)
                print(f"Created backup: {backup_path}")
            
            # Save the updated pickle
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Updated: {pickle_path}")
        
        elif modified and dry_run:
            print(f"[DRY RUN] Would update: {pickle_path}")
        
        else:
            print(f"No changes needed: {pickle_path}")
        
        return modified
    
    except Exception as e:
        print(f"✗ Error processing {pickle_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_pickle_files(base_dir: str) -> List[str]:
    """Find all A3FE pickle files."""
    patterns = [
        "**/Calculation.pkl",
        "**/Leg.pkl", 
        "**/Stage.pkl",
        "**/LamWindow.pkl",
        "**/Simulation.pkl"
    ]
    
    files = []
    for pattern in patterns:
        found = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        files.extend(found)
    
    # Remove duplicates and sort
    unique_files = list(set(files))
    unique_files.sort(key=lambda x: x.count('/'), reverse=True)
    
    return unique_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate A3FE dictionary structures in pickle files")
    parser.add_argument("target", help="Calculation directory or specific pickle file to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.target):
        # Migrate a specific file
        pickle_files = [args.target]
    elif os.path.isdir(args.target):
        # Migrate directory
        pickle_files = find_pickle_files(args.target)
    else:
        print(f"Error: {args.target} is neither a file nor directory")
        return 1
    
    if not pickle_files:
        print("No A3FE pickle files found")
        return 0
    
    print(f"Found {len(pickle_files)} pickle files")
    if args.dry_run:
        print("=== DRY RUN MODE ===")
    
    modified_count = 0
    for pickle_file in pickle_files:
        if migrate_pickle_file(pickle_file, args.dry_run):
            modified_count += 1
    
    print(f"\nSummary: {modified_count}/{len(pickle_files)} files modified")
    
    if not args.dry_run and modified_count > 0:
        print("Migration complete!")
    
    return 0



if __name__ == "__main__":
    exit(main())