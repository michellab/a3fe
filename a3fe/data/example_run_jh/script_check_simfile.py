#!/usr/bin/env python3
"""
Script to check if all simfile.dat files in the directory structure 
have valid temperature values (not None).
"""

import os
import sys
from pathlib import Path

def check_simfile_temperature(simfile_path):
    """
    Check if a simfile.dat file has a valid temperature value.
    
    Args:
        simfile_path (str): Path to the simfile.dat file
        
    Returns:
        tuple: (is_valid, temperature, error_message)
    """
    try:
        with open(simfile_path, "r") as ifile:
            lines = ifile.readlines()

        temp = None  # Temperature in K

        for line in lines:
            vals = line.split()
            # Get the temperature, checking the units
            if line.startswith("#Generating temperature is"):
                temp = vals[3]
                try:
                    unit = vals[4]
                except IndexError:
                    # Must be °C
                    temp, unit = temp.split("°")
                if unit == "C":
                    temp = float(temp) + 273.15  # Convert to K
                else:
                    temp = float(temp)
                break  # Found temperature, no need to continue

        if temp is None:
            return False, None, "Temperature not found in file"
        
        # Additional validation: check if temperature is reasonable
        if temp <= 0:
            return False, temp, f"Invalid temperature value: {temp} K"
            
        return True, temp, None
        
    except FileNotFoundError:
        return False, None, "File not found"
    except Exception as e:
        return False, None, f"Error reading file: {str(e)}"


def check_somd_cfg_file(cfg_path):
    try:
        file_size = os.path.getsize(cfg_path)
        
        if file_size == 0:
            return False, 0, "File is empty"
        
        return True, file_size, None
        
    except FileNotFoundError:
        return False, 0, "File not found"
    except Exception as e:
        return False, 0, f"Error accessing file: {str(e)}"
    
    
def find_simulation_directories(root_directory):
    """
    Find all directories containing either simfile.dat or somd.cfg files.
    
    Args:
        root_directory (str): Root directory to search
        
    Returns:
        dict: Dictionary with directory paths as keys and dict of found files as values
    """
    root_path = Path(root_directory)
    simulation_dirs = {}
    
    # Find all simfile.dat files
    for simfile in root_path.rglob("simfile.dat"):
        dir_path = str(simfile.parent)
        if dir_path not in simulation_dirs:
            simulation_dirs[dir_path] = {'simfile': None, 'somd_cfg': None}
        simulation_dirs[dir_path]['simfile'] = str(simfile)
    
    # Find all somd.cfg files
    for cfg_file in root_path.rglob("somd.cfg"):
        dir_path = str(cfg_file.parent)
        if dir_path not in simulation_dirs:
            simulation_dirs[dir_path] = {'simfile': None, 'somd_cfg': None}
        simulation_dirs[dir_path]['somd_cfg'] = str(cfg_file)
    
    return simulation_dirs


def find_run01_directories(root_directory):
    """
    Find all directories named 'run_01' containing simfile.dat or somd.cfg.
    """
    root_path = Path(root_directory)
    run01_dirs = {}
    for run_dir in root_path.rglob("run_01"):
        if run_dir.is_dir():
            simfile = run_dir / "simfile.dat"
            cfgfile = run_dir / "somd.cfg"
            # Only include if at least one of the files exists
            if simfile.exists() or cfgfile.exists():
                run01_dirs[str(run_dir)] = {
                    'simfile': str(simfile) if simfile.exists() else None,
                    'somd_cfg': str(cfgfile) if cfgfile.exists() else None
                }
    return run01_dirs


def format_file_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    elif size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/(1024*1024):.1f} MB"
    
def main(root_directory=".", run_01_only=True):
    """Main function to check all simfile.dat files."""
    
    if len(sys.argv) > 1:
        root_directory = sys.argv[1]
    
    print(f"Checking simulation files in: {os.path.abspath(root_directory)}")
    print("=" * 80)
    
    # Find all simfile.dat files
    if run_01_only:
        simulation_dirs = find_run01_directories(root_directory)
    else:
        simulation_dirs = find_simulation_directories(root_directory)
    
    if not simulation_dirs:
        print("No simulation files (simfile.dat or somd.cfg) found!")
        return
    
    print(f"Found {len(simulation_dirs)} directories with simulation files")
    print("-" * 80)
    
    # Track results
    valid_simfiles = []
    invalid_simfiles = []
    valid_cfg_files = []
    invalid_cfg_files = []
    missing_files = []
    
    # Check each directory
    for dir_path in sorted(simulation_dirs.keys()):
        files = simulation_dirs[dir_path]
        rel_dir = os.path.relpath(dir_path, root_directory)
        
        # Check simfile.dat
        if 'simfile' in files:
            sf_valid, sf_temp, sf_err = check_simfile_temperature(files['simfile'])
        else:
            sf_valid, sf_temp, sf_err = False, None, "simfile.dat MISSING"

        # Check somd.cfg
        if 'somd_cfg' in files:
            cfg_valid, cfg_size, cfg_err = check_somd_cfg_file(files['somd_cfg'])
        else:
            cfg_valid, cfg_size, cfg_err = False, None, "somd.cfg MISSING"

        # Skip if both are valid
        if sf_valid and cfg_valid:
            continue

        # Otherwise print errors
        print(f"\nDirectory with issues: {rel_dir}")
        print("  " + "-" * 60)

        if not sf_valid:
            print(f"  ✗ simfile.dat    | {sf_err}")
            invalid_simfiles.append((rel_dir, sf_err))
        if not cfg_valid:
            print(f"  ✗ somd.cfg       | {cfg_err}")
            invalid_cfg_files.append((rel_dir, cfg_err))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Total directories checked: {len(simulation_dirs)}")
    print(f"Valid simfile.dat files: {len(valid_simfiles)}")
    print(f"Invalid simfile.dat files: {len(invalid_simfiles)}")
    print(f"Valid somd.cfg files: {len(valid_cfg_files)}")
    print(f"Invalid somd.cfg files: {len(invalid_cfg_files)}")
    print(f"Missing files: {len(missing_files)}")


if __name__ == "__main__":
    main(root_directory=".")