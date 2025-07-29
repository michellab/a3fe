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

def find_all_simfiles(root_directory):
    """
    Find all simfile.dat files in the directory structure.
    
    Args:
        root_directory (str): Root directory to search
        
    Returns:
        list: List of paths to simfile.dat files
    """
    simfile_paths = []
    root_path = Path(root_directory)
    
    # Recursively find all simfile.dat files
    for simfile in root_path.rglob("simfile.dat"):
        simfile_paths.append(str(simfile))
    
    return sorted(simfile_paths)

def main(root_directory="."):
    """Main function to check all simfile.dat files."""
    
    if len(sys.argv) > 1:
        root_directory = sys.argv[1]
    
    print(f"Checking simfile.dat files in: {os.path.abspath(root_directory)}")
    print("=" * 60)
    
    # Find all simfile.dat files
    simfile_paths = find_all_simfiles(root_directory)
    
    if not simfile_paths:
        print("No simfile.dat files found!")
        return
    
    print(f"Found {len(simfile_paths)} simfile.dat files")
    print("-" * 60)
    
    # Check each file
    valid_files = []
    invalid_files = []
    
    for simfile_path in simfile_paths:
        is_valid, temperature, error_msg = check_simfile_temperature(simfile_path)
        
        # Get relative path for cleaner output
        rel_path = os.path.relpath(simfile_path, root_directory)
        
        if is_valid:
            valid_files.append((rel_path, temperature))
            print(f"✓ {rel_path:<60} | T = {temperature:.2f} K")
        else:
            invalid_files.append((rel_path, error_msg))
            print(f"✗ {rel_path:<60} | ERROR: {error_msg}")
    
    # Summary
    print("=" * 60)
    print(f"SUMMARY:")
    print(f"Total files checked: {len(simfile_paths)}")
    print(f"Valid files: {len(valid_files)}")
    print(f"Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print("\nINVALID FILES:")
        for file_path, error in invalid_files:
            print(f"  - {file_path}: {error}")
        sys.exit(1)  # Exit with error code if any files are invalid
    else:
        print("\n🎉 All simfile.dat files have valid temperatures!")
        
        # Optional: Show temperature statistics
        if valid_files:
            temperatures = [temp for _, temp in valid_files]
            print(f"\nTemperature Statistics:")
            print(f"  Min: {min(temperatures):.2f} K")
            print(f"  Max: {max(temperatures):.2f} K")
            print(f"  Avg: {sum(temperatures)/len(temperatures):.2f} K")

if __name__ == "__main__":
    main(root_directory=".")