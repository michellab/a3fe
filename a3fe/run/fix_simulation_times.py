import os
import numpy as np
import subprocess
from pathlib import Path

def truncate_simulations_to_minimum(calc):
    """
    Truncate all simulations to the minimum runtime for each lambda window.
    This ensures all repeats have the same simulation length for proper MBAR analysis.
    
    Parameters
    ----------
    calc : a3fe.Calculation
        The calculation object with potentially inconsistent simulation times
    """
    
    print("=== TRUNCATING SIMULATIONS TO MINIMUM RUNTIME ===\n")
    
    for leg in calc.legs:
        print(f"=== {leg.leg_type.name} LEG ===\n")
        
        for stage in leg.stages:
            print(f"--- {stage.stage_type.name} STAGE ---\n")
            stage_has_issues = False
            
            for lam_window in stage.lam_windows:
                # Get simulation times for all runs
                sim_times = []
                for sim in lam_window.sims:
                    sim_time = sim.get_tot_simtime()
                    sim_times.append(sim_time)
                
                # Check if all times are consistent (within 0.01 ns tolerance)
                min_time = min(sim_times)
                max_time = max(sim_times)
                
                if abs(max_time - min_time) > 0.01:  # More than 0.01 ns difference
                    stage_has_issues = True
                    print(f"Lambda {lam_window.lam:.3f}: Inconsistent times {sim_times}")
                    print(f"  -> Truncating to minimum: {min_time:.6f} ns")
                    
                    # Truncate each simulation to the minimum time
                    for i, sim in enumerate(lam_window.sims):
                        if sim_times[i] > min_time:
                            truncate_simulation_file(sim, min_time)
                            print(f"     Truncated run {sim.run_no}: {sim_times[i]:.6f} -> {min_time:.6f} ns")
                    
                    print()
                else:
                    print(f"Lambda {lam_window.lam:.3f}: ✓ All runs consistent at {min_time:.6f} ns")
            
            if not stage_has_issues:
                print(f"  ✓ Stage {stage.stage_type.name} has no timing issues")
            print()
    
    print("=== TRUNCATION COMPLETE ===")


def truncate_simulation_file(simulation, target_time_ns):
    """
    Truncate a simulation file to a specific time.
    
    Parameters
    ----------
    simulation : a3fe.Simulation
        The simulation object to truncate
    target_time_ns : float
        Target simulation time in nanoseconds
    """
    
    simfile_path = os.path.join(simulation.output_dir, "simfile.dat")
    
    if not os.path.exists(simfile_path):
        print(f"Warning: {simfile_path} does not exist, skipping truncation")
        return
    
    # Calculate the target step number
    # target_step = target_time_ns / simulation.timestep
    # We need to be more careful about this calculation
    
    # Read the simulation file to understand the step intervals
    with open(simfile_path, 'r') as f:
        lines = f.readlines()
    
    # Find data lines (non-comment lines)
    data_lines = []
    header_lines = []
    
    for line in lines:
        if line.startswith('#'):
            header_lines.append(line)
        else:
            try:
                step = int(line.split()[0])
                time_ns = step * simulation.timestep
                if time_ns <= target_time_ns + 1e-10:  # Small tolerance for floating point
                    data_lines.append(line)
                else:
                    break  # Stop when we exceed target time
            except (ValueError, IndexError):
                # Skip malformed lines
                continue
    
    if not data_lines:
        print(f"Warning: No valid data found in {simfile_path}")
        return
    
    # Create backup
    backup_path = simfile_path + ".backup"
    if not os.path.exists(backup_path):
        subprocess.run(['cp', simfile_path, backup_path])
    
    # Write truncated file
    with open(simfile_path, 'w') as f:
        # Write header
        for line in header_lines:
            f.write(line)
        
        # Write truncated data
        for line in data_lines:
            f.write(line)
    
    # Verify the truncation worked
    last_step = int(data_lines[-1].split()[0])
    actual_time = last_step * simulation.timestep
    
    print(f"     Truncated to step {last_step}, actual time: {actual_time:.6f} ns")


def verify_truncation(calc):
    """
    Verify that truncation worked correctly by checking all simulation times again.
    
    Parameters
    ----------
    calc : a3fe.Calculation
        The calculation object to verify
    """
    
    print("\n=== VERIFYING TRUNCATION ===\n")
    
    all_consistent = True
    
    for leg in calc.legs:
        print(f"=== {leg.leg_type.name} LEG ===")
        
        for stage in leg.stages:
            print(f"--- {stage.stage_type.name} STAGE ---")
            
            for lam_window in stage.lam_windows:
                sim_times = []
                for sim in lam_window.sims:
                    sim_time = sim.get_tot_simtime()
                    sim_times.append(sim_time)
                
                min_time = min(sim_times)
                max_time = max(sim_times)
                
                if abs(max_time - min_time) > 0.01:
                    print(f"Lambda {lam_window.lam:.3f}: ❌ Still inconsistent: {sim_times}")
                    all_consistent = False
                else:
                    print(f"Lambda {lam_window.lam:.3f}: ✓ Consistent at {min_time:.6f} ns")
            print()
    
    if all_consistent:
        print("✅ ALL SIMULATIONS NOW HAVE CONSISTENT TIMES!")
    else:
        print("❌ Some simulations still have inconsistent times")
    
    return all_consistent


def fix_simulation_times(calc):
    """
    Complete workflow to fix inconsistent simulation times.
    
    Parameters
    ----------
    calc : a3fe.Calculation
        The calculation object to fix
    """
    truncate_simulations_to_minimum(calc)    
    success = verify_truncation(calc)
    
    if success:
        print("\n🎉 Ready to proceed!")
    else:
        print("\n⚠️  Some issues remain. Manual inspection may be required.")
    
    return success

