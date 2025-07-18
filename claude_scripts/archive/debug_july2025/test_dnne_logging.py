#!/usr/bin/env python3
"""
Test DNNE logging to see why PPO_CYCLE logs aren't captured
"""

import subprocess
import sys
import os

def test_dnne_logging():
    """Run DNNE with debug flags and capture all output"""
    print("Testing DNNE logging...")
    
    # Set up environment
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PPO_CYCLE_DEBUG"] = "1"
    env["PPO_STOP_AFTER_CYCLE"] = "1"
    
    # Change to DNNE directory
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    cmd = [
        sys.executable,
        "export_system/exports/Cartpole_PPO/runner.py",
        "--fixed-seed=42",
        "--epochs=1",
        "--headless",
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Environment: PPO_CYCLE_DEBUG={env.get('PPO_CYCLE_DEBUG')}, PPO_STOP_AFTER_CYCLE={env.get('PPO_STOP_AFTER_CYCLE')}")
    
    # Run with real-time output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, env=env)
    
    all_output = []
    ppo_cycle_logs = []
    
    print("\n--- DNNE OUTPUT ---")
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            all_output.append(line.rstrip())
            if "[PPO_CYCLE]" in line:
                ppo_cycle_logs.append(line.rstrip())
    
    process.wait()
    return_code = process.returncode
    
    print(f"\n--- END OF OUTPUT (return code: {return_code}) ---")
    print(f"\nTotal lines captured: {len(all_output)}")
    print(f"PPO_CYCLE lines found: {len(ppo_cycle_logs)}")
    
    if ppo_cycle_logs:
        print("\nPPO_CYCLE logs:")
        for log in ppo_cycle_logs:
            print(f"  {log}")
    else:
        print("\n⚠️  No PPO_CYCLE logs found!")
        
        # Check if process failed early
        if return_code != 0:
            print(f"❌ Process failed with return code {return_code}")
        
        # Look for any errors
        errors = [line for line in all_output if "error" in line.lower() or "exception" in line.lower()]
        if errors:
            print("\nErrors found:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  {error}")

if __name__ == "__main__":
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    test_dnne_logging()