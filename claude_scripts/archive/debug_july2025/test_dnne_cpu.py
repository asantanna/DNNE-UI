#!/usr/bin/env python3
"""
Test DNNE with CPU mode
"""

import subprocess
import sys
import os
import time

def test_dnne_cpu():
    """Test DNNE running in CPU mode"""
    
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PPO_CYCLE_DEBUG"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices to force CPU
    
    cmd = [
        sys.executable,
        "export_system/exports/Cartpole_PPO/runner.py",
        "--fixed-seed=42",
        "--epochs=1",
        "--headless"
    ]
    
    print("[TEST] Running DNNE in CPU mode...")
    print(f"[TEST] Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env
    )
    
    output_lines = []
    ppo_lines = []
    
    try:
        while True:
            if time.time() - start_time > 30:
                print("\n[TEST] Timeout after 30s")
                process.terminate()
                break
                
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                line = line.rstrip()
                output_lines.append(line)
                
                # Print relevant lines
                if any(keyword in line for keyword in ["[PPO_CYCLE]", "error", "Error", "Starting", "CPU", "PhysX"]):
                    print(f"  {line}")
                
                if "[PPO_CYCLE]" in line:
                    ppo_lines.append(line)
                    
                # Check if we got the first PPO cycle data
                if "First PPO cycle complete - exiting now" in line:
                    print("\n[TEST] ✅ PPO cycle completed successfully!")
                    break
                    
    except KeyboardInterrupt:
        process.terminate()
    
    process.wait()
    elapsed = time.time() - start_time
    
    print(f"\n[TEST] Results:")
    print(f"  - Elapsed time: {elapsed:.1f}s")
    print(f"  - Return code: {process.returncode}")
    print(f"  - PPO_CYCLE lines: {len(ppo_lines)}")
    
    if ppo_lines:
        print("\n[TEST] PPO cycle logs:")
        for line in ppo_lines:
            print(f"    {line}")
    
    return process.returncode == 0, ppo_lines

def main():
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    print("Testing DNNE with CPU mode\n")
    
    success, ppo_lines = test_dnne_cpu()
    
    if success and ppo_lines:
        print("\n✅ DNNE is working in CPU mode!")
        print("\nNext steps:")
        print("1. Compare PPO cycle values with IsaacGymEnvs")
        print("2. Debug why GPU mode has CUDA errors")
    else:
        print("\n❌ DNNE still has issues even in CPU mode")

if __name__ == "__main__":
    main()