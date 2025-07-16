#!/usr/bin/env python3
"""
Minimal test to verify PPO cycle logging works
"""

import os
import sys
import subprocess
import signal
import time

def run_with_timeout(cmd, timeout_sec=30):
    """Run command with timeout"""
    print(f"Running command with {timeout_sec}s timeout: {' '.join(cmd)}")
    
    # Set up environment
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PPO_CYCLE_DEBUG"] = "1"
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True, 
        bufsize=1, 
        env=env,
        preexec_fn=os.setsid  # Create new process group for proper cleanup
    )
    
    output_lines = []
    ppo_cycle_lines = []
    
    try:
        while True:
            # Check timeout
            if time.time() - start_time > timeout_sec:
                print(f"\n⏱️  Timeout reached ({timeout_sec}s)")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                break
                
            # Read output with timeout
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                line = line.rstrip()
                output_lines.append(line)
                
                # Print relevant lines
                if any(keyword in line for keyword in ["[PPO_CYCLE]", "Error", "Starting", "initialized", "PPO"]):
                    print(line)
                    
                if "[PPO_CYCLE]" in line:
                    ppo_cycle_lines.append(line)
                    
                # Stop after we see first PPO cycle complete
                if "Buffer full" in line and "starting PPO update" in line:
                    print("\n✅ Found PPO cycle trigger! Stopping...")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    break
                    
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
    
    return output_lines, ppo_cycle_lines

def main():
    # Change to DNNE directory
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    print("Testing PPO cycle logging with minimal runner...\n")
    
    cmd = [
        sys.executable,
        "export_system/exports/Cartpole_PPO/runner.py",
        "--fixed-seed=42",
        "--epochs=1",
        "--headless"
    ]
    
    output, ppo_lines = run_with_timeout(cmd, timeout_sec=30)
    
    print(f"\n📊 Summary:")
    print(f"   Total output lines: {len(output)}")
    print(f"   PPO_CYCLE lines: {len(ppo_lines)}")
    
    if ppo_lines:
        print("\n✅ PPO_CYCLE logging is working!")
        print("First few PPO_CYCLE lines:")
        for line in ppo_lines[:5]:
            print(f"   {line}")
    else:
        print("\n❌ No PPO_CYCLE logs found")
        
        # Look for initialization
        init_lines = [l for l in output if "initialized" in l.lower()]
        if init_lines:
            print("\nInitialization lines found:")
            for line in init_lines[:3]:
                print(f"   {line}")
        
        # Look for errors
        error_lines = [l for l in output if "error" in l.lower() or "exception" in l.lower()]
        if error_lines:
            print("\nErrors found:")
            for line in error_lines[:3]:
                print(f"   {line}")

if __name__ == "__main__":
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    main()