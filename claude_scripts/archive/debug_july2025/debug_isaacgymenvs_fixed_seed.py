#!/usr/bin/env python3
"""
IsaacGymEnvs wrapper script with fixed seed support for debugging PPO differences
"""

import subprocess
import sys
import os
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run IsaacGymEnvs with fixed seed for debugging")
    parser.add_argument("--fixed-seed", type=int, default=42, help="Fixed random seed")
    parser.add_argument("--num-envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--max-iterations", type=int, default=1, help="Max iterations (1 for single cycle)")
    parser.add_argument("--capture-stdout", action="store_true", help="Capture and display all stdout")
    args = parser.parse_args()
    
    # Activate conda environment
    print("🔧 Activating conda environment...")
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    # Change to IsaacGymEnvs directory
    isaacgymenvs_path = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs"
    os.chdir(isaacgymenvs_path)
    print(f"📁 Changed directory to: {os.getcwd()}")
    
    # Build IsaacGymEnvs command with fixed seed
    cmd = [
        sys.executable,
        "train.py",
        "task=Cartpole",
        f"seed={args.fixed_seed}",
        f"num_envs={args.num_envs}",
        f"max_iterations={args.max_iterations}",
        "train.params.config.horizon_length=16",  # Match DNNE
        "train.params.config.minibatch_size=8192",  # Match DNNE
        "torch_deterministic=True",  # Enable deterministic mode
        "train.params.config.normalize_input=True",  # Ensure normalization
        "train.params.config.normalize_value=True",  # Ensure value normalization
    ]
    
    if args.headless:
        cmd.append("headless=True")
    else:
        cmd.append("force_render=True")
    
    # Add debug logging
    cmd.append("train.params.config.print_stats=True")
    
    print(f"🚀 Running IsaacGymEnvs with fixed seed: {args.fixed_seed}")
    print(f"📊 Command: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run with output capture
    if args.capture_stdout:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                
                # Look for key debug values
                if "RunningMeanStd:" in line:
                    print(f"🔍 FOUND: {line.rstrip()}")
                elif "current training device:" in line:
                    print(f"🔍 FOUND: {line.rstrip()}")
                elif "fps step:" in line and args.max_iterations == 1:
                    # For single iteration, we might want to stop after first step
                    print("🛑 First step complete, stopping...")
                    process.terminate()
                    break
        
        process.wait()
    else:
        # Run normally
        subprocess.run(cmd)
    
    print("\n✅ IsaacGymEnvs run complete")

if __name__ == "__main__":
    main()