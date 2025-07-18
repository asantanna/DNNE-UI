#!/usr/bin/env python3
"""
Test script to run IsaacGymEnvs with PPO_CYCLE_DEBUG
"""

import os
import sys
import subprocess

# Set environment variables
os.environ['PPO_CYCLE_DEBUG'] = '1'
os.environ['USE_RL_GAMES_DNNE'] = '1'

# Change to IsaacGymEnvs directory
ige_path = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs"
os.chdir(ige_path)

# Run train.py with timeout
cmd = [
    sys.executable,
    "isaacgymenvs/train.py",
    "task=Cartpole",
    "headless=True",
    "num_envs=512",
    "train.params.config.max_epochs=1",
    "train.params.config.horizon_length=16",
    "train.params.config.minibatch_size=8192",
]

print("Running IsaacGymEnvs with PPO_CYCLE_DEBUG=1")
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

try:
    # Run with timeout
    result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)
    
    # Save output
    with open("/tmp/ige_debug_output.txt", "w") as f:
        f.write(result.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)
    
    # Print key debug lines
    for line in result.stdout.split('\n'):
        if 'PPO_CYCLE_DEBUG' in line or 'Started to train' in line:
            print(line)
            
except subprocess.TimeoutExpired:
    print("IGE training timed out after 10 seconds")
except Exception as e:
    print(f"Error running IGE: {e}")