#!/usr/bin/env python3
"""
Test script for single PPO cycle with debug output in IsaacGymEnvs
Using DEFAULT parameters from CartpolePPO.yaml
"""
import subprocess
import os
import sys

def run_ige_test():
    # Set environment variables for debugging
    env = os.environ.copy()
    env['PPO_CYCLE_DEBUG'] = '1'
    env['USE_RL_GAMES_DNNE'] = '1'  # Use rl_games_dnne which has debug output
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA
    
    # Command to run IGE with minimal overrides
    # Only override what's necessary for testing
    cmd = [
        'python', 'isaacgymenvs/train.py',
        'task=Cartpole',
        'seed=42',  # Match DNNE seed
        'train.params.config.max_epochs=1',  # Single epoch only
        'headless=True'
        # Let everything else use defaults from CartpolePPO.yaml
    ]
    
    # Run in IGE directory
    cwd = '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs'
    
    # Activate conda environment
    activate_cmd = 'source /home/asantanna/miniconda/bin/activate DNNE_PY38 && '
    full_cmd = activate_cmd + ' '.join(cmd)
    
    # Run and capture output using bash
    process = subprocess.Popen(
        ['/bin/bash', '-c', full_cmd],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Write output to file
    with open('/tmp/ige_ppo_cycle_output.log', 'w') as f:
        for line in process.stdout:
            print(line, end='')  # Also print to console
            f.write(line)
    
    process.wait()
    return process.returncode

if __name__ == '__main__':
    print("Running IsaacGymEnvs PPO single cycle test...")
    print("Using DEFAULT parameters from CartpolePPO.yaml")
    result = run_ige_test()
    print(f"\nTest completed with return code: {result}")