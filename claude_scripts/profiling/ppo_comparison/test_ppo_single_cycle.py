#!/usr/bin/env python3
"""
Test script for single PPO cycle with debug output
Runs DNNE exported code with specific debug settings
"""
import subprocess
import os
import sys

def run_dnne_test():
    # Set environment variables for debugging
    env = os.environ.copy()
    env['PPO_CYCLE_DEBUG'] = '1'
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA
    
    # Command to run DNNE
    cmd = [
        'python', 'runner.py',
        '--fixed-seed', '42',
        '--epochs', '1',  # This is actually many PPO cycles, not just one
        '--headless',
        '--timeout', '30s'  # Safety timeout
    ]
    
    # Run in the exported directory
    cwd = '/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO'
    
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
    with open('/tmp/dnne_ppo_cycle_output.log', 'w') as f:
        for line in process.stdout:
            print(line, end='')  # Also print to console
            f.write(line)
    
    process.wait()
    return process.returncode

if __name__ == '__main__':
    print("Running DNNE PPO single cycle test...")
    result = run_dnne_test()
    print(f"\nTest completed with return code: {result}")