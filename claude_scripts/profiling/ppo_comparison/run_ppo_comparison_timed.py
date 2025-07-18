#!/usr/bin/env python3
"""
Master comparison script for PPO training between DNNE and IsaacGymEnvs
Runs both systems with 1 epoch and measures execution time
"""
import subprocess
import os
import time
import sys

def run_command(cmd, cwd, env, description):
    """Run a command and measure execution time"""
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"Working directory: {cwd}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run command
    if isinstance(cmd, str):
        # For bash commands
        process = subprocess.Popen(
            ['/bin/bash', '-c', cmd],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    else:
        # For regular commands
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    # Capture output
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    
    process.wait()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"{description} completed in {elapsed_time:.2f} seconds")
    print(f"Return code: {process.returncode}")
    print(f"{'='*60}\n")
    
    return process.returncode, elapsed_time, output_lines

def run_dnne_test():
    """Run DNNE with 1 epoch"""
    env = os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA
    
    # Remove timeout - let it run to completion
    cmd = [
        'python', 'runner.py',
        '--fixed-seed', '42',
        '--epochs', '1',
        '--headless'
    ]
    
    cwd = '/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO'
    
    # Activate conda environment
    activate_cmd = 'source /home/asantanna/miniconda/bin/activate DNNE_PY38 && ' + ' '.join(cmd)
    
    return run_command(activate_cmd, cwd, env, "DNNE (1 epoch)")

def run_ige_test():
    """Run IsaacGymEnvs with 1 epoch"""
    env = os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA
    
    cmd = [
        'python', 'train.py',
        'task=Cartpole',
        'train.params.config.max_epochs=1',
        'seed=42',
        'headless=True'
    ]
    
    cwd = '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs'
    
    # Activate conda environment
    activate_cmd = 'source /home/asantanna/miniconda/bin/activate DNNE_PY38 && ' + ' '.join(cmd)
    
    return run_command(activate_cmd, cwd, env, "IsaacGymEnvs (1 epoch)")

def main():
    print("=== PPO Training Comparison: DNNE vs IsaacGymEnvs ===")
    print("Both systems will run for exactly 1 epoch")
    print("Using fixed seed 42 for deterministic results\n")
    
    # Run DNNE
    dnne_code, dnne_time, dnne_output = run_dnne_test()
    
    # Run IGE
    ige_code, ige_time, ige_output = run_ige_test()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"DNNE:")
    print(f"  - Execution time: {dnne_time:.2f} seconds")
    print(f"  - Return code: {dnne_code}")
    print(f"  - Completed successfully: {'Yes' if dnne_code == 0 else 'No'}")
    
    print(f"\nIsaacGymEnvs:")
    print(f"  - Execution time: {ige_time:.2f} seconds")
    print(f"  - Return code: {ige_code}")
    print(f"  - Completed successfully: {'Yes' if ige_code == 0 else 'No'}")
    
    print(f"\nPerformance comparison:")
    print(f"  - DNNE is {ige_time/dnne_time:.2f}x faster" if dnne_time < ige_time else f"  - IGE is {dnne_time/ige_time:.2f}x faster")
    
    # Save outputs for further analysis
    with open('/tmp/dnne_1epoch_output.log', 'w') as f:
        f.writelines(dnne_output)
    
    with open('/tmp/ige_1epoch_output.log', 'w') as f:
        f.writelines(ige_output)
    
    print(f"\nFull outputs saved to:")
    print(f"  - /tmp/dnne_1epoch_output.log")
    print(f"  - /tmp/ige_1epoch_output.log")

if __name__ == '__main__':
    main()