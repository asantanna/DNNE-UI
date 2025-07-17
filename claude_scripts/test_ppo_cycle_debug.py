#!/usr/bin/env python3
"""Test DNNE vs IsaacGymEnvs with PPO_CYCLE_DEBUG enabled"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

def run_test(system, command, env_vars, timeout=30):
    """Run a test and capture output"""
    print(f"\n{'='*60}")
    print(f"Testing {system} with PPO_CYCLE_DEBUG")
    print(f"Command: {command}")
    print(f"Env vars: {env_vars}")
    print(f"{'='*60}\n")
    
    # Prepare environment
    env = os.environ.copy()
    env.update(env_vars)
    
    # Run command
    start_time = time.time()
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True
    )
    
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        elapsed = time.time() - start_time
        
        print(f"\n{system} completed in {elapsed:.2f}s")
        
        # Save output
        output_file = f"/tmp/ppo_cycle_debug_{system.lower().replace(' ', '_')}.log"
        with open(output_file, 'w') as f:
            f.write(f"=== {system} PPO_CYCLE_DEBUG Output ===\n")
            f.write(f"Command: {command}\n")
            f.write(f"Env vars: {env_vars}\n")
            f.write(f"Elapsed: {elapsed:.2f}s\n")
            f.write("\n=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n=== STDERR ===\n")
            f.write(stderr)
        
        print(f"Output saved to: {output_file}")
        
        # Extract key debug lines
        debug_lines = [line for line in stdout.split('\n') if '[DNNE_DEBUG]' in line]
        print(f"\nFound {len(debug_lines)} debug lines")
        
        if debug_lines:
            print("\nFirst 10 debug lines:")
            for line in debug_lines[:10]:
                print(f"  {line}")
        
        return {
            'success': proc.returncode == 0,
            'elapsed': elapsed,
            'output_file': output_file,
            'debug_lines': debug_lines,
            'stdout': stdout,
            'stderr': stderr
        }
        
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"\n{system} timed out after {timeout}s")
        return {
            'success': False,
            'elapsed': timeout,
            'error': 'timeout',
            'stdout': '',
            'stderr': ''
        }

def main():
    """Run PPO_CYCLE_DEBUG comparison test"""
    
    # Test configuration
    common_env = {
        'PPO_CYCLE_DEBUG': '1',
        'FIXED_SEED': '42',
        'EPOCHS_OVERRIDE': '2'  # Short test
    }
    
    # Test 1: DNNE
    print("\n" + "="*80)
    print("TESTING DNNE WITH PPO_CYCLE_DEBUG")
    print("="*80)
    
    dnne_env = common_env.copy()
    dnne_result = run_test(
        "DNNE",
        "cd /mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO && python runner.py",
        dnne_env,
        timeout=60
    )
    
    # Test 2: IsaacGymEnvs
    print("\n" + "="*80)
    print("TESTING ISAACGYMENVS WITH PPO_CYCLE_DEBUG")
    print("="*80)
    
    ige_env = common_env.copy()
    ige_env['USE_RL_GAMES_DEBUG'] = '1'  # Use debug version
    
    ige_result = run_test(
        "IsaacGymEnvs",
        "cd ~/IsaacGymEnvs && python train.py task=Cartpole test=True num_envs=512 headless=True",
        ige_env,
        timeout=60
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nDNNE:")
    print(f"  Success: {dnne_result['success']}")
    print(f"  Elapsed: {dnne_result['elapsed']:.2f}s")
    print(f"  Debug lines: {len(dnne_result.get('debug_lines', []))}")
    
    print(f"\nIsaacGymEnvs:")
    print(f"  Success: {ige_result['success']}")
    print(f"  Elapsed: {ige_result['elapsed']:.2f}s")
    print(f"  Debug lines: {len(ige_result.get('debug_lines', []))}")
    
    # Extract first few actions for comparison
    print("\n" + "="*80)
    print("FIRST ACTION COMPARISON")
    print("="*80)
    
    def extract_first_action(debug_lines):
        for line in debug_lines:
            if "Sampled action:" in line or "Action:" in line:
                return line
        return None
    
    dnne_action = extract_first_action(dnne_result.get('debug_lines', []))
    ige_action = extract_first_action(ige_result.get('debug_lines', []))
    
    print(f"\nDNNE first action: {dnne_action}")
    print(f"IGE first action: {ige_action}")
    
    # Save comparison
    comparison = {
        'timestamp': time.time(),
        'dnne': {
            'success': dnne_result['success'],
            'elapsed': dnne_result['elapsed'],
            'debug_line_count': len(dnne_result.get('debug_lines', [])),
            'output_file': dnne_result.get('output_file', ''),
            'first_action': dnne_action
        },
        'ige': {
            'success': ige_result['success'],
            'elapsed': ige_result['elapsed'], 
            'debug_line_count': len(ige_result.get('debug_lines', [])),
            'output_file': ige_result.get('output_file', ''),
            'first_action': ige_action
        }
    }
    
    comparison_file = '/tmp/ppo_cycle_debug_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {comparison_file}")
    print("\nTo analyze debug output:")
    print(f"  grep -n 'DNNE_DEBUG' {dnne_result.get('output_file', '')} | head -50")
    print(f"  grep -n 'DNNE_DEBUG' {ige_result.get('output_file', '')} | head -50")

if __name__ == '__main__':
    main()