#!/usr/bin/env python3
"""
Run both DNNE and IGE for exactly 1 PPO cycle with debug output
Both systems now support PPO_STOP_AFTER_CYCLE=1
"""
import subprocess
import os
import time

def run_command_with_output(cmd, cwd, env, output_file, description):
    """Run command and save output to file"""
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"Output file: {output_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with open(output_file, 'w') as f:
        process = subprocess.Popen(
            ['/bin/bash', '-c', cmd],
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for completion
        return_code = process.wait()
        
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"✓ {description} completed in {elapsed:.2f} seconds")
    print(f"  Return code: {return_code}")
    
    # Check file size
    file_size = os.path.getsize(output_file)
    print(f"  Output size: {file_size:,} bytes")
    
    # Check for PPO_STOP message
    with open(output_file, 'r') as f:
        content = f.read()
        if "PPO_STOP:" in content or "PPO training complete after" in content:
            print(f"  ✓ Stopped after PPO cycle as requested")
        else:
            print(f"  ⚠️  No PPO stop message found")
    
    return return_code, elapsed

def main():
    print("=== PPO 1-Cycle Comparison (Both Systems) ===")
    print("Both systems will run for exactly 1 PPO cycle (16 steps)")
    print("Debug output enabled for detailed comparison\n")
    
    # Common environment setup
    base_env = os.environ.copy()
    base_env['PPO_CYCLE_DEBUG'] = '1'
    base_env['PPO_STOP_AFTER_CYCLE'] = '1'
    base_env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Run DNNE
    dnne_cmd = 'source /home/asantanna/miniconda/bin/activate DNNE_PY38 && python runner.py --fixed-seed 42 --epochs 999 --headless'
    dnne_cwd = '/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO'
    dnne_output = '/tmp/dnne_1cycle_final.log'
    
    dnne_code, dnne_time = run_command_with_output(
        dnne_cmd, dnne_cwd, base_env, dnne_output, "DNNE (1 PPO cycle)"
    )
    
    # Run IGE with rl_games_dnne
    ige_env = base_env.copy()
    ige_env['USE_RL_GAMES_DNNE'] = '1'  # Use rl_games_dnne for PPO_STOP_AFTER_CYCLE support
    
    ige_cmd = 'source /home/asantanna/miniconda/bin/activate DNNE_PY38 && python train.py task=Cartpole seed=42 headless=True'
    ige_cwd = '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs'
    ige_output = '/tmp/ige_1cycle_final.log'
    
    ige_code, ige_time = run_command_with_output(
        ige_cmd, ige_cwd, ige_env, ige_output, "IGE with rl_games_dnne (1 PPO cycle)"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"DNNE (1 PPO cycle):")
    print(f"  - Time: {dnne_time:.2f}s")
    print(f"  - Success: {'Yes' if dnne_code == 0 else 'No (exit code: ' + str(dnne_code) + ')'}")
    
    print(f"\nIGE with rl_games_dnne (1 PPO cycle):")
    print(f"  - Time: {ige_time:.2f}s") 
    print(f"  - Success: {'Yes' if ige_code == 0 else 'No (exit code: ' + str(ige_code) + ')'}")
    
    print(f"\nOutput files ready for comparison:")
    print(f"  - DNNE: {dnne_output}")
    print(f"  - IGE: {ige_output}")
    
    print(f"\nBoth systems now stop after exactly 1 PPO cycle!")

if __name__ == '__main__':
    main()