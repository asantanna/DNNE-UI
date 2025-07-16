#!/usr/bin/env python3
"""
Run PPO cycle comparison - one at a time to avoid GPU conflicts
"""

import subprocess
import sys
import os
import json
import time

def run_command(cmd, env=None, timeout=30):
    """Run command and capture output"""
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=env, 
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s")
        return "", "Timeout", -1

def main():
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    # First run IsaacGymEnvs
    print("\n" + "="*80)
    print("1. Running IsaacGymEnvs single PPO cycle")
    print("="*80)
    
    isaac_stdout, isaac_stderr, isaac_code = run_command(
        [sys.executable, "claude_scripts/debug_isaacgym_single_cycle.py"]
    )
    
    isaac_logs = [line for line in isaac_stdout.split('\n') if '[PPO_CYCLE]' in line]
    
    print(f"\nIsaacGym return code: {isaac_code}")
    if isaac_logs:
        print("IsaacGym PPO_CYCLE logs found:")
        for log in isaac_logs:
            print(f"  {log}")
    else:
        print("No IsaacGym PPO_CYCLE logs found")
        if isaac_stderr:
            print(f"Errors: {isaac_stderr[:500]}")
    
    # Wait a bit for GPU to settle
    time.sleep(2)
    
    # Then run DNNE
    print("\n" + "="*80)
    print("2. Running DNNE single PPO cycle")
    print("="*80)
    
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PPO_CYCLE_DEBUG"] = "1"
    
    dnne_stdout, dnne_stderr, dnne_code = run_command(
        [sys.executable, "export_system/exports/Cartpole_PPO/runner.py", 
         "--fixed-seed=42", "--epochs=1", "--headless"],
        env=env
    )
    
    dnne_logs = [line for line in dnne_stdout.split('\n') if '[PPO_CYCLE]' in line]
    
    print(f"\nDNNE return code: {dnne_code}")
    if dnne_logs:
        print("DNNE PPO_CYCLE logs found:")
        for log in dnne_logs:
            print(f"  {log}")
    else:
        print("No DNNE PPO_CYCLE logs found")
        if dnne_stderr:
            print(f"Errors: {dnne_stderr[:500]}")
    
    # Compare results
    print("\n" + "="*80)
    print("3. COMPARISON")
    print("="*80)
    
    if isaac_logs and dnne_logs:
        print("✅ Both systems produced PPO_CYCLE logs")
        
        # Extract first few actions for comparison
        print("\nComparing first few actions:")
        
        isaac_actions = []
        dnne_actions = []
        
        for log in isaac_logs:
            if "Step" in log and "action=" in log:
                try:
                    action = float(log.split("action=")[1].split(",")[0])
                    isaac_actions.append(action)
                except:
                    pass
        
        for log in dnne_logs:
            if "Step" in log and "action=" in log:
                try:
                    action = float(log.split("action=")[1].split(",")[0])
                    dnne_actions.append(action)
                except:
                    pass
        
        if isaac_actions and dnne_actions:
            print(f"  Isaac actions: {isaac_actions[:5]}")
            print(f"  DNNE actions:  {dnne_actions[:5]}")
            
            # Compare
            max_diff = max(abs(i - d) for i, d in zip(isaac_actions[:min(5, len(isaac_actions), len(dnne_actions))], 
                                                       dnne_actions[:min(5, len(isaac_actions), len(dnne_actions))]))
            if max_diff < 0.0001:
                print("  ✅ Actions match!")
            else:
                print(f"  ⚠️  Max difference: {max_diff:.6f}")
        
    else:
        print("❌ One or both systems failed to produce PPO_CYCLE logs")
    
    # Save results
    results = {
        "isaac_logs": isaac_logs,
        "dnne_logs": dnne_logs,
        "isaac_code": isaac_code,
        "dnne_code": dnne_code
    }
    
    with open("/tmp/ppo_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to /tmp/ppo_comparison_results.json")

if __name__ == "__main__":
    main()