#!/usr/bin/env python3
"""
Compare single PPO cycles between DNNE and IsaacGymEnvs
"""

import subprocess
import sys
import os
import json

def run_dnne_cycle():
    """Run DNNE for one PPO cycle"""
    print("\n" + "="*80)
    print("Running DNNE single PPO cycle")
    print("="*80)
    
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PPO_CYCLE_DEBUG"] = "1"
    
    cmd = [
        sys.executable,
        "export_system/exports/Cartpole_PPO/runner.py",
        "--fixed-seed=42",
        "--epochs=1",
        "--headless"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    # Extract PPO_CYCLE lines
    dnne_logs = [line for line in result.stdout.split('\n') if '[PPO_CYCLE]' in line]
    
    return dnne_logs, result.returncode

def run_isaac_cycle():
    """Run IsaacGymEnvs for one PPO cycle"""
    print("\n" + "="*80)
    print("Running IsaacGymEnvs single PPO cycle")
    print("="*80)
    
    cmd = [sys.executable, "claude_scripts/debug_isaacgym_single_cycle.py"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract PPO_CYCLE lines
    isaac_logs = [line for line in result.stdout.split('\n') if '[PPO_CYCLE]' in line]
    
    return isaac_logs, result.returncode

def compare_logs(dnne_logs, isaac_logs):
    """Compare the logs from both systems"""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nDNNE PPO_CYCLE logs ({len(dnne_logs)} lines):")
    for log in dnne_logs:
        print(f"  {log}")
    
    print(f"\nIsaacGymEnvs PPO_CYCLE logs ({len(isaac_logs)} lines):")
    for log in isaac_logs:
        print(f"  {log}")
    
    # Extract and compare key values
    print("\n" + "-"*40)
    print("VALUE COMPARISON")
    print("-"*40)
    
    # Extract actions from step logs
    dnne_actions = []
    isaac_actions = []
    
    for log in dnne_logs:
        if "Step" in log and "action=" in log:
            try:
                action_str = log.split("action=")[1].split(",")[0]
                dnne_actions.append(float(action_str))
            except:
                pass
    
    for log in isaac_logs:
        if "Step" in log and "action=" in log:
            try:
                action_str = log.split("action=")[1].split(",")[0]
                isaac_actions.append(float(action_str))
            except:
                pass
    
    if dnne_actions and isaac_actions:
        print("\nFirst 5 actions comparison:")
        print(f"  DNNE:   {dnne_actions[:5]}")
        print(f"  Isaac:  {isaac_actions[:5]}")
        
        # Check if they match
        if len(dnne_actions) >= 5 and len(isaac_actions) >= 5:
            max_diff = max(abs(d - i) for d, i in zip(dnne_actions[:5], isaac_actions[:5]))
            if max_diff < 0.0001:
                print("  ✅ Actions match perfectly!")
            else:
                print(f"  ⚠️  Max action difference: {max_diff:.6f}")
    
    # Save results
    results = {
        "dnne_logs": dnne_logs,
        "isaac_logs": isaac_logs,
        "dnne_actions": dnne_actions,
        "isaac_actions": isaac_actions
    }
    
    with open("/tmp/ppo_cycle_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Full results saved to /tmp/ppo_cycle_comparison.json")

def main():
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    # Run both systems
    dnne_logs, dnne_code = run_dnne_cycle()
    isaac_logs, isaac_code = run_isaac_cycle()
    
    # Check return codes
    print(f"\nReturn codes: DNNE={dnne_code}, Isaac={isaac_code}")
    
    # Compare results
    compare_logs(dnne_logs, isaac_logs)

if __name__ == "__main__":
    main()