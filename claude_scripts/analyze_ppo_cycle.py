#!/usr/bin/env python3
"""
Analyze PPO cycle debug output from performance profiler

Compares the computation values between DNNE and IsaacGymEnvs
to identify where the implementations diverge.
"""

import json
import sys
from pathlib import Path
import numpy as np

def load_debug_data(system_name):
    """Load PPO cycle debug data for a system"""
    file_path = f'/tmp/{system_name}_ppo_cycle_debug.json'
    
    if not Path(file_path).exists():
        print(f"❌ Debug data not found: {file_path}")
        print(f"   Run performance_profiler.py with --ppo-cycle-debug first")
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_values(dnne_data, isaac_data):
    """Compare values between DNNE and IsaacGymEnvs"""
    print("\n" + "="*80)
    print("PPO CYCLE COMPARISON ANALYSIS")
    print("="*80)
    
    # Check if we have data
    if not dnne_data['actions'] or not isaac_data['actions']:
        print("\n❌ No action data captured. Make sure:")
        print("   1. Both systems ran with --ppo-cycle-debug")
        print("   2. The debug logging in nodes is working")
        print("   3. The systems completed at least a few steps")
        return
    
    # Compare actions
    print("\n1. ACTIONS COMPARISON")
    print("-" * 40)
    num_steps = min(len(dnne_data['actions']), len(isaac_data['actions']))
    
    if num_steps > 0:
        print(f"Comparing first {num_steps} steps:")
        print(f"{'Step':>4} {'DNNE':>10} {'Isaac':>10} {'Diff':>10} {'%Diff':>10}")
        print("-" * 54)
        
        max_diff = 0
        max_diff_step = 0
        
        for i in range(num_steps):
            dnne_val = dnne_data['actions'][i]
            isaac_val = isaac_data['actions'][i]
            diff = abs(dnne_val - isaac_val)
            pct_diff = (diff / abs(isaac_val) * 100) if isaac_val != 0 else 0
            
            print(f"{i+1:4d} {dnne_val:10.6f} {isaac_val:10.6f} {diff:10.6f} {pct_diff:9.2f}%")
            
            if diff > max_diff:
                max_diff = diff
                max_diff_step = i + 1
        
        print(f"\nMax action difference: {max_diff:.6f} at step {max_diff_step}")
        
        # Check if actions match
        if max_diff < 1e-6:
            print("✅ Actions match perfectly!")
        elif max_diff < 1e-3:
            print("✅ Actions are very close (likely numerical precision)")
        else:
            print("⚠️  Actions show significant differences")
    
    # Compare values
    print("\n2. VALUE PREDICTIONS COMPARISON")
    print("-" * 40)
    num_steps = min(len(dnne_data['values']), len(isaac_data['values']))
    
    if num_steps > 0:
        print(f"Comparing first {num_steps} steps:")
        print(f"{'Step':>4} {'DNNE':>10} {'Isaac':>10} {'Diff':>10} {'%Diff':>10}")
        print("-" * 54)
        
        max_diff = 0
        max_diff_step = 0
        
        for i in range(num_steps):
            dnne_val = dnne_data['values'][i]
            isaac_val = isaac_data['values'][i]
            diff = abs(dnne_val - isaac_val)
            pct_diff = (diff / abs(isaac_val) * 100) if isaac_val != 0 else 0
            
            print(f"{i+1:4d} {dnne_val:10.6f} {isaac_val:10.6f} {diff:10.6f} {pct_diff:9.2f}%")
            
            if diff > max_diff:
                max_diff = diff
                max_diff_step = i + 1
        
        print(f"\nMax value difference: {max_diff:.6f} at step {max_diff_step}")
        
        # Check if values match
        if max_diff < 1e-6:
            print("✅ Values match perfectly!")
        elif max_diff < 1e-3:
            print("✅ Values are very close (likely numerical precision)")
        else:
            print("⚠️  Values show significant differences")
    
    # Compare rewards
    print("\n3. REWARDS COMPARISON")
    print("-" * 40)
    num_steps = min(len(dnne_data['rewards']), len(isaac_data['rewards']))
    
    if num_steps > 0:
        print(f"Comparing first {num_steps} steps:")
        print(f"{'Step':>4} {'DNNE':>10} {'Isaac':>10} {'Diff':>10}")
        print("-" * 44)
        
        max_diff = 0
        max_diff_step = 0
        
        for i in range(num_steps):
            dnne_val = dnne_data['rewards'][i]
            isaac_val = isaac_data['rewards'][i]
            diff = abs(dnne_val - isaac_val)
            
            print(f"{i+1:4d} {dnne_val:10.6f} {isaac_val:10.6f} {diff:10.6f}")
            
            if diff > max_diff:
                max_diff = diff
                max_diff_step = i + 1
        
        print(f"\nMax reward difference: {max_diff:.6f} at step {max_diff_step}")
        
        # Check if rewards match
        if max_diff < 1e-6:
            print("✅ Rewards match perfectly!")
        elif max_diff < 1e-3:
            print("✅ Rewards are very close (likely numerical precision)")
        else:
            print("⚠️  Rewards show significant differences")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Show captured PPO cycle logs
    print("\nDNNE PPO cycle logs:")
    for log in dnne_data['ppo_cycle_logs'][:5]:
        print(f"  {log}")
    if len(dnne_data['ppo_cycle_logs']) > 5:
        print(f"  ... and {len(dnne_data['ppo_cycle_logs']) - 5} more")
    
    print("\nIsaacGymEnvs PPO cycle logs:")
    for log in isaac_data['ppo_cycle_logs'][:5]:
        print(f"  {log}")
    if len(isaac_data['ppo_cycle_logs']) > 5:
        print(f"  ... and {len(isaac_data['ppo_cycle_logs']) - 5} more")

def main():
    print("🔍 PPO Cycle Analysis Tool")
    
    # Load debug data
    dnne_data = load_debug_data('dnne')
    isaac_data = load_debug_data('isaacgym')
    
    if not dnne_data or not isaac_data:
        print("\n❌ Cannot proceed without debug data from both systems")
        sys.exit(1)
    
    # Compare the values
    compare_values(dnne_data, isaac_data)
    
    print("\n💡 Next steps:")
    print("   - If values diverge early, check initial model weights")
    print("   - If actions differ, check action sampling and distribution")
    print("   - If rewards differ, check environment implementation")
    print("   - If values differ but actions match, check value network")

if __name__ == "__main__":
    main()