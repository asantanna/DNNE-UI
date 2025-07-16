#!/usr/bin/env python3
"""
Direct test of PPO cycle debugging without profiler
"""

import os
import sys
import subprocess

def run_isaacgym_ppo_cycle():
    """Run IsaacGymEnvs with PPO cycle debug"""
    print("🔬 Testing IsaacGymEnvs PPO cycle debug...")
    
    env = os.environ.copy()
    env['PPO_CYCLE_DEBUG'] = '1'
    env['PPO_STOP_AFTER_CYCLE'] = '1'
    env['FIXED_SEED'] = '42'
    
    cmd = [
        'python',
        '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/train.py',
        'task=Cartpole',
        'task.env.numEnvs=512',
        'train.params.config.max_epochs=1',
        'train.params.config.horizon_length=16',
        'train.params.config.minibatch_size=8192',
        'headless=True',
        'test=False',
        'seed=42'
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    # Print output as it comes
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            if '[PPO_CYCLE]' in line:
                print("✅ FOUND PPO_CYCLE OUTPUT!")
    
    process.wait()
    print(f"\nProcess exited with code: {process.returncode}")

def run_dnne_ppo_cycle():
    """Run DNNE with PPO cycle debug"""
    print("\n🔬 Testing DNNE PPO cycle debug...")
    
    env = os.environ.copy()
    env['PPO_CYCLE_DEBUG'] = '1'
    env['PPO_STOP_AFTER_CYCLE'] = '1'
    env['FIXED_SEED'] = '42'
    
    cmd = [
        'python',
        '/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO/runner.py'
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    # Print output as it comes
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            if '[PPO_CYCLE]' in line:
                print("✅ FOUND PPO_CYCLE OUTPUT!")
    
    process.wait()
    print(f"\nProcess exited with code: {process.returncode}")

if __name__ == "__main__":
    # Run both tests
    run_isaacgym_ppo_cycle()
    run_dnne_ppo_cycle()