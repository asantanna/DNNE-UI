#!/usr/bin/env python3
"""
Test GPU cleanup and basic Isaac Gym functionality
"""

import subprocess
import sys
import os
import time

def cleanup_gpu():
    """Try to clean up GPU state"""
    print("[CLEANUP] Attempting GPU cleanup...")
    
    # Try nvidia-smi reset if available
    try:
        result = subprocess.run(["nvidia-smi", "--gpu-reset"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("[CLEANUP] GPU reset successful")
        else:
            print("[CLEANUP] GPU reset not available (requires sudo)")
    except:
        print("[CLEANUP] nvidia-smi not available")
    
    # Kill any hanging python processes
    try:
        subprocess.run(["pkill", "-f", "python.*isaac"], capture_output=True)
        print("[CLEANUP] Killed any hanging Isaac Gym processes")
    except:
        pass
    
    time.sleep(2)  # Give GPU time to settle

def test_basic_isaac():
    """Test basic Isaac Gym without torch import issues"""
    script = '''
import isaacgym
from isaacgym import gymapi

print("[TEST] Isaac Gym imported successfully")

gym = gymapi.acquire_gym()
print("[TEST] Gym acquired")

# Minimal sim params
sim_params = gymapi.SimParams()
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True

# Try to create sim
sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
if sim:
    print("[TEST] ✅ Simulation created successfully!")
    gym.destroy_sim(sim)
else:
    print("[TEST] ❌ Failed to create simulation")
'''
    
    with open("/tmp/test_isaac_basic.py", "w") as f:
        f.write(script)
    
    result = subprocess.run([sys.executable, "/tmp/test_isaac_basic.py"],
                          capture_output=True, text=True)
    
    print("[TEST] Basic Isaac Gym test:")
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0

def test_simple_cartpole():
    """Test creating Cartpole without DNNE"""
    script = '''
import isaacgym
import torch
import numpy as np

# Simple test of creating environments
from isaacgym import gymapi

gym = gymapi.acquire_gym()

# Create sim
sim_params = gymapi.SimParams()
sim_params.dt = 0.01
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True

sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
if not sim:
    print("[TEST] Failed to create sim")
    exit(1)

# Add ground
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# Create one env
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)

# Create a simple box as cartpole substitute
box_asset = gym.create_box(sim, 0.1, 0.1, 0.1, gymapi.AssetOptions())
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 1)
actor = gym.create_actor(env, box_asset, pose, "box", 0, 1, 0)

# Prepare and simulate one step
gym.prepare_sim(sim)
gym.simulate(sim)
gym.fetch_results(sim, True)

print("[TEST] ✅ Simple environment working!")

# Cleanup
gym.destroy_sim(sim)
'''
    
    with open("/tmp/test_simple_env.py", "w") as f:
        f.write(script)
    
    result = subprocess.run([sys.executable, "/tmp/test_simple_env.py"],
                          capture_output=True, text=True, timeout=10)
    
    print("\n[TEST] Simple environment test:")
    print(result.stdout)
    if result.stderr and "invalid resource handle" in result.stderr:
        print("❌ CUDA error detected:", result.stderr[:200])
        return False
    
    return result.returncode == 0

def main():
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    print("GPU/CUDA Diagnostic Tests\n")
    
    # Check GPU
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.free", 
                           "--format=csv,noheader"],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("[GPU] Status:", result.stdout.strip())
    
    # Try cleanup
    cleanup_gpu()
    
    # Test basic Isaac
    if test_basic_isaac():
        print("\n✅ Basic Isaac Gym works")
        
        # Test simple env
        if test_simple_cartpole():
            print("\n✅ Simple environment works")
            print("\nGPU appears healthy. The DNNE CUDA error might be from:")
            print("  1. Import order issues in the exported code")
            print("  2. Multiple Isaac Gym instances conflicting")
            print("  3. GPU memory fragmentation")
        else:
            print("\n❌ Simple environment failed - GPU may need reset")
    else:
        print("\n❌ Basic Isaac Gym failed - serious GPU issue")

if __name__ == "__main__":
    main()