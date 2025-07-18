#!/usr/bin/env python3
"""
Test basic Isaac Gym functionality
"""

import isaacgym
import torch

print("Isaac Gym imported successfully")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# Test basic gym functionality
from isaacgym import gymapi

gym = gymapi.acquire_gym()
print("Gym acquired successfully")

# Check compute device
compute_device = 0
graphics_device = -1  # headless

# Create minimal sim
sim_params = gymapi.SimParams()
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True

print("Creating simulation...")
sim = gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)

if sim is None:
    print("ERROR: Failed to create simulation")
else:
    print("Simulation created successfully")
    gym.destroy_sim(sim)
    print("Simulation destroyed successfully")

print("Basic Isaac Gym test complete")