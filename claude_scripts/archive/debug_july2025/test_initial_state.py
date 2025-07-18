#!/usr/bin/env python3
"""
Test initial state of Isaac Gym Cartpole environment
"""

import sys
import os
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT/isaacgym/python')
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs')

import isaacgym
import torch
import numpy as np
from isaacgym import gymapi, gymtorch

# Create gym
gym = gymapi.acquire_gym()

# Sim params
sim_params = gymapi.SimParams()
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 0.0166
sim_params.substeps = 2

# Create sim
sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)

# Add ground
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# Load asset
asset_root = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/assets"
asset_file = "urdf/cartpole.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Create env
env_lower = gymapi.Vec3(-4.0, -4.0, 0.0)
env_upper = gymapi.Vec3(4.0, 4.0, 4.0)
env = gym.create_env(sim, env_lower, env_upper, 1)

# Create actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
actor = gym.create_actor(env, cartpole_asset, pose, "cartpole", 0, 1, 0)

# Configure DOF
dof_props = gym.get_actor_dof_properties(env, actor)
dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
dof_props['stiffness'][:] = 0.0
dof_props['damping'][:] = 0.0
gym.set_actor_dof_properties(env, actor, dof_props)

# Prepare sim
gym.prepare_sim(sim)

# Get DOF state
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)

print("Initial DOF state shape:", dof_state.shape)
print("Initial DOF state:")
print(dof_state)

# Refresh and get again
gym.refresh_dof_state_tensor(sim)
print("\nAfter refresh:")
print(dof_state)

# Set random initial state
torch.manual_seed(42)
positions = 0.2 * (torch.rand((1, 2), device='cuda') - 0.5)
velocities = 0.5 * (torch.rand((1, 2), device='cuda') - 0.5)

print(f"\nRandom positions: {positions}")
print(f"Random velocities: {velocities}")

# Apply to DOF state
dof_state[0, 0] = positions[0, 0]  # cart pos
dof_state[0, 1] = velocities[0, 0]  # cart vel
dof_state[1, 0] = positions[0, 1]  # pole angle
dof_state[1, 1] = velocities[0, 1]  # pole vel

# Set state
gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

# Refresh and check
gym.refresh_dof_state_tensor(sim)
print("\nAfter setting random state:")
print(dof_state)

# Get observations
dof_pos = dof_state.view(1, 2, 2)[..., 0]
dof_vel = dof_state.view(1, 2, 2)[..., 1]

obs = torch.zeros((1, 4), device='cuda')
obs[:, 0] = dof_pos[:, 0]  # cart pos
obs[:, 1] = dof_vel[:, 0]  # cart vel
obs[:, 2] = dof_pos[:, 1]  # pole angle
obs[:, 3] = dof_vel[:, 1]  # pole vel

print(f"\nObservations: {obs}")

# Clean up
gym.destroy_sim(sim)