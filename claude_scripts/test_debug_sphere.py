#!/usr/bin/env python3
"""
Test script for FrankaDNNE debug sphere visualization feature.
Tests that the debug sphere appears at specified coordinates without physics interactions.
"""

import sys
import os
import torch
import numpy as np
import time

# Add IsaacGymEnvs to path
sys.path.append('/home/asantanna/DNNE/DNNE-LINUX-SUPPORT')
sys.path.append('/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs')

# CRITICAL: Import isaacgym before torch
import isaacgym
from isaacgymenvs import make

def test_debug_sphere():
    """Test the debug sphere visualization in FrankaDNNE environment."""
    
    print("Creating FrankaDNNE environment...")
    
    # Create environment with visualization enabled
    env = make(
        seed=42,
        task="FrankaDNNE",
        num_envs=1,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=False,  # Need viewer to see debug sphere
        cfg_env=None,
        cfg_train=None
    )
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment
    obs = env.reset()
    
    # Test positions for debug sphere (moving in a circle above the table)
    test_positions = []
    radius = 0.3
    height = 1.5
    num_steps = 100
    
    for i in range(num_steps):
        angle = 2 * np.pi * i / num_steps
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height + 0.1 * np.sin(angle * 3)  # Add some vertical movement
        test_positions.append([x, y, z])
    
    print("\nTesting debug sphere visualization...")
    print("The gray sphere should move in a circle above the table.")
    print("It should NOT interact with the robot or other objects.")
    
    # Run simulation with debug sphere
    for step, pos in enumerate(test_positions):
        # Create random action for the robot
        action = torch.randn(1, 7, device="cuda:0") * 0.1
        
        # Create extra_args with debug sphere position
        extra_args = {
            "debug_sphere_pos": pos
        }
        
        # Step environment with debug visualization
        obs, reward, done, info = env.step(action, extra_args)
        
        if step % 20 == 0:
            print(f"Step {step}: Debug sphere at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        # Slow down for visualization
        time.sleep(0.05)
        
        if done.any():
            obs = env.reset()
    
    print("\nTest without debug sphere (should be hidden)...")
    for step in range(50):
        action = torch.randn(1, 7, device="cuda:0") * 0.1
        # Step without extra_args - sphere should stay hidden
        obs, reward, done, info = env.step(action, None)
        time.sleep(0.05)
        if done.any():
            obs = env.reset()
    
    print("\nTest complete! The debug sphere should have:")
    print("1. Appeared as a gray sphere moving in a circle")
    print("2. NOT collided with any objects")
    print("3. Disappeared when extra_args=None")
    
    env.close()

if __name__ == "__main__":
    test_debug_sphere()