#!/usr/bin/env python3
"""Test the Isaac Gym config loader to see why tasks aren't loading"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_nodes.utils import IsaacGymEnvConfigLoader

print("Testing Isaac Gym Config Loader...")
print("-" * 60)

# Create loader instance
loader = IsaacGymEnvConfigLoader.get_instance()
print(f"Loader created: {loader}")
print(f"Base path: {loader.base_path}")
print(f"Task config path: {loader.task_cfg_path}")
print(f"Train config path: {loader.train_cfg_path}")

# Check if paths exist
print(f"\nTask path exists: {loader.task_cfg_path.exists()}")
print(f"Train path exists: {loader.train_cfg_path.exists()}")

# Get available tasks
print("\nGetting available tasks...")
tasks = loader.get_available_tasks()
print(f"Found {len(tasks)} tasks: {tasks}")

# Check cache
print(f"\nConfigs cache: {loader._configs_cache is not None}")
if loader._configs_cache is not None:
    print(f"Cache size: {len(loader._configs_cache)}")

# Try loading a specific task
print("\nTrying to load Cartpole config...")
config = loader.get_task_config("Cartpole")
print(f"Config loaded: {bool(config)}")

# Check the actual AVAILABLE_TASKS that would be used
print("\nChecking AVAILABLE_TASKS as used in isaac_gym_envs.py...")
AVAILABLE_TASKS = ["none"] + loader.get_available_tasks()
print(f"AVAILABLE_TASKS: {AVAILABLE_TASKS}")
print(f"Length: {len(AVAILABLE_TASKS)}")