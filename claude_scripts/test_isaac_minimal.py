#!/usr/bin/env python3
"""
Minimal test of IsaacGymEnvs Cartpole
"""

import sys
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs')
import os

# CRITICAL: Import isaacgym BEFORE torch
import isaacgym

# NOW import torch after isaacgym
import torch

print("[TEST] Imports successful")

# Try to create Cartpole
try:
    from isaacgymenvs.tasks import isaacgym_task_map
    print(f"[TEST] Available tasks: {list(isaacgym_task_map.keys())}")
    
    # Minimal config
    import omegaconf
    cfg = {
        "task": {
            "name": "Cartpole",
            "physics_engine": "physx",
            "env": {
                "numEnvs": 512,
                "envSpacing": 4.0,
                "resetDist": 3.0,
                "maxEffort": 400.0,
                "clipObservations": 5.0,
                "clipActions": 1.0
            },
            "sim": {
                "dt": 0.0166,
                "substeps": 2,
                "up_axis": "z",
                "use_gpu_pipeline": True,
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "num_threads": 4,
                    "solver_type": 1,
                    "use_gpu": True,
                    "num_position_iterations": 4,
                    "num_velocity_iterations": 0,
                    "contact_offset": 0.02,
                    "rest_offset": 0.001,
                    "bounce_threshold_velocity": 0.2,
                    "max_depenetration_velocity": 100.0,
                    "default_buffer_size_multiplier": 5.0,
                    "max_gpu_contact_pairs": 8388608,
                    "num_subscenes": 4,
                    "contact_collection": 0
                }
            }
        },
        "physics_engine": "physx",
        "sim_device": "cuda:0",
        "rl_device": "cuda:0",
        "graphics_device_id": 0,
        "headless": True,
        "seed": 42
    }
    
    print("[TEST] Creating Cartpole environment...")
    
    # Convert to OmegaConf
    cfg_omega = omegaconf.OmegaConf.create(cfg)
    
    # Get task config
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    task_cfg = omegaconf_to_dict(cfg_omega.task)
    
    print(f"[TEST] Task config: {task_cfg['name']}")
    
    # Create environment with all required arguments
    env = isaacgym_task_map["Cartpole"](
        cfg=task_cfg,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=True,
        virtual_screen_capture=False,
        force_render=False
    )
    
    print(f"[TEST] Environment created! Num envs: {env.num_envs}")
    
    # Reset and get observation
    obs_dict = env.reset()
    print(f"[TEST] Reset successful! Observation type: {type(obs_dict)}")
    
    # Extract observation tensor
    if isinstance(obs_dict, dict):
        obs = obs_dict["obs"]
        print(f"[TEST] Observation shape: {obs.shape}")
    else:
        obs = obs_dict
        print(f"[TEST] Observation shape: {obs.shape}")
    
    # Take one step
    action = torch.zeros((env.num_envs, 1), device=obs.device)
    obs_dict, reward, done, info = env.step(action)
    print(f"[TEST] Step successful! Reward shape: {reward.shape}")
    
    print("[TEST] ✅ IsaacGymEnvs Cartpole is working!")
    
except Exception as e:
    import traceback
    print(f"[TEST] ❌ Error: {e}")
    traceback.print_exc()