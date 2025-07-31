"""
Isaac Gym Environments
Provides GPU-accelerated physics simulation environments for reinforcement learning.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase
from custom_nodes.node_colors import get_node_colors

try:
    from .utils.isaac_gym_config_loader import IsaacGymConfigLoader
except ImportError:
    # Fallback
    class IsaacGymConfigLoader:
        def __init__(self):
            pass
        def get_available_tasks(self):
            return ["Cartpole", "Ant", "Humanoid", "Anymal", "BallBalance", "FrankaCabinet"]


class IsaacGymEnvs(RoboticsNodeBase):
    """Isaac Gym Environments
    Provides GPU-accelerated physics simulation environments for reinforcement learning."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("simulation")["color"] 
    BGCOLOR = get_node_colors("simulation")["bgcolor"]
    CATEGORY = "robotics"
    IS_VIRTUAL = True  # This is a virtual node

    def __init__(self):
        super().__init__()
        # Load available tasks
        try:
            loader = IsaacGymConfigLoader()
            self.available_tasks = loader.get_available_tasks()
        except Exception:
            # Fallback to common tasks if config loading fails
            self.available_tasks = ["Cartpole", "Ant", "Humanoid", "Anymal", "BallBalance", "FrankaCabinet"]

    @classmethod
    def INPUT_TYPES(cls):
        # Create instance to get available tasks
        temp_instance = cls()
        task_list = temp_instance.available_tasks
        
        return {
            "required": {
                "task": (task_list, {
                    "default": "Cartpole",
                    "tooltip": "Select an IsaacGymEnvs task - REQUIRED for export"
                }),
                "num_envs": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Number of parallel environments"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 1000000,
                    "tooltip": "Random seed for reproducibility"
                }),
                "control_after_generate": (["fixed", "randomize", "increment", "decrement"], {
                    "default": "fixed",
                    "tooltip": "How to handle seed between runs"
                }),
                "headless": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run in headless mode (no rendering)"
                }),
                "graphics_device_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 7,
                    "tooltip": "GPU device ID for rendering"
                }),
                "sim_device": ("STRING", {
                    "default": "cuda:0",
                    "tooltip": "Device for physics simulation (e.g., cuda:0, cpu)"
                }),
                "physics_engine": (["physx", "flex"], {
                    "default": "physx",
                    "tooltip": "Physics engine backend"
                }),
                "multi_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use multi-GPU simulation"
                }),
                "enable_cameras": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable camera sensors (impacts performance)"
                }),
            },
            "optional": {
                "force_render": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force rendering even in headless mode"
                }),
                "use_gpu_pipeline": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU pipeline for faster training"
                }),
                "num_threads": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "tooltip": "Number of CPU threads (0 = auto)"
                }),
                "solver_type": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 2,
                    "tooltip": "PhysX solver type (0=PGS, 1=TGS)"
                }),
                "num_subscenes": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32,
                    "tooltip": "Number of PhysX subscenes (0 = auto)"
                }),
            }
        }

    RETURN_TYPES = ("ISAAC_ENV_CONFIG",)
    RETURN_NAMES = ("env",)
    FUNCTION = "configure"

    def configure(self, task, num_envs, seed, control_after_generate, headless, graphics_device_id, sim_device, 
                  physics_engine, multi_gpu, enable_cameras, force_render=False, use_gpu_pipeline=True, 
                  num_threads=0, solver_type=1, num_subscenes=0):
        """
        This method is never actually called during export.
        It exists only to satisfy ComfyUI's node interface.
        """
        # Return configuration dict that would be used by PPO_Agent
        config = {
            "task": task,
            "num_envs": num_envs,
            "seed": seed,
            "control_after_generate": control_after_generate,
            "headless": headless,
            "graphics_device_id": graphics_device_id,
            "sim_device": sim_device,
            "physics_engine": physics_engine,
            "multi_gpu": multi_gpu,
            "enable_cameras": enable_cameras,
            "force_render": force_render,
            "use_gpu_pipeline": use_gpu_pipeline,
            "num_threads": num_threads,
            "solver_type": solver_type,
            "num_subscenes": num_subscenes
        }
        
        return (config,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "IsaacGymEnvs": IsaacGymEnvs
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IsaacGymEnvs": "Isaac Gym Environment"
}