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
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors
from custom_nodes.utils.dnne_decorator import dnne_node

from .utils.isaac_gym_config_loader import IsaacGymEnvConfigLoader as IsaacGymConfigLoader


@dnne_node(is_virtual=True)
class IsaacGymEnvsNode(RoboticsNodeBase):
    """Isaac Gym Environments
    Provides GPU-accelerated physics simulation environments for reinforcement learning."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("utility")["color"] 
    BGCOLOR = get_node_colors("utility")["bgcolor"]
    CATEGORY = "robotics"

    def __init__(self):
        super().__init__()
        # Load available tasks - fail fast if config loading fails
        loader = IsaacGymConfigLoader()
        self.available_tasks = loader.get_available_tasks()

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
                "subtask": ("STRING", {
                    "default": "random_target",
                    "tooltip": "Subtask for DNNE environment",
                    "dnne_only": True  # Only show for DNNE environments
                }),
                "dt": ("FLOAT", {
                    "default": 0.01667,  # 60 Hz default
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Simulation timestep (seconds)",
                    "dnne_only": True  # Only show/editable for DNNE environments
                }),
                "num_envs": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Number of parallel environments",
                    # This widget will be hidden for DNNE environments via UI
                    "dnne_hide": True  # Custom metadata for UI to handle
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 1000000,
                    "tooltip": "Random seed for reproducibility"
                }),
                "seed_control": (["fixed", "randomize", "increment", "decrement"], {
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

    RETURN_TYPES = ("ISAAC_ENV_CONFIG_PYDICT",)
    RETURN_NAMES = ("env",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs for DNNE environments."""
        task = kwargs.get("task")
        if not task:
            return "Task selection is required"
            
        # Get loader instance
        loader = IsaacGymConfigLoader()
        
        # Check if this is a DNNE environment
        if loader.is_dnne_environment(task):
            # DNNE environments must have num_envs=1
            num_envs = kwargs.get("num_envs", 1)
            if num_envs != 1:
                return f"DNNE environment '{task}' must use num_envs=1, got {num_envs}"
                
            # Check if subtask is valid
            subtask = kwargs.get("subtask")
            if subtask:
                available_subtasks = loader.get_task_subtasks(task)
                if available_subtasks and subtask not in available_subtasks:
                    return f"Invalid subtask '{subtask}' for {task}. Available: {available_subtasks}"
        
        return True
    
    @classmethod
    def IS_DNNE_ENVIRONMENT(cls, task_name):
        """Check if a task is a DNNE environment."""
        loader = IsaacGymConfigLoader()
        return loader.is_dnne_environment(task_name)


# Node registration
NODE_CLASS_MAPPINGS = {
    "IsaacGymEnvs": IsaacGymEnvsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IsaacGymEnvs": "Isaac Gym Environment Config"
}