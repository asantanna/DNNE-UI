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
                    "tooltip": "Isaac Gym task/environment to use. Each task has different observation and action spaces."
                }),
                "num_envs": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Number of parallel environments. More envs = more parallel data but higher GPU memory usage."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "tooltip": "Random seed for reproducibility. Use -1 for random seed."
                }),
                "device": (["cuda:0", "cuda:1", "cpu"], {
                    "default": "cuda:0",
                    "tooltip": "Device to run simulation on. CUDA strongly recommended for performance."
                }),
                "headless": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run without rendering window. Set False to see visualization (slower)."
                }),
                "force_render": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force rendering even in headless mode (for recording/debugging)."
                }),
                "eval_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Evaluation mode: deterministic actions, no exploration noise."
                })
            },
            "optional": {
                "custom_config": ("DICT", {
                    "tooltip": "Custom configuration to override task defaults"
                })
            }
        }

    RETURN_TYPES = ("GYM_CONFIG",)
    RETURN_NAMES = ("env_config",)
    FUNCTION = "configure_env"

    def configure_env(self, task, num_envs, seed, device, headless, force_render, eval_mode, custom_config=None):
        """Configure Isaac Gym environment parameters"""
        
        # Create configuration
        env_config = {
            "task": task,
            "num_envs": num_envs,
            "seed": seed if seed >= 0 else None,
            "device": device,
            "headless": headless,
            "force_render": force_render,
            "eval_mode": eval_mode,
            "isaac_gym_envs_path": str(get_isaac_gym_envs_path()),
            "sim_device": f"cuda:{device.split(':')[-1]}" if device.startswith("cuda") else "cpu",
            "graphics_device_id": int(device.split(':')[-1]) if device.startswith("cuda") else 0,
        }
        
        # Merge custom config if provided
        if custom_config:
            env_config.update(custom_config)
        
        # Add task-specific defaults
        task_defaults = self._get_task_defaults(task)
        for key, value in task_defaults.items():
            if key not in env_config:
                env_config[key] = value
        
        return (env_config,)

    def _get_task_defaults(self, task: str) -> Dict[str, Any]:
        """Get default configuration for specific tasks"""
        defaults = {
            "Cartpole": {
                "physics_engine": "physx",
                "num_threads": 0,
                "solver_type": 1,
                "use_gpu_pipeline": True,
                "up_axis": "z",
                "dt": 0.01667,  # 60 Hz
            },
            "Ant": {
                "physics_engine": "physx",
                "num_threads": 0,
                "solver_type": 1,
                "use_gpu_pipeline": True,
                "up_axis": "z",
                "dt": 0.01667,
            },
            "Humanoid": {
                "physics_engine": "physx",
                "num_threads": 4,
                "solver_type": 1,
                "use_gpu_pipeline": True,
                "up_axis": "z",
                "dt": 0.0083,  # 120 Hz for stability
            }
        }
        
        return defaults.get(task, defaults["Cartpole"])

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Virtual nodes don't need to track changes
        return False


# Node registration
NODE_CLASS_MAPPINGS = {
    "IsaacGymEnvs": IsaacGymEnvs
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IsaacGymEnvs": "Isaac Gym Environment"
}