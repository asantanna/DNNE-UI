"""
Isaac Gym Environments Virtual Node
Configuration-only node for Isaac Gym environment settings
"""

from typing import Dict, Tuple, Optional
from .base_node import RoboticsNodeBase


class IsaacGymEnvs(RoboticsNodeBase):
    """
    Isaac Gym Environments Configuration Node
    
    Virtual node that provides environment configuration for PPO training.
    This node is skipped during export - its settings are used by PPO_Agent.
    """
    
    # Mark as virtual - this node only provides configuration
    IS_VIRTUAL = True
    
    CATEGORY = "rl"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": ("STRING", {
                    "default": "Cartpole",
                    "multiline": False,
                    "tooltip": "IsaacGymEnvs task name (e.g., Cartpole, Ant, Humanoid, Anymal)"
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
    
    def configure(self, **kwargs):
        """
        This method is never actually called during export.
        It exists only to satisfy ComfyUI's node interface.
        """
        # Return configuration dict that would be used by PPO_Agent
        config = {
            "task": kwargs.get("task", "Cartpole"),
            "num_envs": kwargs.get("num_envs", 64),
            "seed": kwargs.get("seed", 42),
            "headless": kwargs.get("headless", True),
            "graphics_device_id": kwargs.get("graphics_device_id", 0),
            "sim_device": kwargs.get("sim_device", "cuda:0"),
            "physics_engine": kwargs.get("physics_engine", "physx"),
            "multi_gpu": kwargs.get("multi_gpu", False),
            "enable_cameras": kwargs.get("enable_cameras", False),
            "force_render": kwargs.get("force_render", False),
            "use_gpu_pipeline": kwargs.get("use_gpu_pipeline", True),
            "num_threads": kwargs.get("num_threads", 0),
            "solver_type": kwargs.get("solver_type", 1),
            "num_subscenes": kwargs.get("num_subscenes", 0),
        }
        
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        return (config,)
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate input values"""
        # Check sim_device format
        sim_device = kwargs.get("sim_device", "cuda:0")
        if not (sim_device == "cpu" or sim_device.startswith("cuda:")):
            return f"Invalid sim_device: {sim_device}. Use 'cpu' or 'cuda:X'"
        
        # Check that task name is valid (basic check)
        task = kwargs.get("task", "")
        if not task or not task[0].isupper():
            return f"Task name should start with uppercase: {task}"
        
        return True