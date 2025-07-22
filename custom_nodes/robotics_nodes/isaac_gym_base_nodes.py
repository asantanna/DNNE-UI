# isaac_gym_base_nodes.py
"""
Isaac Gym nodes for DNNE - UI ONLY
These nodes are for the visual editor interface only. They do NOT execute.
Actual Isaac Gym functionality is implemented in the export templates.
"""

import torch
from typing import Dict, Optional
from inspect import cleandoc
from .base_node import LearningNodeBase


class IsaacGymEnvNode(LearningNodeBase):
    """
    Isaac Gym Environment Node (UI Only)
    Sets up Isaac Gym environment parameters for export.
    This node does NOT execute - it only defines the UI interface.
    """
    
    CATEGORY = "robotics"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env_name": ("STRING", {
                    "default": "Cartpole",
                    "multiline": False,
                    "tooltip": "Isaac Gym environment name (e.g., Cartpole, Ant, Humanoid)"
                }),
                "num_envs": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Number of parallel environments"
                }),
                "headless": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run in headless mode (no GUI)"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for simulation"
                }),
            },
            "optional": {}
        }
    
    RETURN_TYPES = ("ENV_HANDLE", "TENSOR")
    RETURN_NAMES = ("env_handle", "observations")
    FUNCTION = "setup_environment"
    DESCRIPTION = cleandoc(__doc__)
    
    def setup_environment(self, env_name: str, num_envs: int, headless: bool, device: str):
        """
        UI-only function that returns dummy data.
        Actual implementation is in export templates.
        """
        # Return dummy data for UI connections
        # The actual Isaac Gym environment is created in the exported code
        dummy_handle = {"type": "ENV_HANDLE", "env_name": env_name, "num_envs": num_envs}
        dummy_observations = torch.zeros(num_envs, 4)  # Dummy observation shape
        
        return (dummy_handle, dummy_observations)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute to ensure fresh simulation state"""
        return float("inf")


class IsaacGymStepNode(LearningNodeBase):
    """
    Isaac Gym Step Node (UI Only)
    Steps the Isaac Gym simulation with actions.
    This node does NOT execute - it only defines the UI interface.
    """
    
    CATEGORY = "robotics"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env_handle": ("ENV_HANDLE",),
                "actions": ("TENSOR",),
            },
            "optional": {
                "trigger": ("SYNC",),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "DICT", "TENSOR")
    RETURN_NAMES = ("observations", "rewards", "done", "info", "next_observations")
    FUNCTION = "step_simulation"
    DESCRIPTION = cleandoc(__doc__)
    
    def step_simulation(self, env_handle: Dict, actions: torch.Tensor, trigger: Optional[Dict] = None):
        """
        UI-only function that returns dummy data.
        Actual implementation is in export templates.
        """
        # Extract num_envs from env_handle if available
        num_envs = env_handle.get("num_envs", 1) if isinstance(env_handle, dict) else 1
        
        # Return dummy data for UI connections
        dummy_observations = torch.zeros(num_envs, 4)
        dummy_rewards = torch.zeros(num_envs)
        dummy_done = torch.zeros(num_envs, dtype=torch.bool)
        dummy_info = {}
        dummy_next_observations = torch.zeros(num_envs, 4)
        
        return (dummy_observations, dummy_rewards, dummy_done, dummy_info, dummy_next_observations)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute when inputs change"""
        return float("inf")


# Export the node classes
__all__ = ['IsaacGymEnvNode', 'IsaacGymStepNode']