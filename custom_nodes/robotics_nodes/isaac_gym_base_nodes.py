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


class IsaacGymEnvNode_OLD(LearningNodeBase):
    """
    Isaac Gym Environment Node (UI Only) - OLD VERSION
    Being replaced by new IsaacGymEnvs virtual node
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
    
    RETURN_TYPES = ("ENV_HANDLE", "TENSOR", "CONTEXT")
    RETURN_NAMES = ("env_handle", "observations", "context")
    FUNCTION = "create_env"
    CATEGORY = "robotics"
    DESCRIPTION = cleandoc(__doc__)
    
    def create_env(self, env_name: str, num_envs: int, headless: bool, device: str):
        """UI placeholder - returns dummy data for connections"""
        # Create dummy environment handle
        env_handle = {
            "env_name": env_name,
            "num_envs": num_envs,
            "headless": headless,
            "device": device,
            "initialized": False  # Flag for export system
        }
        
        # Create dummy observations (UI placeholder)
        dummy_observations = torch.zeros(num_envs, 4)  # Simplified observation space
        
        # Create dummy context
        from .robotics_types import Context
        context = Context()
        
        return (env_handle, dummy_observations, context)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Environment setup can change between runs"""
        return float("inf")


class IsaacGymStepNode(LearningNodeBase):
    """
    Isaac Gym Step Node (UI Only)
    Steps the Isaac Gym simulation forward by one timestep.
    This node is for UI connections only - actual execution happens in export.
    """
    
    CATEGORY = "robotics"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env_handle": ("ENV_HANDLE", {
                    "tooltip": "Environment handle from IsaacGymEnvNode"
                }),
                "actions": ("TENSOR", {
                    "tooltip": "Action tensor to apply to environments"
                }),
            },
            "optional": {
                "trigger": ("SYNC", {
                    "tooltip": "Optional trigger for synchronized execution"
                })
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "DICT", "TENSOR")
    RETURN_NAMES = ("observations", "rewards", "done", "info", "next_observations")
    FUNCTION = "step"
    CATEGORY = "robotics"
    DESCRIPTION = cleandoc(__doc__)
    
    def step(self, env_handle: Dict, actions: torch.Tensor, trigger=None):
        """
        UI placeholder - returns dummy data for connections.
        The real implementation is in the export template.
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


# Export the node classes - create alias for compatibility
IsaacGymEnvNode = IsaacGymEnvNode_OLD
__all__ = ['IsaacGymEnvNode', 'IsaacGymEnvNode_OLD', 'IsaacGymStepNode']