# cartpole_action_node.py
"""
Cartpole Action Node - UI ONLY
Converts neural network output to Isaac Gym ACTION format for Cartpole environment.
This node does NOT execute - it only defines the UI interface.
"""

import torch
from typing import Dict
from inspect import cleandoc
from .base_node import LearningNodeBase
from .robotics_types import Action


class CartpoleActionNode(LearningNodeBase):
    """
    Cartpole Action Node (UI Only)
    Converts PPO policy output to ACTION format for Cartpole.
    This node does NOT execute - actual implementation is in export templates.
    """
    
    CATEGORY = "robotics"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "policy": ("POLICY",),
                "max_push_effort": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Maximum force that can be applied to cart"
                }),
            },
            "optional": {}
        }
    
    RETURN_TYPES = ("ACTION",)
    RETURN_NAMES = ("action",)
    FUNCTION = "convert_to_action"
    DESCRIPTION = cleandoc(__doc__)
    
    def convert_to_action(self, policy: dict, max_push_effort: float):
        """
        UI-only function that returns dummy data.
        Actual implementation is in export templates.
        """
        # Return dummy action for UI connections
        dummy_action = Action(
            forces=torch.zeros(2),  # Cartpole has 2 DOF
            joint_commands=None,
            torques=None
        )
        
        return (dummy_action,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute when inputs change"""
        return float("inf")