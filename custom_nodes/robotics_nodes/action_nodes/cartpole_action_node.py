# cartpole_action_node.py
"""
Cartpole Action Node
Converts neural network output to Isaac Gym ACTION format for Cartpole environment
"""

import torch
from typing import Dict
from inspect import cleandoc

# Import base classes and types
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_node import LearningNodeBase
from robotics_types import Action


class CartpoleActionNode(LearningNodeBase):
    """
    Cartpole Action Node
    Converts neural network output to Isaac Gym ACTION format for Cartpole environment
    """
    
    CATEGORY = "robotics/actions"
    
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
    
    def __init__(self):
        super().__init__()
        
    def convert_to_action(self, policy: dict, max_push_effort: float):
        """
        Convert PPO policy output to Isaac Gym ACTION format for Cartpole
        
        Args:
            policy: PolicyOutput dictionary containing action tensor
            max_push_effort: Maximum force scaling factor
            
        Returns:
            ACTION: Properly formatted action for IsaacGymStepNode
        """
        
        # Extract action tensor from PolicyOutput
        action_tensor = policy["action"]
        
        # Ensure action_tensor is properly shaped
        if action_tensor.dim() > 1:
            action_tensor = action_tensor.squeeze()
        
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)
            
        # Scale by max effort (same as IsaacGym Cartpole implementation)
        scaled_force = action_tensor[0] * max_push_effort
        
        # For Cartpole: 2 DOF (cart translation, pole rotation)
        # Only cart (DOF 0) is actuated, pole (DOF 1) is passive
        forces = torch.zeros(2, dtype=torch.float32, device=action_tensor.device)
        forces[0] = scaled_force  # Apply force to cart only
        
        # Create ACTION object
        action = Action(
            forces=forces,
            joint_commands=None,  # Not used for Cartpole
            torques=None          # Not used for Cartpole
        )
        
        # Note: Action force logged for debugging
        # Could add logging here if needed
        
        return (action,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute when inputs change"""
        return float("inf")  # Always changed
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs"""
        network_output = kwargs.get("network_output")
        if network_output is not None and not isinstance(network_output, torch.Tensor):
            return False
        return True