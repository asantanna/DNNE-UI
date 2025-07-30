"""
Cartpole Action Node
Converts neural network outputs to discrete actions for the Cartpole environment.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from inspect import cleandoc
from custom_nodes.base import LearningNodeBase
from custom_nodes.node_colors import get_node_colors


class CartpoleActionNode(LearningNodeBase):
    """Cartpole Action Node
    Converts neural network outputs to discrete actions for the Cartpole environment."""
    
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "compute_action"
    CATEGORY = "robotics/control"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network_output": ("TENSOR", {
                    "tooltip": "Output from neural network. Can be logits or action values."
                }),
                "exploration_noise": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Amount of exploration noise to add (epsilon for epsilon-greedy)"
                }),
                "deterministic": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, always select best action (no exploration)"
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "DICT")
    RETURN_NAMES = ("actions", "action_info")
    COLOR = get_node_colors("actuator")["color"]
    BGCOLOR = get_node_colors("actuator")["bgcolor"]

    def compute_action(self, network_output: torch.Tensor, exploration_noise: float, 
                      deterministic: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Convert network output to discrete Cartpole actions
        
        Cartpole has 2 discrete actions:
        - 0: Push cart to the left
        - 1: Push cart to the right
        """
        
        device = network_output.device
        batch_size = network_output.shape[0]
        
        if deterministic or exploration_noise == 0:
            # Greedy action selection
            actions = torch.argmax(network_output, dim=-1)
        else:
            # Epsilon-greedy exploration
            random_actions = torch.randint(0, 2, (batch_size,), device=device)
            greedy_actions = torch.argmax(network_output, dim=-1)
            
            # Random mask for exploration
            explore_mask = torch.rand(batch_size, device=device) < exploration_noise
            actions = torch.where(explore_mask, random_actions, greedy_actions)
        
        # Prepare action info
        action_probs = torch.softmax(network_output, dim=-1)
        action_info = {
            "raw_output": network_output,
            "action_probs": action_probs,
            "selected_probs": action_probs.gather(1, actions.unsqueeze(1)).squeeze(1),
            "exploration_rate": exploration_noise if not deterministic else 0.0,
            "deterministic": deterministic
        }
        
        return actions, action_info

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Actions depend on network output and exploration settings
        return True

# Node registration
NODE_CLASS_MAPPINGS = {
    "CartpoleAction": CartpoleActionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CartpoleAction": "Cartpole Action"
}