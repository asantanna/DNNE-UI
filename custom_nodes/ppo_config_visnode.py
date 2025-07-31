"""
PPO Config
Configuration node for Proximal Policy Optimization algorithm parameters.
"""

import torch
from typing import Dict, Any, Tuple
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase
from custom_nodes.node_colors import get_node_colors


class PPOConfig(RoboticsNodeBase):
    """PPO Config
    Configuration node for Proximal Policy Optimization algorithm parameters."""
    
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "create_config"
    CATEGORY = "rl"
    IS_VIRTUAL = True  # Configuration-only node
    COLOR = get_node_colors("utility")["color"]
    BGCOLOR = get_node_colors("utility")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Core PPO hyperparameters
                "learning_rate": ("FLOAT", {
                    "default": 3e-4,
                    "min": 1e-6,
                    "max": 1e-1,
                    "step": 1e-6,
                    "tooltip": "Learning rate for policy and value networks"
                }),
                "num_epochs": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Number of PPO epochs per update"
                }),
                "minibatch_size": ("INT", {
                    "default": 8192,
                    "min": 32,
                    "max": 32768,
                    "tooltip": "Size of each minibatch for gradient updates"
                }),
                "clip_param": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "PPO clipping parameter epsilon"
                }),
                "value_loss_coef": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Value function loss coefficient"
                }),
                "entropy_coef": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Entropy bonus coefficient"
                }),
                "gamma": ("FLOAT", {
                    "default": 0.99,
                    "min": 0.8,
                    "max": 0.9999,
                    "step": 0.001,
                    "tooltip": "Discount factor"
                }),
                "gae_lambda": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.8,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "GAE lambda parameter"
                }),
                "max_grad_norm": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Maximum gradient norm for clipping"
                }),
            },
            "optional": {
                # Training duration
                "horizon_length": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 4096,
                    "tooltip": "Rollout horizon length (steps per environment)"
                }),
                "max_iterations": ("INT", {
                    "default": 10000,
                    "min": 1,
                    "max": 1000000,
                    "tooltip": "Maximum training iterations"
                }),
                
                # Learning rate schedule
                "lr_schedule": (["constant", "linear", "adaptive"], {
                    "default": "constant",
                    "tooltip": "Learning rate schedule type"
                }),
                "lr_schedule_kl_threshold": ("FLOAT", {
                    "default": 0.008,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "KL threshold for adaptive learning rate"
                }),
                
                # Advanced PPO settings
                "use_clipped_value_loss": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use clipped value function loss"
                }),
                "normalize_advantage": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize advantages"
                }),
                "normalize_input": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize observations with running statistics"
                }),
                "normalize_value": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize value targets"
                }),
                
                # Misc settings
                "reward_shaper_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "tooltip": "Scale factor for reward shaping"
                }),
                "e_clip": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "PPO dual clip parameter (0 = disabled)"
                }),
                "truncate_grads": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Truncate gradients"
                }),
                "bounds_loss_coef": ("FLOAT", {
                    "default": 0.0001,
                    "min": 0.0,
                    "max": 0.01,
                    "step": 0.0001,
                    "tooltip": "Coefficient for bounds loss term"
                })
            }
        }

    RETURN_TYPES = ("PPO_CONFIG",)
    RETURN_NAMES = ("config",)

    def create_config(self, **kwargs) -> Tuple[Dict[str, Any]]:
        """
        This method is never actually called during export.
        It exists only to satisfy ComfyUI's node interface.
        """
        config = {
            # Core PPO settings
            "learning_rate": kwargs.get("learning_rate", 3e-4),
            "epochs": kwargs.get("num_epochs", 4),
            "minibatch_size": kwargs.get("minibatch_size", 8192),
            "e_clip": kwargs.get("clip_param", 0.2),
            "critic_coef": kwargs.get("value_loss_coef", 0.5),
            "entropy_coef": kwargs.get("entropy_coef", 0.01),
            "gamma": kwargs.get("gamma", 0.99),
            "tau": kwargs.get("gae_lambda", 0.95),
            "grad_norm": kwargs.get("max_grad_norm", 0.5),
            
            # Optional settings
            "horizon_length": kwargs.get("horizon_length", 16),
            "max_epochs": kwargs.get("max_iterations", 10000),
            "lr_schedule": kwargs.get("lr_schedule", "constant"),
            "kl_threshold": kwargs.get("lr_schedule_kl_threshold", 0.008),
            "clip_value": kwargs.get("use_clipped_value_loss", True),
            "normalize_advantage": kwargs.get("normalize_advantage", True),
            "normalize_input": kwargs.get("normalize_input", True),
            "normalize_value": kwargs.get("normalize_value", True),
            "reward_shaper": {"scale_value": kwargs.get("reward_shaper_scale", 1.0)},
            "bounds_loss_coef": kwargs.get("bounds_loss_coef", 0.0001),
            "truncate_grads": kwargs.get("truncate_grads", True),
        }
        
        return (config,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PPOConfig": PPOConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PPOConfig": "PPO Config"
}