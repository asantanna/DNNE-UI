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
                # Core PPO parameters
                "learning_rate": ("FLOAT", {
                    "default": 3e-4,
                    "min": 1e-6,
                    "max": 1e-2,
                    "step": 1e-6,
                    "tooltip": "Learning rate for policy and value networks. Lower = more stable, higher = faster learning."
                }),
                "clip_range": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.1,
                    "max": 0.4,
                    "step": 0.01,
                    "tooltip": "PPO clipping parameter. Controls how much the policy can change per update. 0.2 is standard."
                }),
                "value_loss_coef": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Coefficient for value function loss in combined objective."
                }),
                "entropy_coef": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Entropy bonus coefficient. Higher values encourage exploration."
                }),
                
                # Training configuration
                "n_steps": ("INT", {
                    "default": 2048,
                    "min": 128,
                    "max": 8192,
                    "tooltip": "Number of steps to collect per environment before update."
                }),
                "batch_size": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 512,
                    "tooltip": "Minibatch size for gradient updates."
                }),
                "n_epochs": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Number of epochs to train on collected data."
                }),
                
                # Advantage estimation
                "gamma": ("FLOAT", {
                    "default": 0.99,
                    "min": 0.9,
                    "max": 0.999,
                    "step": 0.001,
                    "tooltip": "Discount factor for future rewards. Higher = more long-term thinking."
                }),
                "gae_lambda": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.9,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "GAE lambda for advantage estimation. Balances bias vs variance."
                }),
                
                # Additional settings
                "max_grad_norm": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Maximum gradient norm for clipping. 0 disables clipping."
                }),
                "normalize_advantage": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize advantages to have mean=0, std=1. Generally improves stability."
                }),
                "use_clipped_value_loss": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use clipped value loss like in OpenAI's PPO implementation."
                })
            }
        }

    RETURN_TYPES = ("PPO_CONFIG",)
    RETURN_NAMES = ("config",)

    def create_config(self, **kwargs) -> Tuple[Dict[str, Any]]:
        """Create PPO configuration dictionary"""
        
        config = {
            # Core PPO parameters
            "learning_rate": kwargs["learning_rate"],
            "clip_range": kwargs["clip_range"],
            "value_loss_coef": kwargs["value_loss_coef"],
            "entropy_coef": kwargs["entropy_coef"],
            
            # Training configuration
            "n_steps": kwargs["n_steps"],
            "batch_size": kwargs["batch_size"],
            "n_epochs": kwargs["n_epochs"],
            
            # Advantage estimation
            "gamma": kwargs["gamma"],
            "gae_lambda": kwargs["gae_lambda"],
            
            # Additional settings
            "max_grad_norm": kwargs["max_grad_norm"],
            "normalize_advantage": kwargs["normalize_advantage"],
            "use_clipped_value_loss": kwargs["use_clipped_value_loss"],
            
            # Computed values
            "minibatch_size": kwargs["batch_size"],
            "num_minibatches": kwargs["n_steps"] // kwargs["batch_size"],
        }
        
        return (config,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PPOConfig": PPOConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PPOConfig": "PPO Config"
}