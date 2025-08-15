"""
PPO Config
Configuration node for Proximal Policy Optimization algorithm parameters.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class PPOConfig(RoboticsNodeBase):
    """PPO Config
    Configuration node for Proximal Policy Optimization algorithm parameters."""
    
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
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

    RETURN_TYPES = ("PPO_CONFIG_PYDICT",)
    RETURN_NAMES = ("config",)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PPOConfig": PPOConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PPOConfig": "PPO Config"
}