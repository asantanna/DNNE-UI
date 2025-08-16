"""
PPO Agent
Complete Proximal Policy Optimization agent implementation for reinforcement learning.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors
from custom_nodes.utils.dnne_decorator import dnne_node


@dnne_node(is_virtual=False)
class PPOAgent(RoboticsNodeBase):
    """PPO Agent
    Complete Proximal Policy Optimization agent implementation for reinforcement learning."""
    
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "rl"
    COLOR = get_node_colors("rl")["color"]
    BGCOLOR = get_node_colors("rl")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env_config": ("ISAAC_ENV_CONFIG_PYDICT", {
                    "tooltip": "Environment configuration from IsaacGymEnvs node"
                }),
                "ppo_config": ("PPO_CONFIG_PYDICT", {
                    "tooltip": "PPO algorithm configuration from PPOConfig node"
                }),
                "max_iterations": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 1000000,
                    "tooltip": "Maximum number of training iterations"
                }),
                "checkpoint_interval": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Save checkpoint every N iterations. 0 disables checkpointing."
                }),
                "eval_interval": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Evaluate agent every N iterations. 0 disables evaluation."
                }),
                "eval_episodes": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of episodes for evaluation"
                }),
                "log_interval": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Log training statistics every N iterations"
                }),
                "save_path": ("STRING", {
                    "default": "./ppo_checkpoints",
                    "tooltip": "Directory to save checkpoints and logs"
                })
            },
            "optional": {
                "resume_from": ("STRING", {
                    "default": "",
                    "tooltip": "Path to checkpoint to resume training from"
                }),
                "balancing_config": ("BALANCING_CONFIG_PYDICT", {
                    "tooltip": "Optional balancing configuration from BalancerConfig node"
                })
            }
        }

    RETURN_TYPES = ("PPO_AGENT_OBJ", "PPO_TRAINING_STATS_PYDICT", "PPO_EVAL_STATS_PYDICT")
    RETURN_NAMES = ("agent", "training_stats", "eval_stats")


# Node registration
NODE_CLASS_MAPPINGS = {
    "PPOAgent": PPOAgent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PPOAgent": "PPO Agent"
}