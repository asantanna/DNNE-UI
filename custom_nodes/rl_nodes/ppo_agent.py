"""
PPO Agent Node
Central node that consolidates PPO training configuration and executes training
"""

from typing import Dict, Tuple, Optional
from custom_nodes.robotics_nodes import RoboticsNodeBase


class PPOAgent(RoboticsNodeBase):
    """
    PPO Agent Node
    
    This is the central node that consolidates environment and training configuration
    from virtual nodes and exports as a complete PPO training script.
    """
    
    # NOT virtual - this node does the actual export
    IS_VIRTUAL = False
    
    CATEGORY = "rl"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env": ("ISAAC_ENV_CONFIG", {
                    "tooltip": "Environment configuration from IsaacGymEnvs node"
                }),
                "config": ("PPO_CONFIG", {
                    "tooltip": "PPO training configuration from PPO_Config node"
                }),
            },
            "optional": {
                # Balancing configuration
                "balancing_config": ("BALANCING_CONFIG", {
                    "tooltip": "Optional balancing configuration for performance targets"
                }),
                
                # Network architecture
                "network_mlp_layers": ("STRING", {
                    "default": "[256, 128, 64]",
                    "tooltip": "Hidden layer sizes for MLP (e.g., [256, 128, 64])"
                }),
                "network_activation": (["elu", "relu", "tanh", "selu"], {
                    "default": "elu",
                    "tooltip": "Activation function for network"
                }),
                "separate_value_network": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use separate networks for policy and value"
                }),
                
                # Checkpoint settings
                "checkpoint_interval": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Save checkpoint every N iterations (0 = disabled)"
                }),
                "keep_checkpoints": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of recent checkpoints to keep"
                }),
                "load_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "Path to checkpoint to load (empty = train from scratch)"
                }),
                
                # Logging settings
                "log_interval": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Log statistics every N iterations"
                }),
                "save_interval": ("INT", {
                    "default": 1000,
                    "min": 0,
                    "max": 100000,
                    "tooltip": "Save full model every N iterations"
                }),
                "experiment_name": ("STRING", {
                    "default": "PPO_DNNE",
                    "tooltip": "Name for experiment (used in logging)"
                }),
                
                # Training control
                "mixed_precision": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use mixed precision training (faster on newer GPUs)"
                }),
                "multi_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use multi-GPU training if available"
                }),
            }
        }
    
    RETURN_TYPES = ("RL_METRICS",)
    RETURN_NAMES = ("metrics",)
    FUNCTION = "train"
    OUTPUT_NODE = True  # This node produces training output
    
    def train(self, env, config, **kwargs):
        """
        This method is never actually called during normal operation.
        During export, this node generates a complete training script.
        """
        # Check for balancing configuration
        balancing_config = kwargs.get("balancing_config", None)
        if balancing_config and isinstance(balancing_config, dict):
            print(f"PPO Agent: Received balancing configuration - {balancing_config.get('node_name', 'Unnamed')}")
            if balancing_config.get("throughput", {}).get("target_percentage"):
                print(f"  - Target throughput: {balancing_config['throughput']['target_percentage']}%")
            if balancing_config.get("scheduling", {}).get("priority"):
                print(f"  - Priority: {balancing_config['scheduling']['priority']}")
        
        # In UI mode, just return dummy metrics
        metrics = {
            "status": "configured",
            "env_task": env.get("task", "Unknown"),
            "num_envs": env.get("num_envs", 0),
            "learning_rate": config.get("learning_rate", 0),
            "has_balancing": balancing_config is not None,
            "message": "PPO Agent configured. Export to generate training script."
        }
        
        return (metrics,)
    
    @classmethod
    def VALIDATE_INPUTS(cls, env, config, **kwargs):
        """Validate that environment and config are properly connected"""
        if not isinstance(env, dict) or "task" not in env:
            return "Invalid environment configuration"
        
        if not isinstance(config, dict) or "learning_rate" not in config:
            return "Invalid PPO configuration"
        
        # Validate network architecture string
        mlp_layers = kwargs.get("network_mlp_layers", "[256, 128, 64]")
        try:
            import ast
            layers = ast.literal_eval(mlp_layers)
            if not isinstance(layers, list) or not all(isinstance(x, int) for x in layers):
                return "network_mlp_layers must be a list of integers"
        except:
            return "Invalid network_mlp_layers format. Use e.g. [256, 128, 64]"
        
        return True
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """This node should re-export if any inputs change"""
        return float("nan")  # Always mark as changed