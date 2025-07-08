# rl_nodes/__init__.py
"""
DNNE Reinforcement Learning Nodes Package
This package contains RL algorithm implementations for DNNE
"""

# Import RL types and base classes
from .rl_types import *

# Import node implementations
from .ppo_agent import PPOAgentNode
from .ppo_trainer import PPOTrainerNode

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register PPO nodes
NODE_CLASS_MAPPINGS["PPOAgentNode"] = PPOAgentNode
NODE_DISPLAY_NAME_MAPPINGS["PPOAgentNode"] = "PPO Agent (Actor-Critic)"

NODE_CLASS_MAPPINGS["PPOTrainerNode"] = PPOTrainerNode  
NODE_DISPLAY_NAME_MAPPINGS["PPOTrainerNode"] = "PPO Trainer"

# Export the mappings for ComfyUI to discover
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Initialize RL types when the module is imported
print("Initializing DNNE RL Nodes...")
register_rl_types()
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} RL nodes")