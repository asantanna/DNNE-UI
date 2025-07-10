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

# Define RL node data as tuples (key, class, display_name)
# Future nodes: just add to this list and they'll be automatically sorted alphabetically
_RL_NODES = [
    ("PPOAgentNode", PPOAgentNode, "PPO Agent (Actor-Critic)"),
    ("PPOTrainerNode", PPOTrainerNode, "PPO Trainer"),
    # Add new RL nodes here - they'll be automatically sorted by display name
]

# Generate sorted dictionaries automatically by display name
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for key, node_class, display_name in sorted(_RL_NODES, key=lambda x: x[2]):  # Sort by display name
    NODE_CLASS_MAPPINGS[key] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[key] = display_name

# Export the mappings for ComfyUI to discover
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Initialize RL types when the module is imported
print("Initializing DNNE RL Nodes...")
register_rl_types()
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} RL nodes")