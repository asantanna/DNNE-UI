# rl_nodes/__init__.py
"""
DNNE Reinforcement Learning Nodes Package
This package contains RL algorithm implementations for DNNE
"""

# Import RL types and base classes
from .rl_types import *

# Import old node implementations for compatibility
# from .ppo_agent_OLD import PPOAgentNode  # Commented out - new version in ml_nodes
# from .ppo_trainer_OLD import PPOTrainerNode  # Commented out - new version in ml_nodes

# Define RL node data as tuples (key, class, display_name)
# Future nodes: just add to this list and they'll be automatically sorted alphabetically
_RL_NODES = [
    # Old nodes commented out - new PPO nodes are in ml_nodes package
    # ("PPOAgentNode_OLD", PPOAgentNode, "PPO Agent (Actor-Critic) OLD"),
    # ("PPOTrainerNode_OLD", PPOTrainerNode, "PPO Trainer OLD"),
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