# robotics_nodes/__init__.py
"""
DNNE Robotics Nodes Package
This initializes all robotics-specific nodes and types for the DNNE system
"""

# Import all the types and base classes
from .robotics_types import *
from .base_node import *

# Import specific node implementations as you create them
# from .sensor_nodes import *
# from .controller_nodes import *
# from .isaac_gym_base_nodes import *  # Old nodes - commented out
# from .cartpole_action_node import CartpoleActionNode  # Old node - commented out
from .isaac_gym_envs import IsaacGymEnvs

# Define robotics node data as tuples (key, class, display_name)
# Future nodes: just add to this list and they'll be automatically sorted alphabetically
_ROBOTICS_NODES = [
    ("IsaacGymEnvs", IsaacGymEnvs, "Isaac Gym Environments"),
    # Future nodes to be implemented:
    # ("IsaacGymEnvNode_OLD", IsaacGymEnvNode_OLD, "Isaac Gym Environment (OLD)"),
    # ("IsaacGymStepNode", IsaacGymStepNode, "Isaac Gym Step"),
    # ("CartpoleActionNode", CartpoleActionNode, "Cartpole Action Converter"),
    # ("CartpoleRewardNode", CartpoleRewardNode, "Cartpole Reward Calculator"),
    # ("RoboticsCameraNode", CameraNode, "Camera Sensor"),
]

# Generate sorted dictionaries automatically by display name
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for key, node_class, display_name in sorted(_ROBOTICS_NODES, key=lambda x: x[2]):  # Sort by display name
    NODE_CLASS_MAPPINGS[key] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[key] = display_name

# Export the mappings for ComfyUI to discover
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Initialize robotics types when the module is imported
print("Initializing DNNE Robotics Nodes...")
register_robotics_types()
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} robotics nodes")