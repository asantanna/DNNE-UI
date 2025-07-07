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
from .isaac_gym_nodes import *

# This is the standard ComfyUI way to register custom nodes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register the example node from base_node.py
NODE_CLASS_MAPPINGS["RoboticsExampleIMU"] = ExampleSensorNode
NODE_DISPLAY_NAME_MAPPINGS["RoboticsExampleIMU"] = "IMU Sensor"

# Register Isaac Gym nodes
NODE_CLASS_MAPPINGS["IsaacGymEnvNode"] = IsaacGymEnvNode
NODE_DISPLAY_NAME_MAPPINGS["IsaacGymEnvNode"] = "Isaac Gym Environment"

NODE_CLASS_MAPPINGS["IsaacGymStepNode"] = IsaacGymStepNode
NODE_DISPLAY_NAME_MAPPINGS["IsaacGymStepNode"] = "Isaac Gym Step"

NODE_CLASS_MAPPINGS["ORNode"] = ORNode
NODE_DISPLAY_NAME_MAPPINGS["ORNode"] = "OR/ANY Router"

# Register Cartpole-specific nodes
NODE_CLASS_MAPPINGS["CartpoleActionNode"] = CartpoleActionNode
NODE_DISPLAY_NAME_MAPPINGS["CartpoleActionNode"] = "Cartpole Action Converter"

NODE_CLASS_MAPPINGS["CartpoleRewardNode"] = CartpoleRewardNode
NODE_DISPLAY_NAME_MAPPINGS["CartpoleRewardNode"] = "Cartpole Reward Calculator"

# As you create more nodes, add them here:
# NODE_CLASS_MAPPINGS["RoboticsCameraNode"] = CameraNode
# NODE_DISPLAY_NAME_MAPPINGS["RoboticsCameraNode"] = "Camera Sensor"

# Export the mappings for ComfyUI to discover
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Initialize robotics types when the module is imported
print("Initializing DNNE Robotics Nodes...")
register_robotics_types()
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} robotics nodes")