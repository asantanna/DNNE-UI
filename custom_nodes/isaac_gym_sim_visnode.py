"""
Isaac Gym Simulator Interface
Queue-based interface to Isaac Gym environments for real-time simulation.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class IsaacGymSimNode(RoboticsNodeBase):
    """Isaac Gym Simulator Interface
    Queue-based interface to Isaac Gym environments for real-time simulation."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("simulation")["color"]
    BGCOLOR = get_node_colors("simulation")["bgcolor"]
    CATEGORY = "robotics"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env_config": ("ISAAC_ENV_CONFIG", {
                    "tooltip": "Environment configuration from Isaac Gym Environment Config node"
                }),
                "action": ("TENSOR", {
                    "tooltip": "Actions to execute in the environment"
                }),
            },
            "optional": {
                "reset": ("TRIGGER", {
                    "tooltip": "Manual reset trigger (optional)"
                }),
                "reset_when_done": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically reset environment when episode ends"
                }),
                "render": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable rendering for debugging/visualization"
                }),
                "null_action": ("STRING", {
                    "default": "",
                    "tooltip": "Null action for initialization (comma-separated values). Enter manually for environments without nullAction in YAML."
                }),
                "camera_position": ("STRING", {
                    "default": "1.2, 1.2, 1.0",
                    "tooltip": "Initial camera position (x, y, z). Example: 1.2, 1.2, 1.0"
                }),
                "camera_target": ("STRING", {
                    "default": "0.0, 0.0, 0.5",
                    "tooltip": "Camera look-at target point (x, y, z). Example: 0.0, 0.0, 0.5"
                }),
            }
        }

    RETURN_TYPES = ("TENSOR", "TRIGGER")
    RETURN_NAMES = ("observation", "done")
    FUNCTION = None  # DNNE nodes don't execute in UI, only export


# Node registration
NODE_CLASS_MAPPINGS = {
    "IsaacGymSim": IsaacGymSimNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IsaacGymSim": "Isaac Gym Simulator"
}