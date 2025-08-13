"""
Isaac Gym Simulator Interface
Queue-based interface to Isaac Gym environments for real-time simulation.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase
from custom_nodes.node_colors import get_node_colors


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
    FUNCTION = "step_environment"

    def __init__(self):
        super().__init__()
        self.env = None
        self.env_config = None
        self.device = None

    def step_environment(self, config: Dict[str, Any], action: torch.Tensor, 
                        reset: Optional[Any] = None, reset_when_done: bool = True, 
                        render: bool = False, null_action: str = "",
                        camera_position: str = "1.2, 1.2, 1.0",
                        camera_target: str = "0.0, 0.0, 0.5") -> Tuple[torch.Tensor, Optional[Any]]:
        """
        This method is called during UI execution only.
        The actual environment stepping happens in the exported queue-based code.
        """
        # In the UI, we just validate inputs and return dummy outputs
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary from Isaac Gym Environment Config node")
        
        if not isinstance(action, torch.Tensor):
            raise ValueError("Action must be a tensor")
        
        # Get expected observation shape from config
        task = config.get("task")
        if not task:
            raise ValueError("IsaacGymSim: config must provide 'task' field")
        
        # Create dummy observation based on task
        # These are approximate observation sizes for common tasks
        obs_sizes = {
            "Cartpole": 4,
            "Ant": 60,
            "Humanoid": 108,
            "Anymal": 48,
            "BallBalance": 38,
            "FrankaCabinet": 23,
        }
        
        obs_size = obs_sizes.get(task)
        if obs_size is None:
            raise ValueError(f"IsaacGymSim: Unknown task '{task}'. Supported tasks: {list(obs_sizes.keys())}")
        
        # Create dummy observation tensor
        observation = torch.zeros((1, obs_size), dtype=torch.float32)
        
        # Done is a trigger signal (we'll return None in UI mode)
        done_signal = None
        
        return (observation, done_signal)


# Node registration
NODE_CLASS_MAPPINGS = {
    "IsaacGymSim": IsaacGymSimNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IsaacGymSim": "Isaac Gym Simulator"
}