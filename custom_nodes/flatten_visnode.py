"""
Flatten Node
Flattens multi-dimensional tensors into 2D for fully connected layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class FlattenNode(RoboticsNodeBase):
    """Flatten Node
    Flattens multi-dimensional tensors into 2D for fully connected layers."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("layer")["color"]
    BGCOLOR = get_node_colors("layer")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to flatten. Typically from convolutional layers before feeding to fully connected layers."}),
                "start_dim": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 4,
                    "tooltip": "First dimension to flatten. Default 1 preserves batch dimension and flattens all others."
                }),
                "end_dim": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 4,
                    "tooltip": "Last dimension to flatten. Default -1 means flatten to the last dimension."
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "INT")
    RETURN_NAMES = ("output", "flattened_size")
    FUNCTION = "flatten_tensor"
    CATEGORY = "ml"

    def flatten_tensor(self, input, start_dim, end_dim):
        # Flatten the tensor
        output = torch.flatten(input, start_dim=start_dim, end_dim=end_dim)
        
        # Calculate flattened size (useful for determining input size of next linear layer)
        if start_dim == 1 and end_dim == -1:
            # Common case: preserve batch, flatten rest
            flattened_size = output.shape[1] if len(output.shape) > 1 else output.shape[0]
        else:
            flattened_size = output.shape[start_dim] if start_dim < len(output.shape) else 1
        
        return (output, flattened_size)

# Node registration
NODE_CLASS_MAPPINGS = {
    "Flatten": FlattenNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flatten": "Flatten"
}