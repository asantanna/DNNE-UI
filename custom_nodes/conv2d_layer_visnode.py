"""
Conv2D Layer Node
2D Convolutional layer for processing image data with learnable filters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class Conv2DLayerNode(RoboticsNodeBase):
    """Conv2D Layer Node
    2D Convolutional layer for processing image data with learnable filters."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("layer")["color"]
    BGCOLOR = get_node_colors("layer")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor in NCHW format (batch, channels, height, width)"}),
                "in_channels": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 2048,
                    "tooltip": "Number of input channels. For grayscale images use 1, for RGB use 3."
                }),
                "out_channels": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 2048,
                    "tooltip": "Number of output channels (filters). More filters can capture more features but increase computation."
                }),
                "kernel_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 11,
                    "tooltip": "Size of the convolutional kernel. Common values: 3, 5, 7. Larger kernels capture larger spatial features."
                }),
                "stride": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "tooltip": "Stride of the convolution. Values > 1 reduce spatial dimensions. Stride 2 halves the output size."
                }),
                "padding": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Zero-padding added to input. Use kernel_size//2 to maintain spatial dimensions with stride=1."
                }),
                "activation": (["none", "relu", "tanh", "sigmoid", "leaky_relu"], {
                    "default": "relu",
                    "tooltip": "Activation function to apply after convolution. ReLU is most common for hidden layers."
                })
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_conv"
    CATEGORY = "ml"

    def __init__(self):
        super().__init__()
        self.conv = None

    def apply_conv(self, input, in_channels, out_channels, kernel_size, stride, padding, activation):
        # Create conv layer if needed
        if self.conv is None:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            device = input.device if isinstance(input, torch.Tensor) else torch.device("cpu")
            self.conv = self.conv.to(device)

        # Apply convolution
        output = self.conv(input)

        # Apply activation
        if activation == "relu":
            output = F.relu(output)
        elif activation == "tanh":
            output = torch.tanh(output)
        elif activation == "sigmoid":
            output = torch.sigmoid(output)
        elif activation == "leaky_relu":
            output = F.leaky_relu(output, negative_slope=0.01)

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "Conv2DLayer": Conv2DLayerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Conv2DLayer": "Conv2D Layer"
}