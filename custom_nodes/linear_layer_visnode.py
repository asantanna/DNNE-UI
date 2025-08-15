"""
Linear Layer Node
Represents a fully connected (dense) layer in a neural network with optional activation.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors
from custom_nodes.utils.dnne_decorator import dnne_node


@dnne_node(is_virtual=True)
class LinearLayerNode(RoboticsNodeBase):
    """Linear Layer Node
    Represents a fully connected (dense) layer in a neural network with optional activation."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("layer")["color"]
    BGCOLOR = get_node_colors("layer")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*TENSOR", {"tooltip": "Input tensor to transform (automatically flattened if > 2D)"}),
                "output_size": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 4096,
                    "tooltip": "Number of output features (neurons) in this layer"
                }),
                "bias": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to include learnable bias parameters"
                }),
                "activation": (["none", "relu", "tanh", "sigmoid", "leaky_relu"], {
                    "default": "relu",
                    "tooltip": "Activation function to apply after linear transformation"
                }),
                "dropout": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.9,
                    "tooltip": "Dropout probability for regularization (0.0 = no dropout)"
                }),
                "weight_init": (["auto", "kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform", "normal", "uniform", "none"], {
                    "default": "auto",
                    "widget": {"name": "weight_init"},
                    "tooltip": "Weight initialization method. 'auto' chooses based on activation function: Kaiming for ReLU/LeakyReLU, Xavier for tanh/sigmoid"
                })
            }
        }

    RETURN_TYPES = ("LAYER_TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"

# Node registration
NODE_CLASS_MAPPINGS = {
    "LinearLayer": LinearLayerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LinearLayer": "Linear Layer"
}