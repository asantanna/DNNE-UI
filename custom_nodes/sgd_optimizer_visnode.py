"""
SGD Optimizer Node
Stochastic Gradient Descent optimizer for training neural networks.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class SGDOptimizerNode(RoboticsNodeBase):
    """SGD Optimizer Node
    Stochastic Gradient Descent optimizer for training neural networks."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("training")["color"]
    BGCOLOR = get_node_colors("training")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("*MODEL_OBJ", {
                    "tooltip": "Neural network model to optimize. Must have parameters to train."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0001,
                    "max": 1.0,
                    "step": 0.0001,
                    "tooltip": "Learning rate controls step size. Start with 0.01 or 0.001, adjust based on loss curve."
                }),
                "momentum": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 0.999,
                    "step": 0.001,
                    "tooltip": "Momentum factor accelerates SGD in relevant direction. 0.9 is a good default."
                }),
                "weight_decay": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.0001,
                    "tooltip": "L2 penalty (regularization). Helps prevent overfitting. Try 0.0001 to 0.001."
                })
            }
        }

    RETURN_TYPES = ("SGD_OPTIMIZER_OBJ",)
    RETURN_NAMES = ("optimizer",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"

# Node registration
NODE_CLASS_MAPPINGS = {
    "SGDOptimizer": SGDOptimizerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SGDOptimizer": "SGD Optimizer"
}