"""
Geometric Loss Node
Computes various geometric distance and divergence metrics between predictions and estimates.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class GeometricLossNode(RoboticsNodeBase):
    """Geometric Loss Node
    Computes various geometric distance and divergence metrics between predictions and estimates.
    
    Metrics:
    - Max Abs Error: Maximum absolute difference (L∞ norm)
    - Euclidean Dist: Euclidean distance (L2 norm)
    - Manhattan Dist: Sum of absolute differences (L1 norm)
    - KL Div: Kullback-Leibler divergence between normalized distributions
    - Norm KL Div: Normalized KL divergence (0-1 scale, divided by log(n))
    """
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("training")["color"]
    BGCOLOR = get_node_colors("training")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "predictions": ("*TENSOR", {
                    "tooltip": "Predicted values tensor. Can be any shape - will be flattened for distance computation."
                }),
                "estimates": ("*TENSOR", {
                    "tooltip": "Target/estimate values tensor. Must match shape of predictions."
                }),
                "error_metric": (["Max Abs Error", "Euclidean Dist", "Manhattan Dist", "KL Div", "Norm KL Div"], {
                    "default": "Euclidean Dist",
                    "tooltip": "Metric to compute: Max Abs Error (L∞), Euclidean Dist (L2), Manhattan Dist (L1), KL Divergence, or Normalized KL Div (0-1 scale)"
                }),
            }
        }

    RETURN_TYPES = ("GEOMETRIC_LOSS_SCALAR",)
    RETURN_NAMES = ("output",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"

# Node registration
NODE_CLASS_MAPPINGS = {
    "GeometricLoss": GeometricLossNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeometricLoss": "Geometric Loss"
}