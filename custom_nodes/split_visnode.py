"""
Split Node for tensor splitting with configurable slicing
Splits a single tensor input into up to 4 outputs based on indices or sizes
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class SplitNode(RoboticsNodeBase):
    """
    Split Node for tensor splitting
    Splits a single tensor input into up to 4 outputs along a specified dimension
    
    Modes:
    - by index: Split at specific indices (e.g., "10,20,30" splits at positions 10, 20, 30)
    - by size: Split into chunks of specific sizes (e.g., "10,10,10,10" creates 4 chunks of size 10)
    
    The number of outputs depends on the split_pos specification:
    - For "by index" with N indices: Creates N+1 outputs
    - For "by size" with N non-zero sizes: Creates N outputs
    """
    
    CATEGORY = "utility"
    COLOR = get_node_colors("utility")["color"]
    BGCOLOR = get_node_colors("utility")["bgcolor"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*TENSOR", {"tooltip": "Input tensor to split"}),
                "dimension": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 3,
                    "tooltip": "Dimension along which to split the tensor (0=batch, 1=channels, etc.)"
                }),
                "split_mode": (["by index", "by size"], {
                    "default": "by index",
                    "tooltip": "How to interpret split_pos: as split indices or as chunk sizes"
                }),
                "split_pos": ("STRING", {
                    "default": "10,20,30",
                    "tooltip": "Comma-separated values. For 'by index': split points. For 'by size': chunk sizes."
                }),
            }
        }
    
    RETURN_TYPES = ("*TENSOR", "*TENSOR", "*TENSOR", "*TENSOR")
    RETURN_NAMES = ("output_a", "output_b", "output_c", "output_d")
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    DESCRIPTION = cleandoc(__doc__)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Split": SplitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Split": "Split"
}