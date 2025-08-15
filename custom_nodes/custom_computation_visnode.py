"""
Custom Computation Node
Executes user-defined Python functions on tensors.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors
from custom_nodes.utils.dnne_decorator import dnne_node


@dnne_node(is_virtual=False)
class CustomComputationNode(RoboticsNodeBase):
    """Custom Computation Node
    Executes user-defined Python functions on tensors."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("utility")["color"]
    BGCOLOR = get_node_colors("utility")["bgcolor"]
    CATEGORY = "utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*TENSOR", {
                    "tooltip": "Input tensor to process with custom function"
                }),
                "src_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to Python file containing compute(input: Tensor) -> Tensor function"
                }),
            }
        }

    RETURN_TYPES = ("CUSTOMCOMP_OUTPUT_TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate that src_path is provided."""
        src_path = kwargs.get("src_path", "").strip()
        if not src_path:
            return "src_path is required - must point to a Python file with compute() function"
        return True


# Node registration
NODE_CLASS_MAPPINGS = {
    "CustomComputation": CustomComputationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomComputation": "Custom Computation"
}