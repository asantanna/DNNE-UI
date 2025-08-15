"""
Concat Node for tensor concatenation with synchronization modes
Concatenates up to 4 tensor inputs with configurable waiting/padding behavior
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors
from custom_nodes.utils.dnne_decorator import dnne_node


@dnne_node(is_virtual=False)
class ConcatNode(RoboticsNodeBase):
    """
    Concat Node for tensor concatenation
    Concatenates up to 4 tensor inputs along dimension 0 (batch dimension)
    
    Modes:
    - wait for all: Waits until all connected inputs have data before concatenating
    - as available: Outputs immediately when any input arrives, padding missing inputs
    
    Padding modes (for "as available" mode):
    - pad with zeros: Missing inputs are replaced with zero tensors
    - hold previous: Missing inputs use the last received data for that input
    """
    
    CATEGORY = "utility"
    COLOR = get_node_colors("utility")["color"]
    BGCOLOR = get_node_colors("utility")["bgcolor"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["wait for all", "as available"], {"default": "wait for all"}),
                "pad_mode": (["pad with zeros", "hold previous"], {"default": "pad with zeros"}),
            },
            "optional": {
                "input_a": ("*TENSOR",),
                "input_b": ("*TENSOR",),
                "input_c": ("*TENSOR",),
                "input_d": ("*TENSOR",),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)  # Note: outputs TENSOR, not *TENSOR
    RETURN_NAMES = ("output",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    DESCRIPTION = cleandoc(__doc__)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Concat": ConcatNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Concat": "Concat"
}