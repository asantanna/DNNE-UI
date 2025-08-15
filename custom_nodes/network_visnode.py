"""
Network Node
Consolidates multiple LinearLayer nodes into a single PyTorch Sequential model with checkpoint support.
For checkpoint debugging: check console logs or exported code for actual node ID.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class NetworkNode(RoboticsNodeBase):
    """
    Network Node
    Consolidates multiple LinearLayer nodes into a single PyTorch Sequential model with checkpoint support.
    For checkpoint debugging: check console logs or exported code for actual node ID.
    """


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*TENSOR", {"tooltip": "Input tensor to process through the neural network"}),
                "to_output": ("*TENSOR", {"tooltip": "Loop-back connection from the last layer output"}),
                # Checkpoint parameters - must be widgets to save to widgets_values
                "checkpoint_enabled": ("BOOLEAN", {"default": True, "widget": {"name": "checkpoint_enabled"}, "tooltip": "Enable automatic checkpoint saving for this network. Checkpoints saved to 'node_<ID>' subdirectories."}),
                "checkpoint_trigger_type": (["epoch", "time", "best_metric"], {"default": "epoch", "widget": {"name": "checkpoint_trigger_type"}, "tooltip": "When to save checkpoints: every N steps, time intervals, or metric improvements"}),
                "checkpoint_trigger_value": ("STRING", {"default": "50", "widget": {"name": "checkpoint_trigger_value"}, "tooltip": "Trigger value: number (steps), time format (1h30m), or 'min'/'max' (metrics)"}),
                "checkpoint_load_on_start": ("BOOLEAN", {"default": False, "widget": {"name": "checkpoint_load_on_start"}, "tooltip": "Automatically load saved checkpoint when network starts"}),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("LAYER_TENSOR", "NETWORK_OUTPUT_TENSOR", "NETWORK_MODEL_OBJ")
    RETURN_NAMES = ("layers", "output", "model")
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("network")["color"]
    BGCOLOR = get_node_colors("network")["bgcolor"]

# Node registration
NODE_CLASS_MAPPINGS = {
    "Network": NetworkNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Network": "Neural Network"
}