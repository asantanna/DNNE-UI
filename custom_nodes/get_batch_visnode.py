"""
Get Batch Node
Retrieves the next batch from a DataLoader and tracks epoch progress and statistics.
"""

from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase
from custom_nodes.node_colors import get_node_colors


class GetBatchNode(RoboticsNodeBase):
    """Get Batch Node
    Retrieves the next batch from a DataLoader and tracks epoch progress and statistics."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("data")["color"]
    BGCOLOR = get_node_colors("data")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataloader": ("DATALOADER", {
                    "tooltip": "Configured PyTorch DataLoader that provides batched data. Should come from BatchSamplerNode."
                }),
                "schema": ("SCHEMA", {
                    "tooltip": "Dataset schema with metadata about batch structure and data types. Used for output validation and downstream nodes."
                }),
                "trigger": ("SYNC", {
                    "tooltip": "Synchronization trigger that controls when to fetch the next batch. Connect to training loop or other control nodes."
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "BOOLEAN", "DICT")
    RETURN_NAMES = ("images", "labels", "epoch_complete", "epoch_stats")
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"

# Node registration
NODE_CLASS_MAPPINGS = {
    "GetBatch": GetBatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetBatch": "Get Batch"
}