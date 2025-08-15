"""
Get Batch Node
Retrieves the next batch from a DataLoader and tracks epoch progress and statistics.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors
from custom_nodes.utils.dnne_decorator import dnne_node


@dnne_node(is_virtual=False)
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
                "dataloader": ("*DATALOADER_OBJ", {
                    "tooltip": "Configured PyTorch DataLoader that provides batched data. Should come from BatchSamplerNode."
                }),
                "schema": ("*SCHEMA_PYDICT", {
                    "tooltip": "Dataset schema with metadata about batch structure and data types. Used for output validation and downstream nodes."
                }),
                "trigger": ("*TRIGGER", {
                    "tooltip": "Synchronization trigger that controls when to fetch the next batch. Connect to training loop or other control nodes."
                })
            }
        }

    RETURN_TYPES = ("BATCH_IMAGE_TENSOR", "BATCH_LABEL_TENSOR", "BOOLEAN", "BATCH_EPOCH_STATS_PYDICT")
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