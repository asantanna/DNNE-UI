"""
Get Batch Node
Retrieves the next batch from a DataLoader and tracks epoch progress and statistics.
"""

import torch
from torch.utils.data import DataLoader
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
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
    FUNCTION = "get_batch"
    CATEGORY = "ml"

    def get_batch(self, dataloader, schema, trigger):
        context = get_context()
        
        # Initialize tracking variables
        if "dataloader_iter" not in context.memory:
            context.memory["dataloader_iter"] = iter(dataloader)
            context.memory["epoch_complete"] = False
            context.memory["current_epoch"] = 0
            context.memory["batch_in_epoch"] = 0
            context.memory["total_batches_per_epoch"] = len(dataloader)

        try:
            images, labels = next(context.memory["dataloader_iter"])
            context.memory["batch_in_epoch"] += 1
            epoch_complete = False
        except StopIteration:
            # End of epoch
            context.memory["dataloader_iter"] = iter(dataloader)
            images, labels = next(context.memory["dataloader_iter"])
            epoch_complete = True
            context.memory["current_epoch"] += 1
            context.memory["batch_in_epoch"] = 1
            context.episode_count += 1

        # Create epoch stats
        epoch_stats = {
            "epoch": context.memory["current_epoch"],
            "batch": context.memory["batch_in_epoch"],
            "total_batches": context.memory["total_batches_per_epoch"],
            "progress": context.memory["batch_in_epoch"] / context.memory["total_batches_per_epoch"],
            "completed": epoch_complete
        }

        return (images, labels, epoch_complete, epoch_stats)

# Node registration
NODE_CLASS_MAPPINGS = {
    "GetBatch": GetBatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetBatch": "Get Batch"
}