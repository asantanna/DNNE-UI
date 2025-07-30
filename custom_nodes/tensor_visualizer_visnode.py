"""
Tensor Visualizer Node
Visualizes tensor data including distributions, statistics, and shapes for debugging.
"""

import torch
import numpy as np
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase
from custom_nodes.node_colors import get_node_colors
import json
import math


class TensorVisualizerNode(RoboticsNodeBase):
    """Tensor Visualizer Node
    Visualizes tensor data including distributions, statistics, and shapes for debugging."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("visualization")["color"]
    BGCOLOR = get_node_colors("visualization")["bgcolor"]
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TENSOR", {
                    "tooltip": "Tensor to visualize. Can be any shape or type."
                }),
                "name": ("STRING", {
                    "default": "tensor",
                    "tooltip": "Name to identify this tensor in the visualization"
                }),
                "bins": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "tooltip": "Number of bins for histogram visualization"
                }),
                "sample_size": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 10000,
                    "tooltip": "Maximum number of values to sample for visualization (to limit UI data)"
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "UI")
    RETURN_NAMES = ("tensor", "visualization")
    FUNCTION = "visualize_tensor"
    CATEGORY = "ml"

    def visualize_tensor(self, tensor, name, bins, sample_size):
        # Pass through the tensor unchanged
        output_tensor = tensor
        
        # Prepare visualization data
        vis_data = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad
        }
        
        # Convert to numpy for statistics
        tensor_np = tensor.detach().cpu().numpy()
        flat_tensor = tensor_np.flatten()
        
        # Sample if too large
        if len(flat_tensor) > sample_size:
            indices = np.random.choice(len(flat_tensor), sample_size, replace=False)
            sampled = flat_tensor[indices]
        else:
            sampled = flat_tensor
        
        # Compute statistics
        vis_data["stats"] = {
            "min": float(np.min(flat_tensor)),
            "max": float(np.max(flat_tensor)),
            "mean": float(np.mean(flat_tensor)),
            "std": float(np.std(flat_tensor)),
            "median": float(np.median(sampled)),  # Use sampled for expensive operations
            "num_elements": int(tensor.numel()),
            "num_nan": int(np.isnan(flat_tensor).sum()),
            "num_inf": int(np.isinf(flat_tensor).sum())
        }
        
        # Create histogram
        if not (np.isnan(sampled).all() or np.isinf(sampled).all()):
            hist, bin_edges = np.histogram(sampled[~np.isnan(sampled) & ~np.isinf(sampled)], bins=bins)
            vis_data["histogram"] = {
                "counts": hist.tolist(),
                "bins": bin_edges.tolist()
            }
        
        # Sample values for display
        vis_data["sample_values"] = sampled[:min(20, len(sampled))].tolist()
        
        # UI output
        ui_output = {
            "ui": {
                "tensor_info": [vis_data]
            }
        }
        
        return (output_tensor, ui_output)

# Node registration
NODE_CLASS_MAPPINGS = {
    "TensorVisualizer": TensorVisualizerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorVisualizer": "Tensor Visualizer"
}