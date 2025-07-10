"""
Visualization nodes for tensors and training metrics
"""

import torch
import numpy as np
import io
from inspect import cleandoc
from .base import RoboticsNodeBase

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False


class TensorVisualizerNode(RoboticsNodeBase):
    """Tensor Visualizer Node
    Creates visual plots and images from tensor data with support for 1D-4D tensors."""
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TENSOR", {"tooltip": "Input tensor to visualize. Supports 1D (line plot), 2D (heatmap), 3D (RGB/grayscale image), and 4D (batch of images) tensors. Data will be automatically converted from PyTorch tensors to numpy arrays for visualization."}),
                "title": ("STRING", {"default": "Tensor", "tooltip": "Title for the visualization plot. This text will appear as the main title of the generated graph or image visualization to help identify the content being displayed."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "ml/visualization"
    OUTPUT_NODE = True

    def visualize(self, tensor, title):
        if not MATPLOTLIB_AVAILABLE:
            print(f"Visualization skipped: matplotlib not available. Title: {title}")
            return (f"Visualization unavailable: {title}",)
            
        # Convert tensor to numpy
        if isinstance(tensor, torch.Tensor):
            data = tensor.detach().cpu().numpy()
        else:
            data = np.array(tensor)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Handle different tensor shapes
        if len(data.shape) == 1:
            ax.plot(data)
            ax.set_title(f"{title} (1D)")
        elif len(data.shape) == 2:
            im = ax.imshow(data, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title} (2D)")
        elif len(data.shape) == 3:
            # Show first 3 channels as RGB
            if data.shape[0] >= 3:
                rgb = np.transpose(data[:3], (1, 2, 0))
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
                ax.imshow(rgb)
            else:
                ax.imshow(data[0], cmap='gray')
            ax.set_title(f"{title} (3D)")
        elif len(data.shape) == 4:
            # Show grid of first batch
            n_show = min(4, data.shape[0])
            for i in range(n_show):
                ax = plt.subplot(2, 2, i+1)
                if data.shape[1] >= 3:
                    rgb = np.transpose(data[i, :3], (1, 2, 0))
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
                    ax.imshow(rgb)
                else:
                    ax.imshow(data[i, 0], cmap='gray')
                ax.axis('off')
            plt.suptitle(f"{title} (4D batch)")
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        # Convert to tensor format expected by ComfyUI
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        
        return (img_tensor,)