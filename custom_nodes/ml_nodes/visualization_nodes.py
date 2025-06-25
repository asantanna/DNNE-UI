"""
Visualization nodes for tensors and training metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from .base import RoboticsNodeBase


class TensorVisualizerNode(RoboticsNodeBase):
    """Visualize tensor data"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TENSOR",),
                "title": ("STRING", {"default": "Tensor"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "ml/visualization"
    OUTPUT_NODE = True

    def visualize(self, tensor, title):
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