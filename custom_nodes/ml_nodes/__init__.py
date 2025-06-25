"""
ML Nodes for DNNE
Fixed version with all corrections applied
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import sys
import os

# Import base types
from custom_nodes.robotics_nodes.robotics_types import TensorData, Context

# Get the base class
from custom_nodes.robotics_nodes import RoboticsNodeBase

# Global context instance
context = None

def get_context():
    """Get or create global context"""
    global context
    if context is None:
        context = Context()
    return context


class MNISTDatasetNode(RoboticsNodeBase):
    """Load MNIST dataset"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_path": ("STRING", {"default": "./data"}),
                "train": ("BOOLEAN", {"default": True}),
                "download": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("DATASET", "INT", "INT")
    RETURN_NAMES = ("dataset", "num_samples", "num_classes")
    FUNCTION = "load_dataset"
    CATEGORY = "ml/data"

    def load_dataset(self, data_path, train, download):
        # Import here to avoid dependency if not used
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )

        return (dataset, len(dataset), 10)  # MNIST has 10 classes


class BatchSamplerNode(RoboticsNodeBase):
    """Create batches from dataset"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET",),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 512}),
                "shuffle": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1}),  # -1 means random
            }
        }

    RETURN_TYPES = ("DATALOADER",)
    RETURN_NAMES = ("dataloader",)
    FUNCTION = "create_dataloader"
    CATEGORY = "ml/data"

    def create_dataloader(self, dataset, batch_size, shuffle, seed):
        from torch.utils.data import DataLoader

        # Set seed if specified
        generator = None
        if seed >= 0:
            generator = torch.Generator()
            generator.manual_seed(seed)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if torch.cuda.is_available() else False
        )

        return (dataloader,)


class GetBatchNode(RoboticsNodeBase):
    """Get next batch from dataloader"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataloader": ("DATALOADER",)
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "BOOLEAN")
    RETURN_NAMES = ("images", "labels", "epoch_complete")
    FUNCTION = "get_batch"
    CATEGORY = "ml/data"

    def get_batch(self, dataloader):
        context = get_context()
        
        # Get or create iterator in context
        if "dataloader_iter" not in context.memory:
            context.memory["dataloader_iter"] = iter(dataloader)
            context.memory["epoch_complete"] = False

        try:
            images, labels = next(context.memory["dataloader_iter"])
            epoch_complete = False
        except StopIteration:
            # Reset iterator for next epoch
            context.memory["dataloader_iter"] = iter(dataloader)
            images, labels = next(context.memory["dataloader_iter"])
            epoch_complete = True
            context.episode_count += 1

        return (images, labels, epoch_complete)


class LinearLayerNode(RoboticsNodeBase):
    """Fully connected linear layer"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR",),
                "output_size": ("INT", {"default": 128, "min": 1, "max": 4096}),
                "bias": ("BOOLEAN", {"default": True}),
                "activation": (["none", "relu", "tanh", "sigmoid"], {"default": "relu"}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "ml/layers"

    def forward(self, input, output_size, bias, activation, dropout):
        context = get_context()
        
        # Flatten input if needed
        if len(input.shape) > 2:
            input = input.view(input.size(0), -1)

        input_size = input.shape[1]
        layer_key = f"linear_{id(self)}_{input_size}_{output_size}"

        # Get or create layer
        if layer_key not in context.memory:
            layer = nn.Linear(input_size, output_size, bias=bias)
            if torch.cuda.is_available() and input.is_cuda:
                layer = layer.cuda()
            context.memory[layer_key] = layer
        else:
            layer = context.memory[layer_key]

        # Set training mode
        layer.train(context.training)

        # Forward pass
        output = layer(input)

        # Apply activation
        if activation == "relu":
            output = F.relu(output)
        elif activation == "tanh":
            output = torch.tanh(output)
        elif activation == "sigmoid":
            output = torch.sigmoid(output)

        # Apply dropout
        if dropout > 0 and context.training:
            output = F.dropout(output, p=dropout, training=True)

        return (output,)


class Conv2DLayerNode(RoboticsNodeBase):
    """2D Convolutional layer"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR",),
                "out_channels": ("INT", {"default": 32, "min": 1, "max": 512}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 11}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 5}),
                "padding": ("INT", {"default": 1, "min": 0, "max": 5}),
                "activation": (["none", "relu", "tanh", "sigmoid"], {"default": "relu"}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "ml/layers"

    def forward(self, input, out_channels, kernel_size, stride, padding, activation):
        context = get_context()
        
        # Get input channels
        in_channels = input.shape[1]
        layer_key = f"conv2d_{id(self)}_{in_channels}_{out_channels}_{kernel_size}"

        # Get or create layer
        if layer_key not in context.memory:
            layer = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            if torch.cuda.is_available() and input.is_cuda:
                layer = layer.cuda()
            context.memory[layer_key] = layer
        else:
            layer = context.memory[layer_key]

        # Set training mode
        layer.train(context.training)

        # Forward pass
        output = layer(input)

        # Apply activation
        if activation == "relu":
            output = F.relu(output)
        elif activation == "tanh":
            output = torch.tanh(output)
        elif activation == "sigmoid":
            output = torch.sigmoid(output)

        return (output,)


class ActivationNode(RoboticsNodeBase):
    """Apply activation function"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR",),
                "activation": (["relu", "tanh", "sigmoid", "softmax", "leaky_relu", "elu"], {"default": "relu"}),
                "negative_slope": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0}),  # for leaky_relu
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply"
    CATEGORY = "ml/activation"

    def apply(self, input, activation, negative_slope):
        if activation == "relu":
            output = F.relu(input)
        elif activation == "tanh":
            output = torch.tanh(input)
        elif activation == "sigmoid":
            output = torch.sigmoid(input)
        elif activation == "softmax":
            output = F.softmax(input, dim=-1)
        elif activation == "leaky_relu":
            output = F.leaky_relu(input, negative_slope=negative_slope)
        elif activation == "elu":
            output = F.elu(input)
        
        return (output,)


class DropoutNode(RoboticsNodeBase):
    """Apply dropout"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR",),
                "dropout_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply"
    CATEGORY = "ml/regularization"

    def apply(self, input, dropout_rate):
        context = get_context()
        if context.training and dropout_rate > 0:
            output = F.dropout(input, p=dropout_rate, training=True)
        else:
            output = input
        return (output,)


class BatchNormNode(RoboticsNodeBase):
    """Batch normalization"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR",),
                "momentum": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.99}),
                "eps": ("FLOAT", {"default": 1e-5, "min": 1e-8, "max": 1e-3}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply"
    CATEGORY = "ml/normalization"

    def apply(self, input, momentum, eps):
        context = get_context()
        
        # Determine the number of features
        if len(input.shape) == 2:  # Fully connected layer output
            num_features = input.shape[1]
            layer_key = f"batchnorm1d_{id(self)}_{num_features}"
            layer_class = nn.BatchNorm1d
        elif len(input.shape) == 4:  # Convolutional layer output
            num_features = input.shape[1]
            layer_key = f"batchnorm2d_{id(self)}_{num_features}"
            layer_class = nn.BatchNorm2d
        else:
            raise ValueError(f"Unsupported input shape: {input.shape}")

        # Get or create layer
        if layer_key not in context.memory:
            layer = layer_class(num_features, momentum=momentum, eps=eps)
            if torch.cuda.is_available() and input.is_cuda:
                layer = layer.cuda()
            context.memory[layer_key] = layer
        else:
            layer = context.memory[layer_key]

        # Set training mode
        layer.train(context.training)

        # Apply batch norm
        output = layer(input)
        return (output,)


class FlattenNode(RoboticsNodeBase):
    """Flatten tensor"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR",),
                "start_dim": ("INT", {"default": 1, "min": 0, "max": 3}),
                "end_dim": ("INT", {"default": -1, "min": -1, "max": 3}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "flatten"
    CATEGORY = "ml/tensor"

    def flatten(self, input, start_dim, end_dim):
        output = torch.flatten(input, start_dim, end_dim)
        return (output,)


class CrossEntropyLossNode(RoboticsNodeBase):
    """Cross entropy loss"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "predictions": ("TENSOR",),
                "labels": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("loss",)
    FUNCTION = "compute_loss"
    CATEGORY = "ml/loss"

    def compute_loss(self, predictions, labels):
        loss = F.cross_entropy(predictions, labels)
        return (loss,)


class AccuracyNode(RoboticsNodeBase):
    """Calculate accuracy"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "predictions": ("TENSOR",),
                "labels": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT")
    RETURN_NAMES = ("accuracy", "correct", "total")
    FUNCTION = "calculate"
    CATEGORY = "ml/metrics"

    def calculate(self, predictions, labels):
        _, predicted = torch.max(predictions, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        
        return (accuracy, correct, total)


class SGDOptimizerNode(RoboticsNodeBase):
    """SGD optimizer"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0}),
                "momentum": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.99}),
            }
        }

    RETURN_TYPES = ("OPTIMIZER",)
    RETURN_NAMES = ("optimizer",)
    FUNCTION = "create_optimizer"
    CATEGORY = "ml/optimization"

    def create_optimizer(self, learning_rate, momentum):
        context = get_context()
        
        # Collect all parameters from stored layers
        parameters = []
        for key, module in context.memory.items():
            if isinstance(module, nn.Module):
                parameters.extend(module.parameters())
        
        if not parameters:
            raise ValueError("No parameters found to optimize. Create layers first.")
        
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)
        return (optimizer,)


class TrainingStepNode(RoboticsNodeBase):
    """Perform training step"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loss": ("TENSOR",),
                "optimizer": ("OPTIMIZER",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "training_step"
    CATEGORY = "ml/training"

    def training_step(self, loss, optimizer):
        loss_tensor = self.ensure_tensor(loss)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss_tensor.backward()
        
        # Update weights
        optimizer.step()
        
        # Training step complete
        return ()


class CreateContextNode(RoboticsNodeBase):
    """Create a new context"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reset": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "create"
    CATEGORY = "ml/control"

    def create(self, reset):
        global context
        if reset or context is None:
            context = Context()
        return ()


class SetModeNode(RoboticsNodeBase):
    """Set training/eval mode"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["train", "eval"], {"default": "train"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_mode"
    CATEGORY = "ml/control"

    def set_mode(self, mode):
        context = get_context()
        context.training = (mode == "train")
        
        # Update all stored modules
        for key, module in context.memory.items():
            if isinstance(module, nn.Module):
                module.train(context.training)
        
        return ()


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
        # Import here to avoid dependency if not used
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        
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


# Node mappings
NODE_CLASS_MAPPINGS = {
    # Data nodes
    "MNISTDataset": MNISTDatasetNode,
    "BatchSampler": BatchSamplerNode,
    "GetBatch": GetBatchNode,

    # Layer nodes
    "LinearLayer": LinearLayerNode,
    "Conv2DLayer": Conv2DLayerNode,
    "Activation": ActivationNode,
    "Dropout": DropoutNode,
    "BatchNorm": BatchNormNode,
    "Flatten": FlattenNode,

    # Training nodes
    "CrossEntropyLoss": CrossEntropyLossNode,
    "Accuracy": AccuracyNode,
    "SGDOptimizer": SGDOptimizerNode,
    "TrainingStep": TrainingStepNode,

    # Control nodes
    "CreateContext": CreateContextNode,
    "SetMode": SetModeNode,

    # Visualization
    "TensorVisualizer": TensorVisualizerNode,
}

# Display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "MNISTDataset": "MNIST Dataset",
    "BatchSampler": "Batch Sampler",
    "GetBatch": "Get Batch",
    "LinearLayer": "Linear Layer",
    "Conv2DLayer": "Conv2D Layer",
    "Activation": "Activation",
    "Dropout": "Dropout",
    "BatchNorm": "Batch Normalization",
    "Flatten": "Flatten",
    "CrossEntropyLoss": "Cross Entropy Loss",
    "Accuracy": "Accuracy",
    "SGDOptimizer": "SGD Optimizer",
    "TrainingStep": "Training Step",
    "CreateContext": "Create Context",
    "SetMode": "Set Mode",
    "TensorVisualizer": "Tensor Visualizer",
}

# Export
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
 