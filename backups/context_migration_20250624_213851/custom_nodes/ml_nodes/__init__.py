# custom_nodes/ml_nodes/__init__.py
"""
Machine Learning nodes for DNNE
Includes data loading, neural network layers, and training components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

# Import base types
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from robotics_nodes.robotics_types import TensorData, Context
from robotics_nodes.base_node import RoboticsNodeBase

# Data Loading Nodes
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
        
        return (dataset, len(dataset), 10)


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
            generator=generator
        )
        
        return (dataloader,)


class GetBatchNode(RoboticsNodeBase):
    """Get next batch from dataloader"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input": ("TENSOR",)}}
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "CONTEXT", "BOOLEAN")
    RETURN_NAMES = ("images", "labels", "context", "epoch_complete")
    FUNCTION = "get_batch"
    CATEGORY = "ml/data"
    
    def get_batch(self, dataloader):
        if context is None:
            context = Context()
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
            context.episode_count += 1  # Use episode_count as epoch count
        
        return (
            TensorData(images, shape_info=f"Batch images: {images.shape}"),
            TensorData(labels, shape_info=f"Batch labels: {labels.shape}"),
            context,
            epoch_complete
        )


# Neural Network Layer Nodes
class LinearLayerNode(RoboticsNodeBase):
    """Fully connected linear layer with optional activation"""
    
    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "output_size": ("INT", {"default": 128, "min": 1, "max": 4096}),
                "activation": (["none", "relu", "sigmoid", "tanh", "softmax", "leaky_relu"], 
                             {"default": "relu"}),
                "bias": ("BOOLEAN", {"default": True}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99}),
                "weight_init": (["xavier", "kaiming", "normal", "zeros"], {"default": "xavier"}),
            },
            
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output")
    FUNCTION = "forward"
    CATEGORY = "ml/layers"
    
    def __init__(self):
        super().__init__()
        self.layer = None
        self.layer_id = None
    
    def forward(self, input_tensor, output_size, activation, bias, dropout, 
                weight_init):
        if context is None:
            context = Context()
        
        # Convert input
        x = self.ensure_tensor(input_tensor)
        
        # Flatten if needed (except batch dimension)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        input_size = x.shape[1]
        
        # Create unique layer ID
        if self.layer_id is None:
            self.layer_id = f"linear_{id(self)}"
        
        # Get or create layer in context
        if self.layer_id not in context.memory:
            self.layer = nn.Linear(input_size, output_size, bias=bias)
            
            # Initialize weights
            if weight_init == "xavier":
                nn.init.xavier_uniform_(self.layer.weight)
            elif weight_init == "kaiming":
                nn.init.kaiming_uniform_(self.layer.weight, nonlinearity='relu')
            elif weight_init == "normal":
                nn.init.normal_(self.layer.weight, std=0.02)
            elif weight_init == "zeros":
                nn.init.zeros_(self.layer.weight)
            
            if bias:
                nn.init.zeros_(self.layer.bias)
            
            context.memory[self.layer_id] = self.layer
        else:
            self.layer = context.memory[self.layer_id]
        
        # Forward pass through linear layer
        x = self.layer(x)
        
        # Apply activation
        if activation == "relu":
            x = F.relu(x)
        elif activation == "sigmoid":
            x = torch.sigmoid(x)
        elif activation == "tanh":
            x = torch.tanh(x)
        elif activation == "softmax":
            x = F.softmax(x, dim=-1)
        elif activation == "leaky_relu":
            x = F.leaky_relu(x, 0.01)
        # "none" means no activation
        
        # Apply dropout if specified
        if context.training and dropout > 0:
            x = F.dropout(x, p=dropout, training=True)
        
        return (
            TensorData(x, shape_info=f"Linear output: {x.shape}"),
            context
        )


class ActivationNode(RoboticsNodeBase):
    """Activation functions"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "activation": (["relu", "sigmoid", "tanh", "softmax", "leaky_relu"], 
                             {"default": "relu"}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_activation"
    CATEGORY = "ml/layers"
    
    def apply_activation(self, input_tensor, activation):
        x = self.ensure_tensor(input_tensor)
        
        if activation == "relu":
            output = F.relu(x)
        elif activation == "sigmoid":
            output = torch.sigmoid(x)
        elif activation == "tanh":
            output = torch.tanh(x)
        elif activation == "softmax":
            output = F.softmax(x, dim=-1)
        elif activation == "leaky_relu":
            output = F.leaky_relu(x, 0.01)
        
        return (TensorData(output, shape_info=f"{activation} output: {output.shape}"),)


class DropoutNode(RoboticsNodeBase):
    """Dropout layer for regularization"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "dropout_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.99}),
            },
            
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_dropout"
    CATEGORY = "ml/layers"
    
    def apply_dropout(self, input_tensor, dropout_rate):
        x = self.ensure_tensor(input_tensor)
        
        # Check training mode from context
        training = context.training if context else True
        
        if training and dropout_rate > 0:
            output = F.dropout(x, p=dropout_rate, training=True)
        else:
            output = x
        
        return (TensorData(output, shape_info=f"Dropout output: {output.shape}"),)


# Loss and Training Nodes
class CrossEntropyLossNode(RoboticsNodeBase):
    """Calculate cross entropy loss"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "predictions": ("TENSOR",),
                "labels": ("TENSOR",),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "FLOAT")
    RETURN_NAMES = ("loss_tensor", "loss_value")
    FUNCTION = "calculate_loss"
    CATEGORY = "ml/training"
    
    def calculate_loss(self, predictions, labels):
        pred_tensor = self.ensure_tensor(predictions)
        label_tensor = self.ensure_tensor(labels)
        
        # Ensure labels are long type for cross entropy
        if label_tensor.dtype != torch.long:
            label_tensor = label_tensor.long()
        
        loss = F.cross_entropy(pred_tensor, label_tensor)
        
        return (
            TensorData(loss, shape_info="Loss scalar"),
            loss.item()
        )


class AccuracyNode(RoboticsNodeBase):
    """Calculate classification accuracy"""
    
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
    FUNCTION = "calculate_accuracy"
    CATEGORY = "ml/metrics"
    OUTPUT_NODE = True
    
    def calculate_accuracy(self, predictions, labels):
        pred_tensor = self.ensure_tensor(predictions)
        label_tensor = self.ensure_tensor(labels)
        
        # Get predicted classes
        _, predicted = torch.max(pred_tensor, 1)
        
        # Calculate accuracy
        correct = (predicted == label_tensor).sum().item()
        total = label_tensor.size(0)
        accuracy = correct / total if total > 0 else 0.0
        
        return (accuracy, correct, total)


class SGDOptimizerNode(RoboticsNodeBase):
    """SGD optimizer for training"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("CONTEXT",),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0}),
                "momentum": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.99}),
            }
        }
    
    RETURN_TYPES = ("OPTIMIZER")
    RETURN_NAMES = ("optimizer")
    FUNCTION = "create_optimizer"
    CATEGORY = "ml/training"
    
    def create_optimizer(self, context, learning_rate, momentum):
        # Collect all parameters from layers stored in context
        params = []
        for key, value in context.memory.items():
            if isinstance(value, nn.Module):
                params.extend(value.parameters())
        
        if not params:
            raise ValueError("No trainable parameters found in context!")
        
        # Create optimizer
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
        
        # Store in context
        context.memory["optimizer"] = optimizer
        
        return (optimizer,)


class TrainingStepNode(RoboticsNodeBase):
    """Perform one training step (backward + optimizer step)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loss": ("TENSOR",),
                "optimizer": ("OPTIMIZER",),
                "context": ("CONTEXT",),
            }
        }
    
    RETURN_TYPES = ("CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "training_step"
    CATEGORY = "ml/training"
    
    def training_step(self, loss, optimizer, context):
        loss_tensor = self.ensure_tensor(loss)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss_tensor.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Update step count
        context.step_count += 1
        
        return (context,)


# Control and Special Layer Nodes
class CreateContextNode(RoboticsNodeBase):
    """Create a new context for training"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_mode": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "create_context"
    CATEGORY = "ml/control"
    
    def create_context(self, training_mode):
        context = Context()
        context.training = training_mode
        return (context,)


class SetModeNode(RoboticsNodeBase):
    """Set training or inference mode in context"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("CONTEXT",),
                "mode": (["train", "eval"], {"default": "train"}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "set_mode"
    CATEGORY = "ml/control"
    
    def set_mode(self, context, mode):
        context.training = (mode == "train")
        print(f"Set mode to: {'training' if context.training else 'inference'}")
        return (context,)


class SoftmaxNode(RoboticsNodeBase):
    """Standalone softmax activation with temperature scaling"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "dim": ("INT", {"default": -1, "min": -3, "max": 3}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_softmax"
    CATEGORY = "ml/layers"
    
    def apply_softmax(self, input_tensor, temperature, dim):
        x = self.ensure_tensor(input_tensor)
        
        # Apply temperature scaling
        if temperature != 1.0:
            x = x / temperature
        
        # Apply softmax
        output = F.softmax(x, dim=dim)
        
        return (TensorData(output, shape_info=f"Softmax output: {output.shape}"),)


class BatchNormNode(RoboticsNodeBase):
    """Batch normalization layer"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "momentum": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.99}),
                "eps": ("FLOAT", {"default": 1e-5, "min": 1e-8, "max": 1e-3}),
            },
            
        }
    
    RETURN_TYPES = ("TENSOR")
    RETURN_NAMES = ("output")
    FUNCTION = "forward"
    CATEGORY = "ml/layers"
    
    def __init__(self):
        super().__init__()
        self.bn_layer = None
        self.layer_id = None
    
    def forward(self, input_tensor, momentum, eps):
        if context is None:
            context = Context()
        
        x = self.ensure_tensor(input_tensor)
        
        # Create unique layer ID
        if self.layer_id is None:
            self.layer_id = f"batchnorm_{id(self)}"
        
        # Determine number of features based on input shape
        if x.dim() == 2:  # (batch, features)
            num_features = x.shape[1]
            bn_class = nn.BatchNorm1d
        elif x.dim() == 4:  # (batch, channels, height, width)
            num_features = x.shape[1]
            bn_class = nn.BatchNorm2d
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        
        # Get or create batch norm layer
        if self.layer_id not in context.memory:
            self.bn_layer = bn_class(num_features, momentum=momentum, eps=eps)
            context.memory[self.layer_id] = self.bn_layer
        else:
            self.bn_layer = context.memory[self.layer_id]
        
        # Set training mode based on context
        if context.training:
            self.bn_layer.train()
        else:
            self.bn_layer.eval()
        
        # Forward pass
        output = self.bn_layer(x)
        
        return (
            TensorData(output, shape_info=f"BatchNorm output: {output.shape}"),
            context
        )


class Conv2DLayerNode(RoboticsNodeBase):
    """2D Convolutional layer with optional activation and pooling"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "out_channels": ("INT", {"default": 32, "min": 1, "max": 512}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 11}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 4}),
                "padding": ("INT", {"default": 1, "min": 0, "max": 5}),
                "activation": (["none", "relu", "sigmoid", "tanh", "leaky_relu"], 
                             {"default": "relu"}),
                "pool_type": (["none", "max", "avg"], {"default": "none"}),
                "pool_size": ("INT", {"default": 2, "min": 1, "max": 4}),
            },
            
        }
    
    RETURN_TYPES = ("TENSOR")
    RETURN_NAMES = ("output")
    FUNCTION = "forward"
    CATEGORY = "ml/layers"
    
    def __init__(self):
        super().__init__()
        self.layer = None
        self.layer_id = None
    
    def forward(self, input_tensor, out_channels, kernel_size, stride, padding,
                activation, pool_type, pool_size):
        if context is None:
            context = Context()
        
        x = self.ensure_tensor(input_tensor)
        
        # Ensure 4D tensor (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        in_channels = x.shape[1]
        
        # Create unique layer ID
        if self.layer_id is None:
            self.layer_id = f"conv2d_{id(self)}"
        
        # Get or create layer
        if self.layer_id not in context.memory:
            self.layer = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            nn.init.kaiming_normal_(self.layer.weight, mode='fan_out', nonlinearity='relu')
            if self.layer.bias is not None:
                nn.init.zeros_(self.layer.bias)
            
            context.memory[self.layer_id] = self.layer
        else:
            self.layer = context.memory[self.layer_id]
        
        # Forward pass
        x = self.layer(x)
        
        # Apply activation
        if activation == "relu":
            x = F.relu(x)
        elif activation == "sigmoid":
            x = torch.sigmoid(x)
        elif activation == "tanh":
            x = torch.tanh(x)
        elif activation == "leaky_relu":
            x = F.leaky_relu(x, 0.01)
        
        # Apply pooling
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        
        return (
            TensorData(x, shape_info=f"Conv output: {x.shape}"),
            context
        )


class FlattenNode(RoboticsNodeBase):
    """Flatten tensor for transition from conv to linear layers"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "start_dim": ("INT", {"default": 1, "min": 0, "max": 3}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "flatten"
    CATEGORY = "ml/layers"
    
    def flatten(self, input_tensor, start_dim):
        x = self.ensure_tensor(input_tensor)
        output = torch.flatten(x, start_dim=start_dim)
        return (TensorData(output, shape_info=f"Flattened: {output.shape}"),)


# Visualization Node
class TensorVisualizerNode(RoboticsNodeBase):
    """Visualize tensor data (e.g., MNIST images)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TENSOR",),
                "num_samples": ("INT", {"default": 4, "min": 1, "max": 16}),
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "ml/visualization"
    OUTPUT_NODE = True
    
    def visualize(self, tensor, num_samples, normalize):
        import torchvision.utils as vutils
        
        data = self.ensure_tensor(tensor)
        
        # Take first num_samples
        if data.shape[0] > num_samples:
            data = data[:num_samples]
        
        # Ensure 4D tensor (B, C, H, W)
        if data.dim() == 3:
            data = data.unsqueeze(1)  # Add channel dimension
        elif data.dim() == 2:
            # Assume square images
            size = int(data.shape[1] ** 0.5)
            data = data.view(-1, 1, size, size)
        
        # Create grid
        grid = vutils.make_grid(data, nrow=int(num_samples**0.5), 
                               normalize=normalize, scale_each=True)
        
        # Convert to numpy for display (H, W, C)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        return (grid_np,)


# Register custom types
CUSTOM_TYPES = {
    "DATASET": "DATASET",
    "DATALOADER": "DATALOADER", 
    "OPTIMIZER": "OPTIMIZER",
    "IMAGE": "IMAGE",
}

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
    "Softmax": SoftmaxNode,
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

NODE_DISPLAY_NAME_MAPPINGS = {
    # Data nodes
    "MNISTDataset": "MNIST Dataset",
    "BatchSampler": "Batch Sampler",
    "GetBatch": "Get Batch",
    
    # Layer nodes
    "LinearLayer": "Linear Layer",
    "Conv2DLayer": "Conv2D Layer",
    "Activation": "Activation Function",
    "Dropout": "Dropout",
    "Softmax": "Softmax",
    "BatchNorm": "Batch Normalization",
    "Flatten": "Flatten",
    
    # Training nodes
    "CrossEntropyLoss": "Cross Entropy Loss",
    "Accuracy": "Calculate Accuracy",
    "SGDOptimizer": "SGD Optimizer",
    "TrainingStep": "Training Step",
    
    # Control nodes
    "CreateContext": "Create Context",
    "SetMode": "Set Mode (Train/Eval)",
    
    # Visualization
    "TensorVisualizer": "Tensor Visualizer",
}

# Register the custom types
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} ML nodes")