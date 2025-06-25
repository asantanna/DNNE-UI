"""
Neural network layer nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import RoboticsNodeBase, get_context


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