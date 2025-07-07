"""
Sample node data and configurations for testing.

Provides mock node inputs, widget values, and connection data
for testing node functionality without full workflow execution.
"""

import torch
import numpy as np

# Sample node input data for different node types
LINEAR_LAYER_DATA = {
    "inputs": {
        "in_features": 784,
        "out_features": 10,
        "bias": True,
        "device": "cpu"
    },
    "widgets": {
        "weight_init": "xavier",
        "bias_init": "zeros"
    }
}

MNIST_DATASET_DATA = {
    "inputs": {
        "batch_size": 32,
        "download": False,
        "data_path": "./data"
    },
    "widgets": {
        "train": True,
        "transform": "normalize"
    }
}

NETWORK_DATA = {
    "inputs": {
        "device": "cpu"
    },
    "widgets": {
        "input_shape": [1, 28, 28],
        "num_classes": 10
    }
}

SGD_OPTIMIZER_DATA = {
    "inputs": {
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001
    },
    "widgets": {
        "nesterov": False
    }
}

CROSS_ENTROPY_LOSS_DATA = {
    "inputs": {
        "reduction": "mean",
        "ignore_index": -100
    },
    "widgets": {
        "label_smoothing": 0.0
    }
}

ISAAC_GYM_ENV_DATA = {
    "inputs": {
        "task": "Cartpole",
        "num_envs": 512,
        "device": "cpu"
    },
    "widgets": {
        "sim_device": "cpu",
        "graphics_device_id": 0,
        "headless": True
    }
}

# Sample tensor data for testing
def create_sample_tensor(shape, dtype=torch.float32, device="cpu"):
    """Create a sample tensor with given shape and properties."""
    return torch.randn(shape, dtype=dtype, device=device)

def create_sample_batch(batch_size=32, num_features=784, num_classes=10):
    """Create a sample batch of data and labels."""
    data = torch.randn(batch_size, num_features)
    labels = torch.randint(0, num_classes, (batch_size,))
    return data, labels

def create_sample_mnist_batch(batch_size=32):
    """Create a sample MNIST-like batch."""
    # MNIST images: 28x28 grayscale
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels

def create_sample_cartpole_state(num_envs=512):
    """Create sample Cartpole observation data."""
    # Cartpole has 4 state variables: [x, x_dot, theta, theta_dot]
    return torch.randn(num_envs, 4)

def create_sample_actions(num_envs=512, action_dim=1):
    """Create sample action data."""
    return torch.randint(0, 2, (num_envs, action_dim))  # Binary actions for Cartpole

# Node connection data for testing
SAMPLE_CONNECTIONS = {
    "simple": [
        ("1", "output", "2", "input")
    ],
    "training_loop": [
        ("1", "dataset", "2", "dataset"),
        ("2", "sampler", "3", "sampler"), 
        ("3", "batch", "4", "input"),
        ("4", "predictions", "5", "predictions"),
        ("3", "targets", "5", "targets"),
        ("5", "loss", "6", "loss")
    ],
    "robotics": [
        ("1", "observations", "2", "input"),
        ("2", "actions", "3", "actions"),
        ("1", "env", "3", "env")
    ]
}

# Mock export template variables
SAMPLE_TEMPLATE_VARS = {
    "linear_layer": {
        "NODE_ID": "node_1",
        "CLASS_NAME": "LinearLayer",
        "IN_FEATURES": 784,
        "OUT_FEATURES": 10,
        "BIAS": True,
        "DEVICE": "cpu"
    },
    "mnist_dataset": {
        "NODE_ID": "node_2", 
        "CLASS_NAME": "MNISTDataset",
        "BATCH_SIZE": 32,
        "DOWNLOAD": False,
        "DATA_PATH": "./data"
    },
    "network": {
        "NODE_ID": "node_3",
        "CLASS_NAME": "Network", 
        "DEVICE": "cpu",
        "INPUT_SHAPE": [1, 28, 28],
        "NUM_CLASSES": 10
    }
}

# Expected node input/output types for validation
NODE_IO_TYPES = {
    "LinearLayer": {
        "inputs": ["model"],
        "outputs": ["model"]
    },
    "MNISTDataset": {
        "inputs": [],
        "outputs": ["dataset"] 
    },
    "Network": {
        "inputs": ["input"],
        "outputs": ["predictions", "model"]
    },
    "SGDOptimizer": {
        "inputs": ["model"],
        "outputs": ["optimizer"]
    },
    "CrossEntropyLoss": {
        "inputs": ["predictions", "targets"],
        "outputs": ["loss"]
    },
    "TrainingStep": {
        "inputs": ["loss", "optimizer"],
        "outputs": ["ready_signal"]
    }
}