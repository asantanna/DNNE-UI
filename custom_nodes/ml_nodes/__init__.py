"""
ML Nodes for DNNE
"""

# Import all node classes
from .data_nodes import MNISTDatasetNode, BatchSamplerNode, GetBatchNode
from .layer_nodes import (
    LinearLayerNode, Conv2DLayerNode, ActivationNode, 
    DropoutNode, BatchNormNode, FlattenNode
)
from .training_nodes import (
    CrossEntropyLossNode, AccuracyNode, 
    SGDOptimizerNode, TrainingStepNode
)
from .control_nodes import CreateContextNode, SetModeNode
from .visualization_nodes import TensorVisualizerNode

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
