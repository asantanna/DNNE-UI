"""
ML Nodes for DNNE
"""

# Import all node classes
from .data_nodes import MNISTDatasetNode, BatchSamplerNode, GetBatchNode
from .layer_nodes import (
    NetworkNode, LinearLayerNode, Conv2DLayerNode, ActivationNode, 
    DropoutNode, BatchNormNode, FlattenNode
)
from .training_nodes import (
    CrossEntropyLossNode, AccuracyNode, 
    SGDOptimizerNode, TrainingStepNode, EpochTrackerNode
)
from .visualization_nodes import TensorVisualizerNode

# Define node data as tuples (key, class, display_name)
# Future nodes: just add to this list and they'll be automatically sorted alphabetically
_ML_NODES = [
    ("MNISTDataset", MNISTDatasetNode, "MNIST Dataset"),
    ("BatchSampler", BatchSamplerNode, "Batch Sampler"),
    ("GetBatch", GetBatchNode, "Get Batch"),
    ("Network", NetworkNode, "Neural Network"),
    ("LinearLayer", LinearLayerNode, "Linear Layer"),
    ("Conv2DLayer", Conv2DLayerNode, "Conv2D Layer"),
    ("Activation", ActivationNode, "Activation"),
    ("Dropout", DropoutNode, "Dropout"),
    ("BatchNorm", BatchNormNode, "Batch Normalization"),
    ("Flatten", FlattenNode, "Flatten"),
    ("CrossEntropyLoss", CrossEntropyLossNode, "Cross Entropy Loss"),
    ("Accuracy", AccuracyNode, "Accuracy"),
    ("SGDOptimizer", SGDOptimizerNode, "SGD Optimizer"),
    ("TrainingStep", TrainingStepNode, "Training Step"),
    ("EpochTracker", EpochTrackerNode, "Epoch Tracker"),
    ("TensorVisualizer", TensorVisualizerNode, "Tensor Visualizer"),
]

# Generate sorted dictionaries automatically by display name
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for key, node_class, display_name in sorted(_ML_NODES, key=lambda x: x[2]):  # Sort by display name
    NODE_CLASS_MAPPINGS[key] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[key] = display_name

# Export
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
