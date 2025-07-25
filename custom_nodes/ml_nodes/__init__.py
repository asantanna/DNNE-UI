"""
ML Nodes for DNNE
Machine Learning nodes including supervised learning and reinforcement learning
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
from .ppo_config import PPOConfig
from .ppo_agent import PPOAgent

# Define node data as tuples (key, class, display_name)
_ML_NODES = [
    # Data nodes
    ("MNISTDataset", MNISTDatasetNode, "MNIST Dataset"),
    ("BatchSampler", BatchSamplerNode, "Batch Sampler"),
    ("GetBatch", GetBatchNode, "Get Batch"),
    
    # Layer nodes
    ("Network", NetworkNode, "Neural Network"),
    ("LinearLayer", LinearLayerNode, "Linear Layer"),
    ("Conv2DLayer", Conv2DLayerNode, "Conv2D Layer"),
    ("Activation", ActivationNode, "Activation"),
    ("Dropout", DropoutNode, "Dropout"),
    ("BatchNorm", BatchNormNode, "Batch Normalization"),
    ("Flatten", FlattenNode, "Flatten"),
    
    # Training nodes
    ("CrossEntropyLoss", CrossEntropyLossNode, "Cross Entropy Loss"),
    ("Accuracy", AccuracyNode, "Accuracy"),
    ("SGDOptimizer", SGDOptimizerNode, "SGD Optimizer"),
    ("TrainingStep", TrainingStepNode, "Training Step"),
    ("EpochTracker", EpochTrackerNode, "Epoch Tracker"),
    
    # Visualization
    ("TensorVisualizer", TensorVisualizerNode, "Tensor Visualizer"),
    
    # RL nodes
    ("PPOConfig", PPOConfig, "PPO Config"),
    ("PPOAgent", PPOAgent, "PPO Agent"),
]

# Generate sorted dictionaries automatically by display name
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for key, node_class, display_name in sorted(_ML_NODES, key=lambda x: x[2]):  # Sort by display name
    NODE_CLASS_MAPPINGS[key] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[key] = display_name

# Export
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']