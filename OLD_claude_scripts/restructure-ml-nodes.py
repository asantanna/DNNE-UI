#!/usr/bin/env python3
"""
Script to restructure ml_nodes into separate files
Run this from your DNNE-UI directory
"""

import os
import shutil
from datetime import datetime

def restructure_ml_nodes():
    """Restructure ml_nodes into separate files"""
    print("=== Restructuring ML Nodes ===\n")
    
    ml_nodes_dir = "custom_nodes/ml_nodes"
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{ml_nodes_dir}/__init__.py.backup_{timestamp}"
    
    if os.path.exists(f"{ml_nodes_dir}/__init__.py"):
        shutil.copy2(f"{ml_nodes_dir}/__init__.py", backup_path)
        print(f"Created backup: {backup_path}")
    
    # File contents (from the artifacts above)
    files = {
        "__init__.py": '''"""
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
''',
        
        "base.py": '''"""
Base utilities for ML nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Import base types
from custom_nodes.robotics_nodes.robotics_types import TensorData, Context
from custom_nodes.robotics_nodes import RoboticsNodeBase

# Global context instance
context = None

def get_context():
    """Get or create global context"""
    global context
    if context is None:
        context = Context()
    return context
''',
    }
    
    # Write the files
    for filename, content in files.items():
        filepath = os.path.join(ml_nodes_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created: {filepath}")
    
    print("\n✓ Basic structure created!")
    print("\nNOTE: You'll need to copy the node class code from the artifacts into:")
    print("  - data_nodes.py")
    print("  - layer_nodes.py") 
    print("  - training_nodes.py")
    print("  - control_nodes.py")
    print("  - visualization_nodes.py")
    print("\nOr use the complete files from the artifacts in the chat.")

if __name__ == "__main__":
    restructure_ml_nodes()