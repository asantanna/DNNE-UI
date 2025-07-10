"""
Training-related nodes (loss, optimizer, metrics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from .base import RoboticsNodeBase, get_context


class CrossEntropyLossNode(RoboticsNodeBase):
    """CrossEntropyLoss
    Computes cross-entropy loss between predictions and labels with accuracy calculation."""
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "predictions": ("TENSOR", {"tooltip": "Model predictions/logits tensor with shape (batch_size, num_classes). Raw output from neural network before softmax."}),
                "labels": ("TENSOR", {"tooltip": "Ground truth class labels tensor with shape (batch_size,). Integer values representing correct class indices (0 to num_classes-1)."}),
            }
        }

    RETURN_TYPES = ("TENSOR", "FLOAT")
    RETURN_NAMES = ("loss", "accuracy")
    FUNCTION = "compute_loss"
    CATEGORY = "ml"

    def compute_loss(self, predictions, labels):
        loss = F.cross_entropy(predictions, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(predictions, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        
        return (loss, accuracy)


class AccuracyNode(RoboticsNodeBase):
    """Accuracy
    Calculates classification accuracy by comparing predictions with ground truth labels."""
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "predictions": ("TENSOR", {"tooltip": "Model predictions/logits tensor with shape (batch_size, num_classes). Used to compute predicted classes via argmax."}),
                "labels": ("TENSOR", {"tooltip": "Ground truth class labels tensor with shape (batch_size,). Integer values representing correct class indices for accuracy calculation."}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT")
    RETURN_NAMES = ("accuracy", "correct", "total")
    FUNCTION = "calculate"
    CATEGORY = "ml"

    def calculate(self, predictions, labels):
        _, predicted = torch.max(predictions, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        
        return (accuracy, correct, total)


class SGDOptimizerNode(RoboticsNodeBase):
    """SGDOptimizer
    Creates SGD optimizer with configurable learning rate and momentum for training neural networks."""
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network": ("MODEL", {"tooltip": "Neural network model whose parameters will be optimized. Connect from Network node's model output."}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.01, "tooltip": "Step size for gradient descent updates. Common values: 0.001-0.1. Higher values train faster but may overshoot optimal weights."}),
                "momentum": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Momentum factor for SGD. Helps accelerate gradients in consistent directions and dampens oscillations. 0.9 is typical, 0.0 disables momentum."}),
            }
        }

    RETURN_TYPES = ("OPTIMIZER",)
    RETURN_NAMES = ("optimizer",)
    FUNCTION = "create_optimizer"
    CATEGORY = "ml"

    def create_optimizer(self, network, learning_rate, momentum):
        context = get_context()
        
        # The network input is just for UI connectivity - we still collect parameters from context
        # In the future, this could be improved to use the specific network connection
        parameters = []
        for key, module in context.memory.items():
            if isinstance(module, nn.Module):
                parameters.extend(module.parameters())
        
        if not parameters:
            raise ValueError("No parameters found to optimize. Create layers first.")
        
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)
        return (optimizer,)


class TrainingStepNode(RoboticsNodeBase):
    """TrainingStep
    Performs a complete training step: zero gradients, backward pass, and parameter update."""
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loss": ("TENSOR", {"tooltip": "Loss tensor to backpropagate. Scalar tensor (single value) computed from loss function like CrossEntropyLoss."}),
                "optimizer": ("OPTIMIZER", {"tooltip": "Optimizer instance (SGD, Adam, etc.) that will update model parameters. Connect from SGDOptimizer or similar node."}),
            }
        }

    RETURN_TYPES = ("SYNC",)
    RETURN_NAMES = ("ready",)
    FUNCTION = "training_step"
    CATEGORY = "ml"

    def training_step(self, loss, optimizer):
        loss_tensor = self.ensure_tensor(loss)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss_tensor.backward()
        
        # Update weights
        optimizer.step()
        
        # Create ready signal for synchronization
        import time
        ready_signal = {
            "signal_type": "ready",
            "timestamp": time.time(),
            "source_node": "training_step",
            "metadata": {"phase": "training_complete"}
        }
        
        return (ready_signal,)


class EpochTrackerNode(RoboticsNodeBase):
    """EpochTracker
    Tracks training progress across epochs, providing epoch statistics and convergence monitoring."""
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "epoch_stats": ("DICT", {"tooltip": "Dictionary containing current epoch statistics like batch count, running totals, etc. Usually from GetBatch or similar nodes."}),
                "loss": ("TENSOR", {"tooltip": "Current batch loss tensor for tracking training progress. Used to compute epoch averages and convergence metrics."}),
                "accuracy": ("*", {"tooltip": "Current batch accuracy (float) or accuracy metrics. Can be from CrossEntropyLoss or AccuracyNode. Used for epoch averaging."}),
            },
            "optional": {
                "max_epochs": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip": "Maximum number of training epochs. Training will stop when this limit is reached or manually interrupted."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("training_summary",)
    FUNCTION = "track_progress"
    CATEGORY = "ml"

    def track_progress(self, epoch_stats, loss, accuracy, max_epochs=10):
        # This is a placeholder for UI - actual logic is in the template
        return ({"epoch": 0, "avg_loss": 0.0, "avg_accuracy": 0.0},)