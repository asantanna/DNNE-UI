"""
Training-related nodes (loss, optimizer, metrics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import RoboticsNodeBase, get_context


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

    RETURN_TYPES = ("TENSOR", "FLOAT")
    RETURN_NAMES = ("loss", "accuracy")
    FUNCTION = "compute_loss"
    CATEGORY = "ml/loss"

    def compute_loss(self, predictions, labels):
        loss = F.cross_entropy(predictions, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(predictions, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        
        return (loss, accuracy)


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
                "network": ("MODEL",),  # Connection from Network node's model output
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "momentum": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.99, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("OPTIMIZER",)
    RETURN_NAMES = ("optimizer",)
    FUNCTION = "create_optimizer"
    CATEGORY = "ml/optimization"

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


class EpochTrackerNode(RoboticsNodeBase):
    """Track training progress across epochs"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "epoch_stats": ("DICT",),
                "loss": ("TENSOR",),
                "accuracy": ("*",),
            },
            "optional": {
                "max_epochs": ("INT", {"default": 10, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("training_summary",)
    FUNCTION = "track_progress"
    CATEGORY = "ml/training"

    def track_progress(self, epoch_stats, loss, accuracy, max_epochs=10):
        # This is a placeholder for UI - actual logic is in the template
        return ({"epoch": 0, "avg_loss": 0.0, "avg_accuracy": 0.0},)