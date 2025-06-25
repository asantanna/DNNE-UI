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
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "momentum": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 0.99, "step": 0.01}),
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