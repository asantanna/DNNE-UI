"""Node implementation for CrossEntropyLoss (ID: 51)"""
from typing import Dict, Any
import torch
import torch.nn as nn
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class LossNode_51(QueueNode):
    """Cross-entropy loss computation node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["predictions", "labels"])
        self.setup_outputs(["loss", "accuracy"])
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
    
    async def compute(self, predictions, labels) -> Dict[str, Any]:
        # Ensure tensors are on the same device
        labels = labels.to(predictions.device)
        
        # Compute loss
        loss = self.criterion(predictions, labels)
        
        # Compute accuracy for classification
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        
        # Only log in verbose mode - EpochTracker will show summaries
        import builtins
        if hasattr(builtins, 'VERBOSE') and builtins.VERBOSE:
            self.logger.info(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}")
        
        return {
            "loss": loss,
            "accuracy": accuracy
        }
