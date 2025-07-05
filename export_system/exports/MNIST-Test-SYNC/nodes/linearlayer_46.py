"""Node implementation for LinearLayer (ID: 46)"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class LinearLayerNode_46(QueueNode):
    """Linear layer with activation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input"])
        self.setup_outputs(["output"])
        
        # Create layer
        self.linear = nn.Linear(128, 10, bias=True)
        self.dropout = nn.Dropout(0) if 0 > 0 else None
        self.activation = "none"
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = self.linear.to(self.device)
        
    def get_parameters(self):
        """Return model parameters for optimizer"""
        return self.linear.parameters()
        
    async def compute(self, input) -> Dict[str, Any]:
        # Ensure input is on correct device
        x = input.to(self.device)
        
        # Flatten if needed (for MNIST)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass
        x = self.linear(x)
        
        # Activation
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        elif self.activation == "sigmoid":
            x = torch.sigmoid(x)
        
        # Dropout if training
        if self.dropout is not None:
            x = self.dropout(x)
        
        return {"output": x}
