"""Node implementation for Network (ID: 40)"""
import asyncio
import time
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class NetworkNode_40(QueueNode):
    """Neural network with multiple layers"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input"])
        self.setup_outputs(["layers", "output", "model"])
        
        # Build network from detected layers: [{'node_id': '42', 'output_size': 128, 'bias': True, 'activation': 'relu', 'dropout': 0, 'input_size': 784}, {'node_id': '43', 'output_size': 128, 'bias': True, 'activation': 'relu', 'dropout': 0, 'input_size': 128}, {'node_id': '46', 'output_size': 10, 'bias': True, 'activation': 'none', 'dropout': 0, 'input_size': 128}]
        layers = []
        layers.append(nn.Linear(784, 128, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 128, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 10, bias=True))
        
        self.network = nn.Sequential(*layers)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self.network.to(self.device)
        
        self.logger.info(f"Created network with 3 layers: 784 -> 10")
        
    def get_parameters(self):
        """Return network parameters for optimizer"""
        return self.network.parameters()
        
    async def run(self):
        """Override run to emit model reference once at startup"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            # Emit model reference once for optimizer
            await self.send_output("model", self)
            
            # Now run the normal compute loop
            while self.running:
                # Gather all required inputs
                inputs = {}
                for input_name in self.required_inputs:
                    value = await self.input_queues[input_name].get()
                    inputs[input_name] = value
                
                # Execute compute
                start_time = time.time()
                outputs = await self.compute(**inputs)
                self.last_compute_time = time.time() - start_time
                self.compute_count += 1
                
                # Send outputs (except model which was already sent)
                for output_name, value in outputs.items():
                    if output_name != "model":
                        await self.send_output(output_name, value)
                        
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def compute(self, input) -> Dict[str, Any]:
        # Ensure input is on correct device
        x = input.to(self.device)
        
        # Flatten if needed (for MNIST)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        
        # Forward pass through the entire network
        output = self.network(x)
        
        return {
            "layers": None,  # This output is just for UI connectivity
            "output": output
        }
