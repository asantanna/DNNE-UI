"""Node implementation for MNISTDataset (ID: 1)"""
import asyncio
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class MNISTDatasetNode_1(QueueNode):
    """MNIST dataset loader"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])  # No inputs
        self.setup_outputs(["dataset", "schema"])
        
        # Setup dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        
        # Create schema describing the dataset
        self.schema = {
            "outputs": {
                "images": {
                    "type": "tensor",
                    "shape": (28, 28),
                    "flattened_size": 784,
                    "dtype": "float32"
                },
                "labels": {
                    "type": "tensor", 
                    "shape": (),
                    "num_classes": 10,
                    "dtype": "int64"
                }
            },
            "num_samples": len(self.dataset)
        }
        
    async def compute(self) -> Dict[str, Any]:
        # Return dataset and its schema
        return {
            "dataset": self.dataset,
            "schema": self.schema
        }
    
    async def run(self):
        """Override run to emit dataset once"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            # Emit dataset once
            outputs = await self.compute()
            for output_name, value in outputs.items():
                await self.send_output(output_name, value)
            
            # Keep running but don't emit again
            while self.running:
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
