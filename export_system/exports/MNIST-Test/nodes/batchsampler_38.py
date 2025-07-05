"""Node implementation for BatchSampler (ID: 38)"""
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class BatchSamplerNode_38(QueueNode):
    """Batch sampler that wraps a dataset and emits batches"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["dataset", "schema"])
        self.setup_outputs(["dataloader", "schema"])
        
        # Sampler parameters
        self.batch_size = 32
        self.shuffle = True
        self.seed = 851
        
    async def compute(self, dataset, schema) -> Dict[str, Any]:
        # Create dataloader from dataset
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed) if self.shuffle else None
        )
        
        self.logger.info(f"Created dataloader with batch_size={self.batch_size}, shuffle={self.shuffle}")
        
        # Pass through the schema unchanged
        return {
            "dataloader": dataloader,
            "schema": schema
        }
