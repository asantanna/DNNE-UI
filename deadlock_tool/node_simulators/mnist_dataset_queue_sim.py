"""
MNISTDatasetNode simulator for deadlock analysis.
MNISTDatasetNode provides MNIST dataset batches.
"""

from typing import Dict, Any
from .base_node_sim import BaseNodeSimulator, NodeState

class MNISTDatasetNodeSimulator(BaseNodeSimulator):
    """
    Simulates MNISTDatasetNode behavior.
    
    MNISTDatasetNode:
    - Provides MNIST dataset batches 
    - Outputs data continuously (sensor node pattern)
    - No inputs required
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Dataset configuration
        self.batch_size = node_config.get('batch_size', 32)
        self.is_train = node_config.get('is_train', True)
        
        # No inputs required (dataset is a source)
        self.inputs_required = set()
        
        # Dataset outputs batches of data
        self.outputs = {'batch'}
        
        # Track output count
        self.output_count = 0
        self.max_outputs = 1000  # Limit for simulation
        
    def can_execute(self) -> bool:
        """
        MNISTDatasetNode can always execute (data source).
        Limited by max_outputs for simulation.
        """
        return self.output_count < self.max_outputs
        
    def execute(self) -> Dict[str, Any]:
        """
        Output a batch of MNIST data.
        
        Returns:
            Dict with batch output
        """
        self.state = NodeState.EXECUTING
        self.output_count += 1
        
        self.logger.info(f"Outputting MNIST batch #{self.output_count} (size={self.batch_size})")
        
        return {
            'batch': {
                'type': 'mnist_batch',
                'source_node': self.node_id,
                'batch_size': self.batch_size,
                'is_train': self.is_train,
                'batch_num': self.output_count
            }
        }
        
    def reset(self):
        """Reset dataset to start from beginning"""
        super().reset()
        self.output_count = 0
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'output_count': self.output_count,
            'batch_size': self.batch_size,
            'is_train': self.is_train,
            'max_outputs': self.max_outputs
        })
        return info