"""
ML node simulators for deadlock analysis.
Includes dataset loaders, samplers, loss nodes, etc.
"""

from typing import Dict, Any, Set
from .base_node_sim import BaseNodeSimulator, NodeState

class BatchSamplerNodeSimulator(BaseNodeSimulator):
    """Simulates BatchSamplerNode behavior."""
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # BatchSampler takes dataset input and outputs batches
        self.inputs_required = {'dataset'}
        self.outputs = {'batch'}
        
        self.batch_size = node_config.get('batch_size', 32)
        self.batch_count = 0
        
    def execute(self) -> Dict[str, Any]:
        """Sample a batch from the dataset."""
        self.state = NodeState.EXECUTING
        self.batch_count += 1
        
        self.logger.info(f"Sampling batch #{self.batch_count} (size={self.batch_size})")
        
        # Clear consumed inputs
        self.clear_input('dataset')
        
        return {
            'batch': {
                'type': 'batch',
                'batch_num': self.batch_count,
                'batch_size': self.batch_size
            }
        }

class CIFAR10DatasetNodeSimulator(BaseNodeSimulator):
    """Simulates CIFAR10DatasetNode behavior."""
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # No inputs required (dataset is a source)
        self.inputs_required = set()
        self.outputs = {'dataset'}
        
        self.output_count = 0
        self.max_outputs = 1000
        
    def can_execute(self) -> bool:
        """Dataset can always provide data."""
        return self.output_count < self.max_outputs
        
    def execute(self) -> Dict[str, Any]:
        """Output dataset reference."""
        self.state = NodeState.EXECUTING
        self.output_count += 1
        
        self.logger.info(f"Providing CIFAR10 dataset (output #{self.output_count})")
        
        return {
            'dataset': {
                'type': 'cifar10_dataset',
                'source_node': self.node_id,
                'output_num': self.output_count
            }
        }

class GetBatchNodeSimulator(BaseNodeSimulator):
    """Simulates GetBatchNode behavior."""
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Takes batch input, outputs data and labels
        self.inputs_required = {'batch'}
        self.outputs = {'data', 'labels'}
        
        self.batch_count = 0
        
    def execute(self) -> Dict[str, Any]:
        """Extract data and labels from batch."""
        self.state = NodeState.EXECUTING
        self.batch_count += 1
        
        self.logger.info(f"Extracting data and labels from batch #{self.batch_count}")
        
        # Clear consumed inputs
        self.clear_input('batch')
        
        return {
            'data': {
                'type': 'batch_data',
                'batch_num': self.batch_count
            },
            'labels': {
                'type': 'batch_labels', 
                'batch_num': self.batch_count
            }
        }

class LossNodeSimulator(BaseNodeSimulator):
    """Simulates CrossEntropyLoss node behavior."""
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Takes predictions and labels
        self.inputs_required = {'predictions', 'labels'}
        self.outputs = {'loss'}
        
        self.loss_count = 0
        
    def execute(self) -> Dict[str, Any]:
        """Calculate loss from predictions and labels."""
        self.state = NodeState.EXECUTING
        self.loss_count += 1
        
        self.logger.info(f"Computing loss #{self.loss_count}")
        
        # Clear consumed inputs
        self.clear_input('predictions')
        self.clear_input('labels')
        
        return {
            'loss': {
                'type': 'cross_entropy_loss',
                'loss_num': self.loss_count,
                'value': 0.5  # Mock value
            }
        }

class EpochTrackerNodeSimulator(BaseNodeSimulator):
    """Simulates EpochTrackerNode behavior."""
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Takes step_complete signals
        self.inputs_required = {'step_complete'}
        self.outputs = {'epoch_info'}
        
        self.step_count = 0
        self.epoch_count = 0
        self.steps_per_epoch = node_config.get('steps_per_epoch', 100)
        
    def execute(self) -> Dict[str, Any]:
        """Track training progress."""
        self.state = NodeState.EXECUTING
        self.step_count += 1
        
        # Check for epoch completion
        if self.step_count % self.steps_per_epoch == 0:
            self.epoch_count += 1
            self.logger.info(f"Epoch {self.epoch_count} completed")
        
        self.logger.info(f"Step {self.step_count} completed")
        
        # Clear consumed inputs
        self.clear_input('step_complete')
        
        return {
            'epoch_info': {
                'type': 'epoch_info',
                'step': self.step_count,
                'epoch': self.epoch_count
            }
        }

class BalancerNodeSimulator(BaseNodeSimulator):
    """Simulates BalancerNode behavior."""
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Takes epoch_info input and outputs trigger
        self.inputs_required = {'epoch_info'}
        self.outputs = {'trigger'}
        
        self.trigger_count = 0
        
    def execute(self) -> Dict[str, Any]:
        """Pass through epoch info as trigger."""
        self.state = NodeState.EXECUTING
        self.trigger_count += 1
        
        self.logger.info(f"Passing trigger #{self.trigger_count}")
        
        # Clear consumed inputs
        self.clear_input('epoch_info')
        
        return {
            'trigger': {
                'type': 'balancer_trigger',
                'trigger_num': self.trigger_count
            }
        }