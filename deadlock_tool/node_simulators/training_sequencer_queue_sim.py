"""
TrainingSequencer node simulator for deadlock analysis.
Coordinates multiple optimizer backward passes to prevent gradient conflicts.
"""

from typing import Dict, Any, List
from .base_node_sim import BaseNodeSimulator, NodeState

class TrainingSequencerSimulator(BaseNodeSimulator):
    """
    Simulates TrainingSequencer node behavior.
    
    TrainingSequencer nodes:
    - Receive multiple loss values
    - Coordinate backward passes in specified order
    - Pass through loss values to corresponding optimizers
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # TrainingSequencer configuration
        self.order = node_config.get('order', [1, 2, 3, 4])
        self.retain_graph = node_config.get('retain_graph', True)
        self.connected_losses = node_config.get('connected_losses', [1, 2])
        
        # Dynamic input/output configuration based on connected losses
        self.inputs_required = {f'loss{i}' for i in self.connected_losses}
        self.outputs = {f'to_opt{i}' for i in self.connected_losses}
        
        # Track received losses
        self.received_losses = {}
        
    def process_input(self, input_name: str, data: Any, timestamp: float = None) -> None:
        """Process incoming loss values"""
        super().process_input(input_name, data, timestamp)
        
        if input_name.startswith('loss'):
            self.received_losses[input_name] = data
            self.logger.debug(f"Received {input_name}, now have {len(self.received_losses)}/{len(self.inputs_required)} losses")
    
    def ready_to_compute(self) -> bool:
        """Check if all required losses have been received"""
        return len(self.received_losses) == len(self.inputs_required)
    
    def compute(self) -> Dict[str, Any]:
        """
        Simulate sequenced training computation.
        In reality, this coordinates backward passes in order.
        For simulation, we just pass through the losses.
        """
        self.logger.info(f"Sequencing training for {len(self.connected_losses)} optimizers")
        
        # Pass through losses to corresponding optimizers
        outputs = {}
        for i in self.connected_losses:
            loss_key = f'loss{i}'
            output_key = f'to_opt{i}'
            if loss_key in self.received_losses:
                outputs[output_key] = self.received_losses[loss_key]
                self.logger.debug(f"Passing {loss_key} to {output_key}")
        
        # Clear received losses for next iteration
        self.received_losses.clear()
        
        return outputs
    
    def reset(self) -> None:
        """Reset node state"""
        super().reset()
        self.received_losses.clear()