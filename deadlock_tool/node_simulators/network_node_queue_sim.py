"""
Network node simulator for deadlock analysis.
Network nodes represent neural network forward passes.
"""

from typing import Dict, Any
from .base_node_sim import BaseNodeSimulator, NodeState

class NetworkNodeSimulator(BaseNodeSimulator):
    """
    Simulates neural network node behavior.
    
    Network nodes:
    - Receive input tensors
    - Perform forward pass (simulated)
    - Output result tensors
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Network configuration
        self.network_type = node_config.get('network_type', 'mlp')
        
        # Input/output configuration  
        self.inputs_required = {'input'}  # Standard network expects single input
        self.outputs = {'output'}
        
        # Track forward passes
        self.forward_count = 0
        
    def can_execute(self) -> bool:
        """Network can execute when it has input data"""
        return 'input' in self.inputs_available
        
    def execute(self) -> Dict[str, Any]:
        """
        Simulate forward pass through network.
        
        Returns:
            Dict with 'output' containing network output
        """
        if not self.can_execute():
            raise RuntimeError(f"Network {self.node_id} cannot execute: no input available")
            
        self.state = NodeState.EXECUTING
        input_data = self.inputs_available.get('input')
        
        self.forward_count += 1
        self.logger.info(f"Forward pass #{self.forward_count}")
        
        # Simulate network output
        output = {
            'type': 'network_output',
            'source_node': self.node_id,
            'network_type': self.network_type,
            'forward_pass': self.forward_count,
            'input_shape': input_data.get('shape') if isinstance(input_data, dict) else None
        }
        
        return {
            'output': output
        }
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'network_type': self.network_type,
            'forward_count': self.forward_count
        })
        return info