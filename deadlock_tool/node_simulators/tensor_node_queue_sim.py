"""
TensorNode simulator for deadlock analysis.
TensorNode provides constant tensor values.
"""

from typing import Dict, Any
from .base_node_sim import BaseNodeSimulator, NodeState

class TensorNodeSimulator(BaseNodeSimulator):
    """
    Simulates TensorNode behavior.
    
    TensorNode:
    - Provides constant tensor values
    - Can output immediately (no inputs required)
    - Used for providing fixed parameters or constants
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # TensorNode configuration
        self.tensor_value = node_config.get('value', 'constant_tensor')
        
        # No inputs required
        self.inputs_required = set()
        
        # Single tensor output
        self.outputs = {'tensor'}
        
        # Track output count
        self.output_count = 0
        self.has_output = False
        
    def can_execute(self) -> bool:
        """
        TensorNode can always execute (provides constant).
        But typically only outputs once.
        """
        return not self.has_output  # Only output once
        
    def execute(self) -> Dict[str, Any]:
        """
        Output the constant tensor.
        
        Returns:
            Dict with tensor output
        """
        self.state = NodeState.EXECUTING
        self.output_count += 1
        self.has_output = True
        
        self.logger.info(f"Outputting constant tensor (output #{self.output_count})")
        
        return {
            'tensor': {
                'type': 'constant_tensor',
                'source_node': self.node_id,
                'value': self.tensor_value,
                'output_count': self.output_count
            }
        }
        
    def reset(self):
        """Reset to allow outputting again"""
        super().reset()
        self.has_output = False
        self.output_count = 0
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'has_output': self.has_output,
            'output_count': self.output_count,
            'tensor_value': str(self.tensor_value)[:50]  # Truncate for display
        })
        return info