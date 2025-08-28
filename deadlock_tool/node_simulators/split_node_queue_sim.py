"""
Split node simulator for deadlock analysis.
Split nodes take one input and produce multiple outputs.
"""

from typing import Dict, Any
from .base_node_sim import BaseNodeSimulator, NodeState

class SplitNodeSimulator(BaseNodeSimulator):
    """
    Simulates split node behavior.
    
    Split nodes:
    - Take single input
    - Split into multiple outputs (typically 3 for Franka_Coop_Nodes)
    - All outputs get the same data
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Split configuration
        self.num_outputs = node_config.get('num_outputs', 3)  # Default to 3 outputs
        
        # Input/output configuration
        self.inputs_required = {'input'}
        # Outputs like output_a, output_b, output_c
        self.outputs = {f'output_{chr(97+i)}' for i in range(self.num_outputs)}
        
        # Track splits
        self.split_count = 0
        
    def can_execute(self) -> bool:
        """Split can execute when it has input"""
        return 'input' in self.inputs_available
        
    def execute(self) -> Dict[str, Any]:
        """
        Split input into multiple outputs.
        
        Returns:
            Dict with multiple outputs containing the same data
        """
        if not self.can_execute():
            raise RuntimeError(f"Split {self.node_id} cannot execute: no input available")
            
        self.state = NodeState.EXECUTING
        input_data = self.inputs_available.get('input')
        
        self.split_count += 1
        self.logger.info(f"Splitting input into {self.num_outputs} outputs (split #{self.split_count})")
        
        # Create outputs - all get the same data
        outputs = {}
        for output_name in self.outputs:
            outputs[output_name] = {
                'type': 'split_output',
                'source_node': self.node_id,
                'split_count': self.split_count,
                'original_data': input_data
            }
            
        return outputs
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'num_outputs': self.num_outputs,
            'split_count': self.split_count,
            'outputs': list(self.outputs)
        })
        return info