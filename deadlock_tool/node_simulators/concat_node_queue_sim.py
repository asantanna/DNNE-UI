"""
Concat node simulator for deadlock analysis.
Concat nodes require ALL inputs before producing output.
"""

from typing import Dict, Any, Set
from .base_node_sim import BaseNodeSimulator, NodeState

class ConcatNodeSimulator(BaseNodeSimulator):
    """
    Simulates concat node behavior.
    
    Concat nodes:
    - Wait for ALL defined inputs to be available
    - Concatenate inputs along specified dimension
    - Output the concatenated result
    - Clear all inputs after execution
    
    This is a common source of deadlocks when one input never arrives.
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Concat configuration
        self.concat_dim = node_config.get('concat_dim', 1)  # Default to dim 1 (features)
        
        # Extract expected inputs from config or connections
        # In Franka_Coop_Nodes, we see inputs like input_a, input_b, input_c, input_d
        # These will be set when connections are established
        self.inputs_required = set()  # Will be populated from connections
        
        self.outputs = {'output'}
        
    def set_expected_inputs(self, input_names: Set[str]):
        """
        Set the expected input names based on graph connections.
        Called during graph construction.
        """
        self.inputs_required = input_names
        self.logger.debug(f"Expected inputs set to: {input_names}")
        
    def can_execute(self) -> bool:
        """
        Concat can only execute when ALL required inputs are available.
        This is the critical constraint that often causes deadlocks.
        """
        if not self.inputs_required:
            # If no inputs defined yet, can't execute
            return False
            
        # Check if all required inputs are available
        for input_name in self.inputs_required:
            if input_name not in self.inputs_available:
                self.logger.debug(f"Cannot execute: missing input '{input_name}'")
                return False
                
        return True
        
    def execute(self) -> Dict[str, Any]:
        """
        Concatenate all inputs.
        
        Returns:
            Dict with 'output' containing concatenated data
        """
        if not self.can_execute():
            missing = self.get_waiting_for()
            raise RuntimeError(f"Concat {self.node_id} cannot execute: missing inputs {missing}")
            
        self.state = NodeState.EXECUTING
        self.logger.info(f"Executing concat with {len(self.inputs_available)} inputs")
        
        # Sort inputs by name for consistent ordering
        sorted_inputs = sorted(self.inputs_required)
        
        # Collect all input data
        input_data_list = []
        for input_name in sorted_inputs:
            data = self.inputs_available[input_name]
            input_data_list.append(data)
            
        # Simulate concatenation (in real node, this would be tensor.cat)
        # For simulation, we just track that concatenation happened
        concatenated = {
            'type': 'concatenated_tensor',
            'inputs': sorted_inputs,
            'count': len(input_data_list),
            'dim': self.concat_dim,
            'source_node': self.node_id
        }
        
        self.logger.info(f"Concatenated {len(input_data_list)} inputs along dim {self.concat_dim}")
        
        return {
            'output': concatenated
        }
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'concat_dim': self.concat_dim,
            'inputs_expected': len(self.inputs_required),
            'inputs_received': len(self.inputs_available),
            'ready': self.can_execute(),
            'missing_inputs': list(self.get_waiting_for())
        })
        return info
        
    def get_completion_percentage(self) -> float:
        """Get percentage of inputs received"""
        if not self.inputs_required:
            return 0.0
        return len(self.inputs_available) / len(self.inputs_required)