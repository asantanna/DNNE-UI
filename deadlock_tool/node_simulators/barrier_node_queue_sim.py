"""
Barrier node simulator for deadlock analysis.
Barriers hold data until a trigger/release signal is received.
"""

from typing import Dict, Any, Optional
from .base_node_sim import BaseNodeSimulator, NodeState

class BarrierNodeSimulator(BaseNodeSimulator):
    """
    Simulates barrier node behavior.
    
    Barrier nodes:
    - Accept data on 'input'
    - Hold that data until receiving a 'release' trigger signal
    - Output the held data on 'output' when triggered
    - Reset and wait for next data/trigger cycle
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Barrier-specific configuration
        self.inputs_required = {'input'}  # Data input is required
        # Note: 'release' is not in inputs_required because it's a trigger, not data
        
        self.outputs = {'output'}
        
        # Barrier state
        self.has_data = False
        self.has_trigger = False
        self.held_data = None
        self.trigger_signal = None
        
    def process_input(self, input_name: str, data: Any, timestamp: float = None):
        """
        Handle incoming data or trigger signals.
        
        Args:
            input_name: 'input' for data, 'release' for trigger
            data: The data or trigger signal
            timestamp: When received
        """
        if input_name == 'release':
            # This is a trigger signal
            self.has_trigger = True
            self.trigger_signal = data
            self.logger.debug(f"Received trigger signal at {timestamp}")
        else:
            # Regular data input
            super().process_input(input_name, data, timestamp)
            self.has_data = True
            self.held_data = data
            self.logger.debug(f"Received data on '{input_name}' at {timestamp}")
            
        # Check if we can now execute
        if self.state == NodeState.WAITING and self.can_execute():
            self.state = NodeState.READY
            self.logger.info(f"State changed to READY (has_data={self.has_data}, has_trigger={self.has_trigger})")
            
    def can_execute(self) -> bool:
        """
        Barrier can execute when it has both data and a trigger signal.
        """
        return self.has_data and self.has_trigger
        
    def execute(self) -> Dict[str, Any]:
        """
        Release the held data.
        
        Returns:
            Dict with 'output' containing the held data
        """
        if not self.can_execute():
            raise RuntimeError(f"Barrier {self.node_id} cannot execute: has_data={self.has_data}, has_trigger={self.has_trigger}")
            
        self.state = NodeState.EXECUTING
        self.logger.info(f"Executing - releasing held data")
        
        # Release the held data
        output_data = self.held_data
        
        # Include metadata about the trigger if useful
        result = {
            'output': output_data
        }
        
        self.last_execution_time = self.trigger_signal.get('timestamp') if isinstance(self.trigger_signal, dict) else None
        
        return result
        
    def post_execute(self):
        """
        Reset barrier state after execution.
        """
        super().post_execute()
        
        # Reset barrier-specific state
        self.has_data = False
        self.has_trigger = False
        self.held_data = None
        self.trigger_signal = None
        
        self.logger.debug(f"Post-execution reset complete")
        
    def reset(self):
        """Reset to initial state"""
        super().reset()
        self.has_data = False
        self.has_trigger = False
        self.held_data = None
        self.trigger_signal = None
        
    def get_waiting_for(self) -> set:
        """Get what the barrier is waiting for"""
        waiting = set()
        if not self.has_data:
            waiting.add('input')
        if not self.has_trigger:
            waiting.add('release')
        return waiting
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'has_data': self.has_data,
            'has_trigger': self.has_trigger,
            'held_data': self.held_data is not None,
            'waiting_for': list(self.get_waiting_for())
        })
        return info