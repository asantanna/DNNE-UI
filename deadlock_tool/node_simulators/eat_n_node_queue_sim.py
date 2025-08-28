"""
Eat_N node simulator for deadlock analysis.
Eat_N nodes consume N inputs, then switch to passthrough mode.
"""

from typing import Dict, Any, Optional
from .base_node_sim import BaseNodeSimulator, NodeState

class EatNNodeSimulator(BaseNodeSimulator):
    """
    Simulates Eat_N node behavior.
    
    Eat_N nodes have two modes:
    1. Consume mode: Consume first N inputs without producing output
    2. Passthrough mode: After N inputs consumed, pass all subsequent inputs through
    
    When switching from consume to passthrough mode, the node sends trigger signals
    to connected nodes (typically barriers) to release held data.
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Eat_N specific configuration
        self.n = node_config.get('n', 1)  # Number of inputs to consume
        self.consumed_count = 0
        self.passthrough_mode = False
        
        # Input/output configuration
        self.inputs_required = {'input'}  # Primary data input
        self.outputs = {'output', 'trigger'}  # Data output and trigger signal
        
        # Track if we've sent the mode switch trigger
        self.mode_switch_triggered = False
        
    def can_execute(self) -> bool:
        """
        Eat_N is ready to execute whenever it has input.
        In consume mode: consumes the input
        In passthrough mode: passes it through
        """
        return 'input' in self.inputs_available
        
    def execute(self) -> Dict[str, Any]:
        """
        Execute based on current mode.
        
        Returns:
            Empty dict in consume mode (no output)
            Dict with 'output' in passthrough mode
            Dict with 'output' and 'trigger' when switching modes
        """
        if not self.can_execute():
            raise RuntimeError(f"Eat_N {self.node_id} cannot execute: no input available")
            
        self.state = NodeState.EXECUTING
        input_data = self.inputs_available.get('input')
        
        if not self.passthrough_mode:
            # Consume mode
            self.consumed_count += 1
            self.logger.info(f"Consumed input {self.consumed_count}/{self.n}")
            
            if self.consumed_count >= self.n:
                # Switch to passthrough mode
                self.passthrough_mode = True
                self.mode_switch_triggered = True
                self.logger.info(f"Switching to passthrough mode after consuming {self.n} inputs")
                
                # Send both the data and trigger signals
                return {
                    'output': input_data,
                    'trigger': {
                        'signal_type': 'eat_n_satisfied',
                        'source_node': self.node_id,
                        'consumed_count': self.consumed_count,
                        'metadata': {
                            'mode': 'passthrough_activated',
                            'n': self.n
                        }
                    }
                }
            else:
                # Still consuming, no output
                return {}
        else:
            # Passthrough mode - just pass the data through
            self.logger.debug(f"Passthrough mode - forwarding input")
            return {
                'output': input_data
            }
            
    def post_execute(self):
        """
        Clear inputs after execution but maintain mode state.
        """
        self.execution_count += 1
        self.clear_all_inputs()
        self.state = NodeState.WAITING
        # Note: We don't reset passthrough_mode or consumed_count here
        
    def reset(self):
        """Full reset to initial state"""
        super().reset()
        self.consumed_count = 0
        self.passthrough_mode = False
        self.mode_switch_triggered = False
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'n': self.n,
            'consumed_count': self.consumed_count,
            'passthrough_mode': self.passthrough_mode,
            'mode': 'passthrough' if self.passthrough_mode else f'consume({self.consumed_count}/{self.n})',
            'mode_switch_triggered': self.mode_switch_triggered
        })
        return info
        
    def is_satisfied(self) -> bool:
        """Check if the Eat_N has consumed enough inputs"""
        return self.consumed_count >= self.n
        
    def get_progress(self) -> float:
        """Get consumption progress as a percentage"""
        return min(1.0, self.consumed_count / self.n) if self.n > 0 else 1.0