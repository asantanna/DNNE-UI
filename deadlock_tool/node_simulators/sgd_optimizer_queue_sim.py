"""
SGD Optimizer node simulator for deadlock analysis.
SGD nodes can optionally bootstrap the system with initial step_complete signals.
"""

from typing import Dict, Any
from .base_node_sim import BaseNodeSimulator, NodeState

class SGDOptimizerSimulator(BaseNodeSimulator):
    """
    Simulates SGD optimizer node behavior.
    
    SGD optimizer nodes:
    - Receive loss values
    - Update model parameters (simulated)
    - Send step_complete signals to trigger barriers
    - Can optionally bootstrap with initial step_complete
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # SGD configuration
        self.learning_rate = node_config.get('learning_rate', 0.001)
        self.bootstrap_enabled = node_config.get('bootstrap', True)
        self.no_bootstrap_trigger = node_config.get('no_bootstrap_trigger', False)
        
        # Track if we've sent bootstrap
        self.bootstrap_sent = False
        
        # Input/output configuration
        self.inputs_required = {'loss'}
        self.outputs = {'step_complete', 'updated_params'}
        
        # Optimization state
        self.step_count = 0
        
    def should_bootstrap(self) -> bool:
        """Check if this node should send a bootstrap signal"""
        return (self.bootstrap_enabled and 
                not self.no_bootstrap_trigger and 
                not self.bootstrap_sent)
        
    def send_bootstrap(self) -> Dict[str, Any]:
        """
        Send initial bootstrap signal to start the training loop.
        This breaks initial deadlocks by triggering barriers.
        """
        self.bootstrap_sent = True
        self.logger.info(f"Sending bootstrap step_complete signal")
        
        return {
            'step_complete': {
                'signal_type': 'step_complete',
                'source_node': self.node_id,
                'step': 0,
                'metadata': {
                    'phase': 'startup',
                    'bootstrap': True
                }
            }
        }
        
    def can_execute(self) -> bool:
        """
        SGD can execute when it has a loss value.
        Note: Bootstrap happens independently of normal execution.
        """
        return 'loss' in self.inputs_available
        
    def execute(self) -> Dict[str, Any]:
        """
        Simulate optimization step.
        
        Returns:
            Dict with 'step_complete' signal and optionally 'updated_params'
        """
        if not self.can_execute():
            raise RuntimeError(f"SGD {self.node_id} cannot execute: no loss available")
            
        self.state = NodeState.EXECUTING
        loss_value = self.inputs_available.get('loss')
        
        # Simulate parameter update
        self.step_count += 1
        self.logger.info(f"Executing optimization step {self.step_count} with loss")
        
        # Generate step_complete signal
        result = {
            'step_complete': {
                'signal_type': 'step_complete', 
                'source_node': self.node_id,
                'step': self.step_count,
                'loss_value': loss_value,
                'metadata': {
                    'learning_rate': self.learning_rate,
                    'phase': 'training'
                }
            }
        }
        
        # Optionally include updated parameters
        result['updated_params'] = {
            'type': 'parameter_update',
            'step': self.step_count,
            'source_node': self.node_id
        }
        
        return result
        
    def reset(self):
        """Reset to initial state"""
        super().reset()
        self.bootstrap_sent = False
        self.step_count = 0
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'learning_rate': self.learning_rate,
            'bootstrap_enabled': self.bootstrap_enabled,
            'no_bootstrap_trigger': self.no_bootstrap_trigger,
            'bootstrap_sent': self.bootstrap_sent,
            'step_count': self.step_count,
            'can_bootstrap': self.should_bootstrap()
        })
        return info