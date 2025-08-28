"""
SimulationTracker node simulator for deadlock analysis.
SimulationTracker nodes collect metrics from the simulation.
"""

from typing import Dict, Any
from .base_node_sim import BaseNodeSimulator, NodeState

class SimulationTrackerSimulator(BaseNodeSimulator):
    """
    Simulates SimulationTracker node behavior.
    
    SimulationTracker nodes:
    - Collect observation, done, loss, and reward signals
    - Track episode statistics
    - Generally passive (don't block execution)
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # Tracker configuration - these nodes collect multiple metrics
        # but don't require all to execute
        self.inputs_optional = {'observation', 'done', 'loss', 'reward'}
        self.inputs_required = set()  # No specific inputs required
        
        # No outputs - tracker nodes are sinks
        self.outputs = set()
        
        # Tracking state
        self.metrics_collected = 0
        self.episode_count = 0
        
    def can_execute(self) -> bool:
        """
        Tracker can execute whenever it has ANY input.
        These nodes are passive collectors.
        """
        return len(self.inputs_available) > 0
        
    def execute(self) -> Dict[str, Any]:
        """
        Process collected metrics.
        
        Returns:
            Empty dict - trackers don't produce outputs
        """
        if not self.can_execute():
            raise RuntimeError(f"SimulationTracker {self.node_id} cannot execute: no inputs available")
            
        self.state = NodeState.EXECUTING
        
        # Count what we received
        received_types = list(self.inputs_available.keys())
        self.metrics_collected += len(received_types)
        
        # Check for episode completion
        if 'done' in self.inputs_available:
            done_signal = self.inputs_available.get('done')
            if isinstance(done_signal, dict) and done_signal.get('all_done', False):
                self.episode_count += 1
                
        self.logger.info(f"Tracked {len(received_types)} metrics: {received_types}")
        
        # Trackers don't produce output
        return {}
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        info = super().get_state_info()
        info.update({
            'metrics_collected': self.metrics_collected,
            'episode_count': self.episode_count,
            'last_inputs': list(self.inputs_available.keys())
        })
        return info