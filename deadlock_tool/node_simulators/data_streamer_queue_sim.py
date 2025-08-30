"""
DataStreamerNode simulator for deadlock analysis.
Models CSV data streaming with synchronization modes.
"""

from typing import Dict, Any, Optional, List, Set
import logging
from .base_node_sim import BaseNodeSimulator, NodeState

logger = logging.getLogger(__name__)

class DataStreamerNodeSimulator(BaseNodeSimulator):
    """
    Simulator for DataStreamerNode.
    
    The DataStreamer node:
    - Streams data from CSV files row by row
    - Has sync modes: none (continuous), external (waits for sync), timed (frequency based)
    - Optional inputs: sync (trigger), reset (trigger)
    - Outputs: data (tensor), done (trigger), metadata (dict)
    """
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        super().__init__(node_id, node_config)
        
        # DataStreamer has optional sync and reset inputs
        self.inputs_optional = {'sync', 'reset'}
        
        # DataStreamer produces data, done, and metadata outputs
        self.outputs = {'data', 'done', 'metadata'}
        
        # Track sync mode from config
        widgets = node_config.get('widgets', {})
        self.sync_mode = widgets.get('sync_mode', 'none')
        self.auto_first_row = widgets.get('auto_first_row', True)
        
        # State tracking
        self.first_row_sent = False
        self.streaming = True
        self.row_count = 0
        self.max_rows = 1000  # Simulate finite data
        
        logger.debug(f"DataStreamer {node_id} initialized with sync_mode={self.sync_mode}")
    
    def can_execute(self) -> bool:
        """DataStreamer can execute based on sync mode."""
        # If sync_mode is 'none' or 'timed', can always execute
        if self.sync_mode in ['none', 'timed']:
            return self.streaming
            
        # If sync_mode is 'external'
        if self.sync_mode == 'external':
            # First row can be sent automatically if configured
            if self.auto_first_row and not self.first_row_sent:
                return True
            # Otherwise need sync input
            return 'sync' in self.inputs_available
        
        return False
    
    def process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming event and determine node behavior."""
        event_type = event.get('event_type', '')
        
        # Handle sync input
        if event_type == 'QUEUE_GET_SUCCESS' and event.get('input_name') == 'sync':
            self.inputs_available['sync'] = True
            if self.can_execute():
                self.state = NodeState.READY
                logger.debug(f"DataStreamer {self.node_id} ready after sync")
                
        # Handle reset input
        elif event_type == 'QUEUE_GET_SUCCESS' and event.get('input_name') == 'reset':
            self.row_count = 0
            self.streaming = True
            logger.debug(f"DataStreamer {self.node_id} reset")
            
        # Handle output events
        elif event_type == 'QUEUE_PUT':
            output_name = event.get('output_name', 'data')
            if output_name == 'data':
                self.row_count += 1
                self.first_row_sent = True
                
                # Clear sync input after use (for external mode)
                if 'sync' in self.inputs_available:
                    del self.inputs_available['sync']
                    
                # Check if we've reached end of data
                if self.row_count >= self.max_rows:
                    self.streaming = False
                    self.state = NodeState.COMPLETED
                    logger.debug(f"DataStreamer {self.node_id} reached end of data")
                    return {'done': True}  # Signal done
                else:
                    # Ready for next iteration based on mode
                    if self.can_execute():
                        self.state = NodeState.READY
                    else:
                        self.state = NodeState.WAITING
                        
        # Handle waiting for input
        elif event_type == 'QUEUE_GET_WAIT':
            input_name = event.get('input_name')
            if input_name == 'sync' and self.sync_mode == 'external':
                self.state = NodeState.WAITING
                
        return None
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed state information for debugging."""
        return {
            'state': self.state.value,
            'sync_mode': self.sync_mode,
            'row_count': self.row_count,
            'max_rows': self.max_rows,
            'streaming': self.streaming,
            'first_row_sent': self.first_row_sent,
            'inputs_available': list(self.inputs_available.keys()),
            'waiting_for': self._get_waiting_inputs()
        }
    
    def _get_waiting_inputs(self) -> List[str]:
        """Determine what inputs the node is waiting for."""
        if self.sync_mode == 'external' and not self.first_row_sent and not self.auto_first_row:
            return ['sync']
        elif self.sync_mode == 'external' and self.first_row_sent and 'sync' not in self.inputs_available:
            return ['sync']
        return []
    
    def execute(self) -> Dict[str, Any]:
        """
        Simulate DataStreamer execution - output a data row.
        
        Returns:
            Dict with data output and optionally done signal
        """
        if not self.streaming:
            return {}
            
        self.row_count += 1
        self.first_row_sent = True
        self.execution_count += 1
        
        # Simulate output data
        outputs = {'data': f'row_{self.row_count}'}
        
        # Add metadata
        outputs['metadata'] = {
            'row_number': self.row_count,
            'total_rows': self.max_rows
        }
        
        # Check if done
        if self.row_count >= self.max_rows:
            outputs['done'] = True
            self.streaming = False
            self.state = NodeState.COMPLETED
        
        logger.debug(f"DataStreamer {self.node_id} output row {self.row_count}/{self.max_rows}")
        
        return outputs