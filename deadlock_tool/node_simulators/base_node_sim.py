"""
Base simulator class for DNNE nodes in deadlock analysis.
Each node type will have its own simulator that inherits from this base class.
"""

from typing import Dict, Set, Any, Optional, List
from enum import Enum
import logging

class NodeState(Enum):
    """Possible states for a node during simulation"""
    WAITING = "WAITING"           # Waiting for inputs
    READY = "READY"               # Has all inputs, ready to execute
    EXECUTING = "EXECUTING"       # Currently processing
    BLOCKED = "BLOCKED"           # Cannot proceed (output queue full, etc.)
    COMPLETED = "COMPLETED"       # Finished execution

class BaseNodeSimulator:
    """Base class for all node simulators"""
    
    def __init__(self, node_id: str, node_config: Dict[str, Any]):
        """
        Initialize the node simulator.
        
        Args:
            node_id: Unique identifier for this node
            node_config: Configuration from graph structure (class, type, etc.)
        """
        self.node_id = node_id
        self.node_class = node_config.get('class', '')
        self.node_type = node_config.get('type', '')
        
        # State management
        self.state = NodeState.WAITING
        self.inputs_required: Set[str] = set()  # Which inputs must be present
        self.inputs_optional: Set[str] = set()  # Optional inputs
        self.inputs_available: Dict[str, Any] = {}  # Currently available input data
        self.outputs: Set[str] = set()  # Output names this node produces
        
        # Connection tracking
        self.input_connections: Dict[str, List[tuple]] = {}  # input_name -> [(source_node, output_name)]
        self.output_connections: Dict[str, List[tuple]] = {}  # output_name -> [(target_node, input_name)]
        
        # Execution tracking
        self.execution_count = 0
        self.last_execution_time = None
        
        # Logging - use node type for clarity
        # Extract base type from class name (e.g., "SGDOptimizerNode_1" -> "SGD")
        base_type = self.node_class.split('_')[0] if self.node_class else 'Node'
        base_type = base_type.replace('OptimizerNode', '').replace('Node', '')
        if not base_type:
            base_type = 'Node'
        self.logger = logging.getLogger(f"{base_type}.{self.node_id}")
        
    def add_input_connection(self, input_name: str, source_node: str, source_output: str):
        """Register an incoming connection"""
        if input_name not in self.input_connections:
            self.input_connections[input_name] = []
        self.input_connections[input_name].append((source_node, source_output))
        
    def add_output_connection(self, output_name: str, target_node: str, target_input: str):
        """Register an outgoing connection"""
        if output_name not in self.output_connections:
            self.output_connections[output_name] = []
        self.output_connections[output_name].append((target_node, target_input))
        
    def can_execute(self) -> bool:
        """
        Check if node has all required inputs to execute.
        Subclasses should override this for custom logic.
        """
        # Default: all required inputs must be available
        for input_name in self.inputs_required:
            if input_name not in self.inputs_available:
                return False
        return True
        
    def process_input(self, input_name: str, data: Any, timestamp: float = None):
        """
        Handle incoming data.
        
        Args:
            input_name: Name of the input receiving data
            data: The data being received
            timestamp: When the data was received (for event replay)
        """
        self.inputs_available[input_name] = data
        self.logger.debug(f"Received input '{input_name}' at {timestamp}")
        
        # Check if we're now ready to execute
        if self.state == NodeState.WAITING and self.can_execute():
            self.state = NodeState.READY
            self.logger.info(f"State changed to READY")
            
    def clear_input(self, input_name: str):
        """Clear a specific input after consumption"""
        if input_name in self.inputs_available:
            del self.inputs_available[input_name]
            
    def clear_all_inputs(self):
        """Clear all inputs (typically after execution)"""
        self.inputs_available.clear()
        
    def execute(self) -> Dict[str, Any]:
        """
        Simulate node execution.
        
        Returns:
            Dict mapping output names to produced data
        """
        raise NotImplementedError(
            f"FAIL-FAST: Node {self.node_id} ({self.node_class}) has no simulator!\n"
            f"The {self.__class__.__name__} base class cannot simulate execution.\n"
            f"A specific simulator must be implemented for node type: {self.node_type}"
        )
        
    def post_execute(self):
        """
        Called after execution completes.
        Default behavior: clear inputs and return to waiting state.
        """
        self.execution_count += 1
        self.clear_all_inputs()
        self.state = NodeState.WAITING
        
    def reset(self):
        """Reset node to initial state"""
        self.state = NodeState.WAITING
        self.inputs_available.clear()
        self.execution_count = 0
        self.last_execution_time = None
        
    def get_waiting_for(self) -> Set[str]:
        """Get list of inputs this node is waiting for"""
        missing = set()
        for input_name in self.inputs_required:
            if input_name not in self.inputs_available:
                missing.add(input_name)
        return missing
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging"""
        return {
            'node_id': self.node_id,
            'class': self.node_class,
            'state': self.state.value,
            'inputs_required': list(self.inputs_required),
            'inputs_available': list(self.inputs_available.keys()),
            'waiting_for': list(self.get_waiting_for()),
            'execution_count': self.execution_count
        }
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.node_id}, state={self.state.value})"