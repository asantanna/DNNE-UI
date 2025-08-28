"""
Dataflow simulator for deadlock analysis.
Replays events through node simulators to detect deadlocks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
import time

from node_simulators import create_simulator, NodeState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataflowSimulator")

class DataflowSimulator:
    """
    Simulates dataflow through a DNNE graph using node simulators.
    Replays events to detect deadlocks and analyze root causes.
    """
    
    def __init__(self, graph_structure: Dict[str, Any]):
        """
        Initialize simulator with graph structure.
        
        Args:
            graph_structure: Graph with 'nodes' and 'connections'
        """
        self.graph = graph_structure
        self.nodes: Dict[str, Any] = {}  # node_id -> simulator
        self.connections: List[List] = []  # List of [source, output, target, input]
        
        # Event tracking
        self.current_time = 0.0
        self.last_progress_time = 0.0
        self.event_history = []
        
        # Data flow tracking
        self.pending_data: Dict[Tuple, Any] = {}  # (source, output, target, input) -> data
        self.blocked_nodes: Set[str] = set()
        
        # Deadlock detection
        self.deadlock_detected = False
        self.deadlock_time = None
        self.deadlock_timeout = 5.0  # seconds without progress = deadlock
        
        # Initialize graph
        self._build_graph()
        
    def _build_graph(self):
        """Build the simulation graph from structure"""
        # Create node simulators
        for node_id, node_config in self.graph.get('nodes', {}).items():
            simulator = create_simulator(node_id, node_config)
            self.nodes[node_id] = simulator
            logger.debug(f"Created simulator for {node_id}: {simulator.__class__.__name__}")
            
        # Parse connections
        self.connections = self.graph.get('connections', [])
        
        # Build connection mappings for each node
        for conn in self.connections:
            if len(conn) != 4:
                logger.warning(f"Invalid connection format: {conn}")
                continue
                
            source_id, output_name, target_id, input_name = conn
            
            # Register connections with nodes
            if source_id in self.nodes:
                self.nodes[source_id].add_output_connection(output_name, target_id, input_name)
            if target_id in self.nodes:
                self.nodes[target_id].add_input_connection(input_name, source_id, output_name)
                
                # For Concat nodes, track required inputs
                if hasattr(self.nodes[target_id], 'set_expected_inputs'):
                    # Collect all input names for this node
                    input_names = set()
                    for c in self.connections:
                        if c[2] == target_id:  # This connection targets this node
                            input_names.add(c[3])  # Add the input name
                    if input_names:
                        self.nodes[target_id].set_expected_inputs(input_names)
                        
        logger.info(f"Built graph with {len(self.nodes)} nodes and {len(self.connections)} connections")
        
    def check_bootstrap_nodes(self):
        """Check for nodes that can bootstrap and start them"""
        bootstrapped = []
        
        for node_id, simulator in self.nodes.items():
            # Check SGD optimizer bootstrap
            if hasattr(simulator, 'should_bootstrap') and simulator.should_bootstrap():
                logger.info(f"Node {node_id} can bootstrap")
                if hasattr(simulator, 'send_bootstrap'):
                    outputs = simulator.send_bootstrap()
                    self._process_outputs(node_id, outputs, 0.0)
                    bootstrapped.append(node_id)
                    
            # Check IsaacGym bootstrap
            elif hasattr(simulator, 'can_bootstrap') and simulator.can_bootstrap:
                if not simulator.bootstrapped:
                    logger.info(f"Node {node_id} can bootstrap with null action")
                    bootstrapped.append(node_id)
                    # IsaacGym bootstrap happens during execution
                    
        return bootstrapped
        
    def replay_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Replay logged events through the simulation.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Analysis results including deadlock detection
        """
        logger.info(f"Starting event replay with {len(events)} events")
        
        # Check for bootstrap nodes first
        bootstrap_nodes = self.check_bootstrap_nodes()
        if bootstrap_nodes:
            logger.info(f"Bootstrap nodes: {bootstrap_nodes}")
            
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', 0))
        
        # Process each event
        for event in sorted_events:
            self.current_time = event.get('timestamp', self.current_time)
            self._process_event(event)
            
            # Check for deadlock
            if self._is_deadlocked():
                self.deadlock_detected = True
                self.deadlock_time = self.current_time
                logger.warning(f"DEADLOCK DETECTED at t={self.current_time:.3f}s")
                break
                
        # Final analysis
        return self._analyze_results()
        
    def _process_event(self, event: Dict[str, Any]):
        """Process a single event"""
        event_type = event.get('event_type', '')
        node_id = event.get('node_id', '')
        
        if event_type == 'QUEUE_PUT':
            self._handle_queue_put(event)
        elif event_type == 'QUEUE_GET_SUCCESS':
            self._handle_queue_get_success(event)
        elif event_type == 'QUEUE_GET_WAIT':
            self._handle_queue_wait(event)
        elif event_type == 'QUEUE_PUT_BLOCKED':
            self._handle_queue_blocked(event)
            
        self.event_history.append(event)
        
    def _handle_queue_put(self, event: Dict[str, Any]):
        """Handle node producing output"""
        node_id = event['node_id']
        output_name = event.get('output_name', 'output')
        
        logger.debug(f"Node {node_id} produced output '{output_name}'")
        
        # Find connections from this output
        for conn in self.connections:
            if conn[0] == node_id and conn[1] == output_name:
                target_id, input_name = conn[2], conn[3]
                
                # Add to pending data
                key = (node_id, output_name, target_id, input_name)
                self.pending_data[key] = {
                    'timestamp': event['timestamp'],
                    'data': event.get('data', {})
                }
                
                # Deliver to target simulator
                if target_id in self.nodes:
                    simulator = self.nodes[target_id]
                    simulator.process_input(input_name, event.get('data', {}), event['timestamp'])
                    
                    # Check if target can now execute
                    if simulator.can_execute() and simulator.state == NodeState.READY:
                        self._try_execute_node(target_id)
                        
        self.last_progress_time = self.current_time
        
    def _handle_queue_get_success(self, event: Dict[str, Any]):
        """Handle node successfully receiving input"""
        node_id = event['node_id']
        input_name = event.get('input_name', 'input')
        
        logger.debug(f"Node {node_id} received input '{input_name}'")
        
        # Remove from pending data
        for key in list(self.pending_data.keys()):
            if key[2] == node_id and key[3] == input_name:
                del self.pending_data[key]
                break
                
        # Update simulator
        if node_id in self.nodes:
            simulator = self.nodes[node_id]
            # Input already processed in queue_put, check execution
            if simulator.can_execute() and simulator.state == NodeState.READY:
                self._try_execute_node(node_id)
                
        self.last_progress_time = self.current_time
        
    def _handle_queue_wait(self, event: Dict[str, Any]):
        """Handle node waiting for input"""
        node_id = event['node_id']
        input_name = event.get('input_name', 'input')
        
        logger.debug(f"Node {node_id} waiting for input '{input_name}'")
        
        if node_id in self.nodes:
            self.nodes[node_id].state = NodeState.WAITING
            
    def _handle_queue_blocked(self, event: Dict[str, Any]):
        """Handle node blocked on output"""
        node_id = event['node_id']
        logger.debug(f"Node {node_id} blocked on output")
        self.blocked_nodes.add(node_id)
        
    def _try_execute_node(self, node_id: str):
        """Try to execute a node if ready"""
        simulator = self.nodes[node_id]
        
        if not simulator.can_execute():
            return
            
        try:
            # Execute the node
            outputs = simulator.execute()
            
            # Process outputs
            if outputs:
                self._process_outputs(node_id, outputs, self.current_time)
                
            # Post-execution cleanup
            simulator.post_execute()
            
        except NotImplementedError as e:
            # FAIL FAST on missing simulators
            logger.error(f"CRITICAL: {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing {node_id}: {e}")
            # Still fail fast on unexpected errors
            raise RuntimeError(f"Execution failed for {node_id}: {e}") from e
            
    def _process_outputs(self, node_id: str, outputs: Dict[str, Any], timestamp: float):
        """Process outputs from node execution"""
        for output_name, data in outputs.items():
            # Create synthetic QUEUE_PUT event
            synthetic_event = {
                'event_type': 'QUEUE_PUT',
                'node_id': node_id,
                'output_name': output_name,
                'timestamp': timestamp,
                'data': data
            }
            self._handle_queue_put(synthetic_event)
            
    def _is_deadlocked(self) -> bool:
        """Check if system is deadlocked"""
        # Simple timeout-based detection for now
        if self.current_time - self.last_progress_time > self.deadlock_timeout:
            return True
            
        # Check if all nodes are waiting and no pending data
        all_waiting = all(
            node.state == NodeState.WAITING 
            for node in self.nodes.values()
        )
        
        if all_waiting and not self.pending_data:
            # Nothing can progress
            return True
            
        return False
        
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze simulation results"""
        # Collect waiting nodes
        waiting_nodes = {}
        for node_id, simulator in self.nodes.items():
            if simulator.state == NodeState.WAITING:
                waiting_for = simulator.get_waiting_for()
                if waiting_for:
                    waiting_nodes[node_id] = {
                        'class': simulator.node_class,
                        'waiting_for': list(waiting_for),
                        'state_info': simulator.get_state_info()
                    }
                    
        # Build results
        results = {
            'deadlock_detected': self.deadlock_detected,
            'deadlock_time': self.deadlock_time,
            'simulation_time': self.current_time,
            'events_processed': len(self.event_history),
            'waiting_nodes': waiting_nodes,
            'blocked_nodes': list(self.blocked_nodes),
            'pending_data': len(self.pending_data),
            'node_states': {
                node_id: sim.state.value 
                for node_id, sim in self.nodes.items()
            }
        }
        
        return results
        
    def get_detailed_state(self) -> Dict[str, Any]:
        """Get detailed state information for debugging"""
        return {
            'nodes': {
                node_id: sim.get_state_info()
                for node_id, sim in self.nodes.items()
            },
            'pending_data': [
                f"{src}->{tgt}:{inp}" 
                for (src, out, tgt, inp) in self.pending_data.keys()
            ],
            'blocked_nodes': list(self.blocked_nodes)
        }