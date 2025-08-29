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
        
    def _get_node_name(self, node_id: str) -> str:
        """Get a friendly name for a node like 'Barrier(75)' instead of 'dnne.node.75'"""
        if node_id not in self.graph.get('nodes', {}):
            return node_id
            
        node_config = self.graph['nodes'][node_id]
        node_class = node_config.get('class', 'Unknown')
        
        # Extract the base type (e.g., 'BarrierNode_75' -> 'Barrier')
        if '_' in node_class:
            base_type = node_class.rsplit('_', 1)[0]
            # Remove 'Node' suffix if present
            if base_type.endswith('Node'):
                base_type = base_type[:-4]
        else:
            base_type = node_class
            
        return f"{base_type}({node_id})"
    
    def _build_graph(self):
        """Build the simulation graph from structure"""
        # Create node simulators
        for node_id, node_config in self.graph.get('nodes', {}).items():
            simulator = create_simulator(node_id, node_config)
            self.nodes[node_id] = simulator
            logger.debug(f"Created simulator for {self._get_node_name(node_id)}: {simulator.__class__.__name__}")
            
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
        """Check which nodes have bootstrap capability (but don't actually bootstrap)"""
        bootstrap_capable = []
        
        for node_id, simulator in self.nodes.items():
            # Check SGD optimizer bootstrap capability
            if hasattr(simulator, 'should_bootstrap'):
                # Check if this node CAN bootstrap (not if it WILL)
                if hasattr(simulator, 'bootstrap_enabled') and simulator.bootstrap_enabled:
                    if not simulator.no_bootstrap_trigger:
                        logger.debug(f"{self._get_node_name(node_id)} has bootstrap capability")
                        bootstrap_capable.append(node_id)
                    
            # Check IsaacGym bootstrap capability
            elif hasattr(simulator, 'can_bootstrap') and simulator.can_bootstrap:
                logger.debug(f"{self._get_node_name(node_id)} can bootstrap with null action")
                bootstrap_capable.append(node_id)
                    
        return bootstrap_capable
        
    def replay_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Replay logged events through the simulation.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Analysis results including deadlock detection
        """
        logger.info(f"Starting event replay with {len(events)} events")
        
        # Check for bootstrap nodes first (just for info, not to actually bootstrap)
        bootstrap_nodes = self.check_bootstrap_nodes()
        if bootstrap_nodes:
            logger.info(f"Nodes with bootstrap capability: {bootstrap_nodes}")
            
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', 0))
        
        # Initialize time tracking with first event timestamp
        self.start_time = 0
        if sorted_events:
            self.start_time = sorted_events[0].get('timestamp', 0)
            self.current_time = self.start_time
            self.last_progress_time = self.current_time
        
        # Process each event - ONLY replay what actually happened, don't add anything!
        for event in sorted_events:
            self.current_time = event.get('timestamp', self.current_time)
            self._process_event(event)
            
            # Check for deadlock during event processing
            if self._is_deadlocked():
                self.deadlock_detected = True
                self.deadlock_time = self.current_time
                relative_time = self.current_time - self.start_time
                logger.warning(f"DEADLOCK DETECTED at t={relative_time:.3f}s (relative to start)")
                break
                
        # Final check - simplified deadlock detection
        # A system is deadlocked if:
        # 1. Most nodes are waiting
        # 2. The recorded events stopped (no more events after this)
        # This works because the deadlock monitoring stops recording when deadlocked
        
        if not self.deadlock_detected and sorted_events:
            waiting_count = sum(1 for sim in self.nodes.values() if sim.state == NodeState.WAITING)
            total_nodes = len(self.nodes)
            waiting_ratio = waiting_count / max(1, total_nodes)
            
            # For real deadlock traces from DNNE:
            # - The trace ends when deadlock is detected by the monitor
            # - Most nodes will be waiting (typically > 80%)
            # - We see the characteristic pattern of barriers waiting for SGD triggers
            
            if waiting_ratio > 0.8:  # 80% or more nodes waiting
                # This is likely a deadlock since the trace ended here
                self.deadlock_detected = True
                self.deadlock_time = sorted_events[-1].get('timestamp', 0)
                relative_time = self.deadlock_time - self.start_time
                logger.warning(f"DEADLOCK DETECTED at t={relative_time:.3f}s")
                logger.warning(f"  {waiting_count}/{total_nodes} nodes waiting ({waiting_ratio:.1%})")
                logger.warning(f"  System stopped producing events (trace ended)")
                
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
        
        logger.debug(f"{self._get_node_name(node_id)} produced output '{output_name}'")
        
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
                        
        # Update last progress time when data flows
        self.last_progress_time = self.current_time
        
    def _handle_queue_get_success(self, event: Dict[str, Any]):
        """Handle node successfully receiving input"""
        node_id = event['node_id']
        input_name = event.get('input_name', 'input')
        
        logger.debug(f"{self._get_node_name(node_id)} received input '{input_name}'")
        
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
                
        # Update last progress time when data is consumed
        self.last_progress_time = self.current_time
        
    def _handle_queue_wait(self, event: Dict[str, Any]):
        """Handle node waiting for input"""
        node_id = event['node_id']
        input_name = event.get('input_name', 'input')
        
        logger.debug(f"{self._get_node_name(node_id)} waiting for input '{input_name}'")
        
        if node_id in self.nodes:
            self.nodes[node_id].state = NodeState.WAITING
            
    def _handle_queue_blocked(self, event: Dict[str, Any]):
        """Handle node blocked on output"""
        node_id = event['node_id']
        logger.debug(f"{self._get_node_name(node_id)} blocked on output")
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
        # Only check for deadlock after some initial time has passed
        # (nodes need time to initialize)
        elapsed = self.current_time - self.start_time
        if elapsed < 0.1:  # Give 100ms for initialization
            return False
            
        # Timeout-based detection - no progress for deadlock_timeout seconds
        if self.current_time - self.last_progress_time > self.deadlock_timeout:
            return True
            
        # Don't use the all_waiting check during event replay
        # The events tell us what happened, we shouldn't second-guess them
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