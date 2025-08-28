"""
Factory for creating appropriate node simulators based on node class.
"""

from typing import Dict, Any, Optional
import logging

# Import all simulator types
from .base_node_sim import BaseNodeSimulator
from .barrier_node_queue_sim import BarrierNodeSimulator
from .eat_n_node_queue_sim import EatNNodeSimulator
from .concat_node_queue_sim import ConcatNodeSimulator
from .sgd_optimizer_queue_sim import SGDOptimizerSimulator
from .isaac_gym_sim_queue_sim import IsaacGymSimulator
from .network_node_queue_sim import NetworkNodeSimulator

# Logger for factory
logger = logging.getLogger("SimulatorFactory")

# Registry mapping base node types to simulator classes
SIMULATOR_REGISTRY = {
    'BarrierNode': BarrierNodeSimulator,
    'Eat_NNode': EatNNodeSimulator,
    'ConcatNode': ConcatNodeSimulator,
    'SplitNode': BaseNodeSimulator,  # Use base for now
    'SGDOptimizerNode': SGDOptimizerSimulator,
    'IsaacGymSimNode': IsaacGymSimulator,
    'NetworkNode': NetworkNodeSimulator,
    'CustomComputationNode': NetworkNodeSimulator,  # Treat as network for now
    'SimulationTracker': BaseNodeSimulator,  # Use base for passive nodes
    'TensorNode': BaseNodeSimulator,  # Use base for simple data nodes
}

def extract_base_class(node_class: str) -> str:
    """
    Extract base class name from full node class.
    
    Examples:
        'BarrierNode_74' -> 'BarrierNode'
        'SGDOptimizerNode_40' -> 'SGDOptimizerNode'
        'Eat_NNode_73' -> 'Eat_NNode'
    """
    # Split by underscore and rejoin all but the last part (node ID)
    parts = node_class.split('_')
    
    # Handle special case of Eat_NNode which has underscore in base name
    if node_class.startswith('Eat_NNode'):
        return 'Eat_NNode'
        
    # For most nodes, remove the numeric suffix
    if parts and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    
    # Fallback to full class name
    return node_class

def create_simulator(node_id: str, node_config: Dict[str, Any]) -> BaseNodeSimulator:
    """
    Factory function to create appropriate simulator for a node.
    
    Args:
        node_id: Unique identifier for the node
        node_config: Node configuration from graph structure
        
    Returns:
        Appropriate simulator instance for the node type
    """
    node_class = node_config.get('class', '')
    base_class = extract_base_class(node_class)
    
    # Look up simulator class
    simulator_class = SIMULATOR_REGISTRY.get(base_class)
    
    if simulator_class:
        logger.debug(f"Creating {simulator_class.__name__} for {node_id} (class: {node_class})")
        return simulator_class(node_id, node_config)
    else:
        logger.warning(f"No specific simulator for {base_class}, using BaseNodeSimulator for {node_id}")
        return BaseNodeSimulator(node_id, node_config)

def register_simulator(base_class: str, simulator_class):
    """
    Register a new simulator type.
    
    Args:
        base_class: Base node class name (e.g., 'CustomNode')
        simulator_class: Simulator class to use for this node type
    """
    SIMULATOR_REGISTRY[base_class] = simulator_class
    logger.info(f"Registered {simulator_class.__name__} for {base_class}")

def get_available_simulators() -> Dict[str, type]:
    """Get dictionary of all registered simulators"""
    return SIMULATOR_REGISTRY.copy()