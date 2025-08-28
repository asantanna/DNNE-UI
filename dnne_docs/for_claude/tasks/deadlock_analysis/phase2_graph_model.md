# Phase 2: Graph Model Development

## Objective
Build a comprehensive model of the DNNE workflow graph that captures node behaviors, connection semantics, and synchronization requirements.

## Implementation Structure

### File Organization
Each node type will have a corresponding simulator file that mirrors its template:
- **Templates**: `/export_system/templates/nodes/{node_type}_queue.tpl`
- **Simulators**: `/deadlock_tool/node_simulators/{node_type}_queue_sim.py`

This parallel structure ensures:
- Predictable naming conventions
- Easy maintenance when templates change
- Consistent behavior between simulation and runtime
- Modular testing of individual node types

## Tasks

### 2.1 Base Simulator Framework
- [ ] Create `base_node_sim.py` with common interface
- [ ] Define standard methods: `can_execute()`, `process_input()`, `execute()`
- [ ] Implement state management (`WAITING`, `READY`, `EXECUTING`, `BLOCKED`)
- [ ] Add logging/debug capabilities

### 2.2 Node Simulator Implementation
- [ ] `barrier_node_queue_sim.py` (requires data AND trigger signal)
- [ ] `eat_n_node_queue_sim.py` (consumes N inputs, then passthrough)
- [ ] `concat_node_queue_sim.py` (requires ALL inputs before output)
- [ ] `split_node_queue_sim.py` (produces multiple outputs from single input)
- [ ] `sgd_optimizer_queue_sim.py` (bootstrap capability, step_complete signal)
- [ ] `isaac_gym_sim_queue_sim.py` (can bootstrap with null action)
- [ ] `network_node_queue_sim.py` (standard compute node)
- [ ] `custom_computation_node_queue_sim.py` (loss computation)

### 2.3 Connection Semantics
- [ ] Parse connection format: `[source_id, output_name, target_id, input_name]`
- [ ] Track which inputs are required vs optional
- [ ] Handle multi-input requirements (e.g., Concat needs all inputs)
- [ ] Model trigger connections vs data connections

### 2.4 Graph Structure
- [ ] Build adjacency lists for quick traversal
- [ ] Identify source nodes (no inputs or can self-start)
- [ ] Identify sink nodes (no outputs)
- [ ] Map subsystems (PPO, training, simulation groups)

## Implementation Notes

```python
# base_node_sim.py
class BaseNodeSimulator:
    def __init__(self, node_id, node_config):
        self.node_id = node_id
        self.node_class = node_config.get('class', '')
        self.node_type = node_config.get('type', '')
        self.state = 'WAITING'
        self.inputs_required = set()
        self.inputs_available = {}
        self.outputs = set()
        
    def can_execute(self):
        """Check if node has all required inputs"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement can_execute()")
        
    def process_input(self, input_name, data):
        """Handle incoming data"""
        self.inputs_available[input_name] = data
        
    def execute(self):
        """Simulate execution, return outputs produced"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")
        
    def reset(self):
        """Reset to initial state"""
        self.state = 'WAITING'
        self.inputs_available = {}

# barrier_node_queue_sim.py
from base_node_sim import BaseNodeSimulator

class BarrierNodeSimulator(BaseNodeSimulator):
    def __init__(self, node_id, node_config):
        super().__init__(node_id, node_config)
        self.inputs_required = {'input', 'release'}  # Needs both
        self.has_trigger = False
        self.held_data = None
        
    def can_execute(self):
        return 'input' in self.inputs_available and self.has_trigger
        
    def process_input(self, input_name, data):
        if input_name == 'release':
            self.has_trigger = True
        else:
            super().process_input(input_name, data)
            
    def execute(self):
        output_data = self.inputs_available.get('input')
        self.has_trigger = False  # Reset trigger
        return {'output': output_data}

# eat_n_node_queue_sim.py
from base_node_sim import BaseNodeSimulator

class EatNNodeSimulator(BaseNodeSimulator):
    def __init__(self, node_id, node_config):
        super().__init__(node_id, node_config)
        self.n = node_config.get('n', 1)
        self.consumed_count = 0
        self.passthrough_mode = False
        self.inputs_required = {'input'}
        
    def can_execute(self):
        if self.passthrough_mode:
            return 'input' in self.inputs_available
        else:
            # In consume mode, always ready to consume
            return 'input' in self.inputs_available
            
    def execute(self):
        if not self.passthrough_mode:
            self.consumed_count += 1
            if self.consumed_count >= self.n:
                self.passthrough_mode = True
                # Send trigger signals when switching to passthrough
                return {'output': self.inputs_available['input'], 
                        'trigger': {'signal': 'eat_n_satisfied'}}
            else:
                # Consume without output
                return {}
        else:
            # Passthrough mode
            return {'output': self.inputs_available['input']}
```

## Simulator Registry

```python
# simulator_factory.py
from barrier_node_queue_sim import BarrierNodeSimulator
from eat_n_node_queue_sim import EatNNodeSimulator
from concat_node_queue_sim import ConcatNodeSimulator
# ... import other simulators

SIMULATOR_REGISTRY = {
    'BarrierNode': BarrierNodeSimulator,
    'Eat_NNode': EatNNodeSimulator,
    'ConcatNode': ConcatNodeSimulator,
    'SplitNode': SplitNodeSimulator,
    'SGDOptimizerNode': SGDOptimizerSimulator,
    'IsaacGymSimNode': IsaacGymSimulator,
    'NetworkNode': NetworkNodeSimulator,
    'CustomComputationNode': CustomComputationSimulator,
    # Add more as implemented
}

def create_simulator(node_id, node_config):
    """Factory function to create appropriate simulator"""
    # Extract base class name (e.g., "BarrierNode" from "BarrierNode_74")
    node_class = node_config.get('class', '')
    base_class = '_'.join(node_class.split('_')[:-1])  # Remove node ID suffix
    
    if base_class in SIMULATOR_REGISTRY:
        return SIMULATOR_REGISTRY[base_class](node_id, node_config)
    else:
        # Default to basic simulator for unknown types
        return BaseNodeSimulator(node_id, node_config)
```

## Test Cases
- Simple linear pipeline
- Single barrier with trigger
- Eat_N consuming multiple inputs
- Concat waiting for all inputs
- Full Franka_Coop_Nodes graph

## Success Metrics
- Correctly models all node types in Franka_Coop_Nodes
- State transitions match actual runtime behavior
- Can identify which nodes are ready to execute at any point