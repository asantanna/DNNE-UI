# DNNE Export System Architecture

## Overview

The DNNE export system transforms visual node graphs into standalone, executable Python code. This is the core innovation that allows visual workflows to run efficiently on production systems, cloud providers, and robotics simulators.

## System Components

### Graph Exporter (`export_system/graph_exporter.py`)

The main orchestrator that:
1. Parses the visual workflow JSON
2. Fixes corrupted slot mappings from ComfyUI
3. Generates node implementations
4. Creates the runner script
5. Manages imports and dependencies

```python
class GraphExporter:
    def export_workflow(self, workflow_data, output_dir):
        # Parse workflow
        nodes, connections = self.parse_workflow(workflow_data)
        
        # Fix slot corruption issue
        connections = self._fix_corrupted_slots(workflow_data, connections)
        
        # Generate code for each node
        for node in nodes:
            self.generate_node_code(node)
        
        # Create runner with proper wiring
        self.generate_runner(nodes, connections)
```

### Node Templates (`export_system/templates/nodes/`)

Each node type has a corresponding template that defines its exported implementation:

```python
# Example: linear_layer_queue.py template
class LinearLayerNode_{node_id}(QueueNode):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear({input_size}, {output_size}, bias={bias})
        
    async def process(self):
        while True:
            x = await self.get_input()
            output = self.layer(x)
            await self.send_output({output_activation}(output))
```

Templates use placeholders (`{variable}`) that get replaced during export.

### Node Exporters (`export_system/node_exporters/`)

Specialized exporters for each node category:

- **ml_nodes.py**: Handles ML/neural network nodes
- **rl_nodes.py**: Handles RL-specific nodes (PPO, etc.)
- **robotics_nodes.py**: Handles robotics/simulation nodes

```python
class LinearLayerExporter(BaseExporter):
    def get_template_path(self):
        return "linear_layer_queue.py"
    
    def get_template_variables(self, node_data):
        return {
            'node_id': node_data['id'],
            'input_size': node_data['inputs']['input_size'],
            'output_size': node_data['inputs']['output_size'],
            'bias': node_data['inputs'].get('bias', True),
            'output_activation': self.get_activation_code(...)
        }
```

### Special Exporters with Virtual Node Processing

Virtual node processing is a sophisticated architectural pattern in DNNE's export system that enables configuration-only nodes to provide data to code-generating nodes without directly producing export code themselves. This system allows for flexible, modular workflow design while maintaining clean separation between UI configuration and code generation.

#### Critical Architectural Principle: Widget Encapsulation

**IMPORTANT**: A fundamental rule of the export system is that **only a node's own exporter can directly access that node's widgets**. No exporter should ever directly read another node's widget values. Instead, exporters must:

1. Find the target node's exporter class through the node registry
2. Call query methods on that exporter to retrieve needed information
3. Let each exporter handle its own widget extraction and validation

This encapsulation ensures:
- **Maintainability**: Widget structure changes only affect one exporter
- **Validation**: Each exporter validates its own widget data
- **Consistency**: All widget access goes through standardized methods
- **Debugging**: Clear responsibility boundaries for troubleshooting

Example of the principle in action:
```python
# WRONG: Network exporter directly accessing LinearLayer's widget data
layer_output_size = extract_widget_somehow(layer_node, 'output_size')  # Never do this!

# CORRECT: Network exporter querying LinearLayer exporter
exporter = export_utils.get_node_exporter(node_type)
layer_info = exporter.get_layer_pytorch_code(node_id, node_data, input_size)
layer_output_size = layer_info['output_size']
```

#### Understanding Virtual Nodes

Virtual nodes are UI-only nodes that serve as configuration containers. They differ from regular nodes in several key ways:

1. **No Direct Code Generation**: Virtual nodes never generate standalone code. Their `get_template_name()` method returns `None`, and `prepare_template_vars()` returns an empty dictionary.

2. **Configuration Storage**: They hold configuration data in their widget values that other nodes will query and use during code generation.

3. **Dependency Requirement**: Virtual nodes cannot exist independently in an exported workflow. They must always be connected to at least one non-virtual node that will consume their configuration.

4. **Query Response Methods**: While they don't generate templates, virtual nodes implement special methods that respond to queries from non-virtual nodes during the export process.

#### Virtual Node Connection Patterns

Virtual nodes connect to non-virtual nodes in two primary patterns:

##### Direct Connection Pattern
In this pattern, a virtual configuration node connects directly to a non-virtual node that consumes its configuration. Example: PPO Agent with PPOConfig node.

```
[PPOConfig] ----config----> [PPO Agent]
(virtual)                   (non-virtual, generates training code)
```

##### Chain Pattern
In this pattern, multiple virtual nodes form a chain that starts and ends with non-virtual nodes. The non-virtual node at the beginning queries the entire chain to collect configuration. Example: Network Node with LinearLayer nodes.

```
[Network] --layers--> [LinearLayer] --output--> [LinearLayer] --output--> [LinearLayer] --output--> [Network]
(non-virtual)         (virtual)                  (virtual)                 (virtual)              (same node, completing the chain)
```

#### Implementation Details

##### Virtual Node Exporter Structure

Virtual node exporters follow a specific pattern:

```python
class LinearLayerExporter(ExportableNode):
    # Virtual nodes are marked in the visual node with is_virtual=True
    
    @classmethod
    def get_template_name(cls):
        # Virtual nodes don't have templates
        return None
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Virtual nodes don't generate code via templates
        return {}
    
    @classmethod
    def get_layer_pytorch_code(cls, node_id, node_data, input_size=None):
        """Special method called by Network node to get layer configuration"""
        # Extract parameters from widgets
        param_specs = [
            {'name': 'output_size', 'widget_index': 0},
            {'name': 'bias', 'widget_index': 1},
            {'name': 'activation', 'widget_index': 2},
            # ... more parameters
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Return configuration that the Network node will use
        return {
            'layer_code': f"nn.Linear({input_size}, {params['output_size']}, bias={params['bias']})",
            'activation_code': activation_mapping[params['activation']],
            'output_size': params['output_size']
        }
```

##### Non-Virtual Node Querying Virtual Nodes

Non-virtual nodes that consume virtual node configurations implement sophisticated querying logic that **never directly accesses another node's widgets**:

```python
class NetworkExporter(ExportableNode):
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        layer_definitions = []
        
        # Start at the "layers" output and follow the chain
        current_node_id = export_utils.follow_node_connection(node_id, "layers")
        
        # Follow the chain of virtual layer nodes
        visited = set()
        input_size = initial_input_size  # Determined from Network's input
        
        while current_node_id and current_node_id != node_id and current_node_id not in visited:
            visited.add(current_node_id)
            
            # Get the current node's data
            node = export_utils.get_node_by_id(current_node_id)
            if not node:
                break
            
            # CRITICAL: Get the exporter for this node type
            # We NEVER directly extract another node's widgets!
            node_type = node.get('class_type')
            exporter = export_utils.get_node_exporter(node_type)
            
            # Query the virtual node's exporter for its configuration
            # The exporter handles all widget access internally
            if exporter and hasattr(exporter, 'get_layer_pytorch_code'):
                layer_info = exporter.get_layer_pytorch_code(current_node_id, node, input_size)
                
                # Add layer definition to our collection
                layer_definitions.append(f"layers.append({layer_info['layer_code']})")
                
                if layer_info.get('activation_code'):
                    layer_definitions.append(f"layers.append({layer_info['activation_code']})")
                
                # Update input size for next layer
                input_size = layer_info['output_size']
            
            # Follow to the next node in the chain
            current_node_id = export_utils.follow_node_connection(current_node_id, "output")
        
        # Now generate template variables with collected layer definitions
        return {
            "LAYER_DEFINITIONS": "\n".join(layer_definitions),
            # ... other template variables
        }
```

#### Case Study: Network Node Layer Processing

The Network node demonstrates the chain pattern for virtual node processing:

1. **Chain Discovery**: The Network node starts at its "layers" output connection and follows the chain of LinearLayer nodes.

2. **Layer Querying**: For each LinearLayer node in the chain, it calls the special `get_layer_pytorch_code()` method to retrieve the layer's configuration.

3. **Size Propagation**: The Network tracks the output size of each layer and passes it as the input size to the next layer, ensuring proper dimension matching.

4. **Code Generation**: After collecting all layer definitions, the Network node generates a complete PyTorch Sequential model with all the layers properly configured.

5. **Chain Termination**: The chain ends when it loops back to the Network node itself or encounters a non-layer node.

Key utility function used:
```python
def follow_node_connection(node_id: str, output_name: str) -> Optional[str]:
    """Follow a connection from a node's output to find the connected node."""
    # Gets export context to access links
    # Finds the output slot index for the named output
    # Searches links for connection from that output slot
    # Returns the ID of the connected node
```

#### Case Study: Isaac Sim Configuration Extraction

The Isaac Sim node demonstrates the direct connection pattern while respecting widget encapsulation:

1. **Config Node Discovery**: The Isaac Sim node looks for nodes connected to its `env_config` input slot.

2. **Own Widget Extraction**: The Isaac Sim exporter uses `get_node_parameters_batch()` to extract its **own** widget values.

3. **Config Node Widget Access**: The current implementation incorrectly accesses the connected config node's widgets directly using `get_node_parameters_batch()`. This violates the widget encapsulation principle and needs to be refactored to call a query method on the IsaacGymEnvs exporter instead.

4. **Schema Resolution**: The Isaac Sim node properly queries the config node's output schema through its exporter methods.

5. **Fail-Fast Validation**: The exporter validates that all required configuration parameters are present, throwing clear errors if any are missing.

Example of config extraction (showing the current incorrect implementation):
```python
# Find connected IsaacGymEnvs node through links
config_node = None
if all_links:
    for link in all_links:
        if len(link) >= 5 and str(link[3]) == str(node_id) and link[4] == 0:  # env_config is input 0
            source_node_id = str(link[1])
            # Find the source node
            for node in all_nodes:
                if str(node.get('id')) == source_node_id:
                    config_node = node
                    break
            break

if config_node:
    # INCORRECT: This violates widget encapsulation by directly reading config node widgets
    # TODO: Refactor to have IsaacGymEnvsExporter provide a get_env_config() method
    # that Isaac Sim can call, maintaining proper encapsulation
    param_specs = [
        {'name': 'task', 'widget_index': 0},
        {'name': 'num_envs', 'widget_index': 3},
        # ... more parameters
    ]
    # This still uses the standardized method, but ideally would call
    # IsaacGymEnvsExporter.get_env_config() instead
    config_params = cls.get_node_parameters_batch(config_node, param_specs)
else:
    raise ValueError(f"IsaacGymSim node {node_id} has no connected IsaacGymEnvs configuration node.")
```

#### PPO Agent Configuration Collection

The PPO Agent demonstrates a more complex pattern where it collects configuration from multiple virtual nodes. Note that the current implementation has some direct widget access that should ideally be refactored to use proper query methods:

1. **Multiple Config Sources**: PPO Agent accepts three different configuration inputs:
   - `env_config`: Environment configuration from IsaacGymEnvs node
   - `ppo_config`: PPO algorithm configuration from PPOConfig node  
   - `balancing_config`: Optional performance balancing configuration

2. **Config Extraction Methods**: PPO Agent implements separate extraction methods for each config type:
```python
@classmethod
def _extract_ppo_config(cls, ppo_node_id, all_nodes, all_links):
    """Extract PPO configuration from connected PPOConfig virtual node"""
    # Find the config node connected to slot 1 (ppo_config input)
    config_node = # ... find the node ...
    
    # CURRENT: Direct widget access (should be refactored)
    param_specs = [
        {'name': 'learning_rate', 'widget_index': 0},
        {'name': 'num_epochs', 'widget_index': 1},
        # ...
    ]
    raw_params = cls.get_node_parameters_batch(config_node_data, param_specs)
    
    # IDEAL: Should instead call a query method on PPOConfigExporter
    # config_exporter = export_utils.get_node_exporter('PPOConfig')
    # ppo_config = config_exporter.get_ppo_config(config_node_id, config_node_data)
    
    return ppo_config
```

3. **Config Merging**: The agent merges configurations from multiple sources with task-specific YAML configurations, creating a comprehensive training configuration.

**Note on Current Implementation**: The PPO Agent currently accesses virtual node widgets directly using `get_node_parameters_batch()`. While this uses the standardized parameter extraction method, it would be better architecturally if each virtual node exporter provided explicit query methods (like `get_ppo_config()`) that the PPO Agent could call. This would maintain proper encapsulation and make the code more maintainable.

#### Supporting Infrastructure

The export system provides utility functions to support virtual node processing:

##### Export Context Management
```python
# Global context set by GraphExporter during export
def set_export_context(context: Dict[str, Any]):
    """Set the current export context containing nodes, links, and registry"""

def get_node_by_id(node_id: str) -> Optional[Dict]:
    """Get node data by ID from current export context"""

def get_node_exporter(node_type: str):
    """Get exporter class for a node type from the registry"""
```

##### Connection Traversal
```python
def follow_node_connection(node_id: str, output_name: str) -> Optional[str]:
    """Follow a connection from a node's output to find the connected node"""
    # Used to traverse chains of virtual nodes

def get_connected_input(node_id: str, input_name: str) -> Optional[Dict]:
    """Get information about what's connected to a node's input"""
    # Used to find config nodes connected to inputs
```

##### Parameter Extraction Helpers
The `ExportableNode` base class provides standardized methods for nodes to extract their own widget values:

```python
@classmethod
def get_node_parameter(cls, node_data: Dict, param_name: str, default_value=None, widget_index: int = None):
    """Extract a parameter from THIS node's data"""
    # Used by a node's exporter to access its OWN widgets
    # Handles ComfyUI's various data formats transparently

@classmethod
def get_node_parameters_batch(cls, node_data: Dict, param_specs: List[Dict]):
    """Extract multiple parameters from THIS node's data"""
    # Efficient extraction of multiple widget values
    # Used by a node's exporter for its OWN widgets only
```

Note: These methods should only be used by an exporter to access its own node's data, never to access another node's widgets.

#### Virtual Node Schema System

Virtual nodes participate in the schema resolution system to communicate their output types:

```python
@classmethod
def get_initial_output_schema(cls, node_data):
    """Return schema describing this virtual node's outputs"""
    # Even though virtual nodes don't generate code,
    # they provide schema information for type checking
    
    output_size = cls.get_node_parameter(node_data, 'output_size', widget_index=0)
    return {
        "outputs": {
            "output": {
                "type": "tensor",
                "flattened_size": output_size,
                "dtype": "float32"
            }
        }
    }
```

This schema information helps non-virtual nodes understand the data flow through chains of virtual nodes.

#### Best Practices for Virtual Node Implementation

1. **Widget Encapsulation**: **Never directly access another node's widgets**. Always go through that node's exporter class and call appropriate query methods.

2. **Clear Separation**: Virtual nodes should only store configuration, never generate code directly.

3. **Query Methods**: Implement domain-specific query methods (like `get_layer_pytorch_code()`) that non-virtual nodes can call. These methods should be the ONLY way other nodes access your node's configuration.

4. **Own Widget Access**: Use the standardized parameter extraction methods to access your own node's widgets, but never attempt to extract another node's widget data directly.

5. **Fail-Fast Validation**: Virtual node query methods should validate parameters and fail with clear error messages.

6. **Schema Participation**: Even though they don't generate code, virtual nodes should provide accurate output schemas.

7. **Documentation**: Clearly document which non-virtual nodes consume each virtual node type and what query methods are available.

8. **Chain Termination**: When implementing chains, ensure proper termination conditions to avoid infinite loops.

#### Common Patterns and Anti-Patterns

**Good Pattern**: Virtual config nodes that can be reused by multiple non-virtual nodes:
```python
# PPOConfig can be used by different agent implementations
[PPOConfig] -----> [PPO Agent]
     |
     +-----------> [SAC Agent]  
```

**Good Pattern**: Chains that maintain clear data flow:
```python
# Each layer passes its output size to the next
[Network] -> [Linear(784->256)] -> [Linear(256->128)] -> [Linear(128->10)] -> [Network]
```

**Good Pattern**: Proper widget encapsulation through query methods:
```python
# NetworkExporter querying LinearLayerExporter
layer_exporter = export_utils.get_node_exporter('LinearLayer')
layer_config = layer_exporter.get_layer_pytorch_code(layer_id, layer_node, input_size)
```

**Anti-Pattern**: Direct widget access across node boundaries:
```python
# WRONG: NetworkExporter trying to extract LinearLayer's widget data directly
layer_output_size = some_extraction_method(layer_node, 'output_size')  # NEVER do this!
# Must go through LinearLayerExporter's query methods instead
```

**Anti-Pattern**: Virtual nodes generating code directly:
```python
# WRONG: Virtual nodes should never have templates
def get_template_name(cls):
    return "nodes/some_template.tpl"  # Virtual nodes shouldn't do this
```

**Anti-Pattern**: Non-virtual nodes not validating virtual node presence:
```python
# WRONG: Silently using defaults when config is missing
if not config_node:
    config = get_defaults()  # Should fail-fast instead
```

**Anti-Pattern**: Using get_node_parameters_batch on another node's data:
```python
# WRONG: PPOAgent accessing PPOConfig widgets directly
config_params = cls.get_node_parameters_batch(config_node_data, param_specs)
# Should instead call: PPOConfigExporter.get_ppo_config(...)
```

#### Debugging Virtual Node Processing

When debugging virtual node issues:

1. **Check Connection**: Verify virtual nodes are properly connected to non-virtual consumers.

2. **Validate Query Methods**: Ensure virtual nodes implement the expected query methods.

3. **Trace Parameter Extraction**: Log parameter values extracted from virtual nodes.

4. **Verify Chain Traversal**: Add logging to follow_node_connection() calls to trace chain traversal.

5. **Schema Validation**: Check that virtual nodes provide correct output schemas.

Example debugging code:
```python
# Add logging to trace virtual node queries
import logging
logger = logging.getLogger(__name__)

# In non-virtual node
current_node_id = export_utils.follow_node_connection(node_id, "layers")
logger.debug(f"Following chain from {node_id}, found {current_node_id}")

# In virtual node query method
logger.debug(f"LinearLayer {node_id} returning config: {layer_info}")
```

This virtual node processing system enables DNNE to maintain a clean separation between UI configuration and code generation while providing the flexibility needed for complex ML and robotics workflows.

### Queue Framework (`export_system/templates/base/`)

The async runtime that powers exported code:

```python
class QueueNode:
    """Base class for all exported nodes"""
    def __init__(self):
        self.input_queues = {}
        self.output_queues = {}
    
    async def get_input(self, input_name="default"):
        return await self.input_queues[input_name].get()
    
    async def send_output(self, data, output_name="default"):
        for queue in self.output_queues[output_name]:
            await queue.put(data)
```

## Export Process

### 1. Workflow Parsing

The exporter reads the ComfyUI workflow JSON:

```json
{
  "1": {
    "class_type": "PPOAgentNode",
    "inputs": {
      "observations": ["2", 0],
      "input_size": 4,
      "output_size": 1
    }
  }
}
```

### 2. Slot Corruption Fix

ComfyUI's pipeline can corrupts slot indices to 0 sometimes.
This is probably due to ComfyUI not supporting cycles in graphs,
which are ok in DNNE. To compensate, the exporter:
1. Reads the original workflow file
2. Extracts correct slot mappings
3. Restores proper connections

```python
def _fix_corrupted_slots(self, workflow_data, connections):
    # Read original JSON to get correct slots
    original = self._read_original_workflow(workflow_data)
    
    # Map connections to correct slots
    for conn in connections:
        correct_slot = self._find_correct_slot(original, conn)
        conn['to_slot'] = correct_slot
```

### 3. Code Generation

For each node:
1. Select appropriate exporter
2. Load template file
3. Extract template variables
4. Perform string substitution
5. Write generated code

```python
def generate_node_code(self, node):
    exporter = self.get_exporter(node['class_type'])
    template = exporter.load_template()
    variables = exporter.get_template_variables(node)
    code = template.format(**variables)
    self.write_node_file(node['id'], code)
```

### 4. Runner Generation

Creates the main entry point that:
1. Imports all nodes
2. Creates instances
3. Wires connections
4. Starts async execution

```python
# Generated runner.py structure
async def main():
    # Create nodes
    node_1 = PPOAgentNode_1()
    node_2 = ORNode_2()
    
    # Wire connections
    wire_nodes([
        (node_1, "output", node_2, "input"),
        # ... more connections
    ])
    
    # Start execution
    runner = GraphRunner([node_1, node_2])
    await runner.run()
```

## Export Patterns

### Queue-Based Pattern

All nodes follow async queue pattern:

```python
async def process(self):
    while True:
        # Wait for input
        data = await self.get_input()
        
        # Process data
        result = self.compute(data)
        
        # Send output
        await self.send_output(result)
```

Benefits:
- Non-blocking execution
- Natural backpressure
- Easy debugging
- Scalable architecture

### State Management

Nodes can maintain state across iterations:

```python
class PPOTrainerNode(QueueNode):
    def __init__(self):
        super().__init__()
        self.buffer = []  # Persistent state
        
    async def process(self):
        while True:
            trajectory = await self.get_input()
            self.buffer.append(trajectory)
            
            if len(self.buffer) >= self.horizon:
                await self.train()
                self.buffer.clear()
```

### Multi-Input Handling

Nodes can wait for multiple inputs:

```python
async def process(self):
    while True:
        # Wait for all inputs
        obs = await self.get_input("observations")
        reward = await self.get_input("rewards")
        done = await self.get_input("dones")
        
        # Process together
        self.update(obs, reward, done)
```

## File Structure

Exported workflows follow this structure:

```
export_system/exports/WorkflowName/
├── runner.py                    # Main entry point
├── generated_nodes/
│   ├── __init__.py
│   ├── node_1.py               # Individual node implementations
│   ├── node_2.py
│   └── ...
├── framework/
│   ├── __init__.py
│   └── base.py                 # Queue framework
└── requirements.txt            # Dependencies
```

## Advanced Features

### Dynamic Imports

The exporter manages imports intelligently:

```python
def generate_imports(self, nodes):
    imports = set()
    for node in nodes:
        imports.update(node.get_required_imports())
    return sorted(imports)
```

### Parameter Processing

Complex parameter handling:

```python
def process_parameter(self, param):
    if isinstance(param, list):
        # Handle connections
        return self.resolve_connection(param)
    elif isinstance(param, dict):
        # Handle nested parameters
        return {k: self.process_parameter(v) 
                for k, v in param.items()}
    else:
        # Direct value
        return param
```

### Error Handling

Robust error management:

```python
try:
    result = await self.process_input(data)
except Exception as e:
    logger.error(f"Node {self.node_id} error: {e}")
    # Send error downstream or use default
    result = self.get_safe_default()
```

## Customization

### Custom Node Templates

Create new templates:

1. Add template to `templates/nodes/`
2. Create exporter in `node_exporters/`
3. Register in export system

```python
# my_custom_node_queue.py
class MyCustomNode_{node_id}(QueueNode):
    def __init__(self):
        super().__init__()
        self.param = {custom_param}
    
    async def process(self):
        # Custom processing
        pass
```

### Template Variables

Available in all templates:
- `{node_id}`: Unique node identifier
- `{class_name}`: Original node class
- Node-specific parameters

### Export Hooks

Customize export behavior:

```python
class CustomExporter(BaseExporter):
    def post_process_code(self, code):
        # Modify generated code
        return code.replace("old", "new")
    
    def validate_parameters(self, params):
        # Check parameter validity
        assert params['size'] > 0
```

## Best Practices

### 1. Template Design

- Keep templates minimal
- Use base classes for common functionality
- Include comprehensive error handling
- Add logging for debugging

### 2. Parameter Handling

- Validate all parameters
- Provide sensible defaults
- Handle missing values gracefully
- Type check when possible

### 3. Code Generation

- Generate readable code
- Include comments for complex logic
- Maintain consistent formatting
- Preserve variable names from UI

### 4. Performance

- Minimize queue operations
- Batch processing when possible
- Avoid unnecessary copies
- Profile generated code

## Debugging Exported Code

### Enable Logging

```python
# In exported node
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

### Queue Inspection

```python
# Debug queue state
print(f"Queue size: {queue.qsize()}")
print(f"Waiting tasks: {len(queue._getters)}")
```

### Execution Tracing

```python
# Add trace points
async def process(self):
    logger.info(f"Node {self.node_id} processing")
    data = await self.get_input()
    logger.debug(f"Received: {data.shape}")
```

## Future Enhancements

### Planned Features

1. **Compilation Optimization**
   - Merge adjacent nodes
   - Eliminate unnecessary queues
   - Inline simple operations

2. **Multi-Target Export**
   - C++ generation
   - ONNX export
   - TensorRT optimization

3. **Distributed Execution**
   - Multi-machine support
   - Cloud deployment
   - Edge device export

4. **Advanced Debugging**
   - Visual debugger
   - Queue visualization
   - Performance profiling

The export system is the bridge between visual design and production deployment, enabling DNNE workflows to run efficiently anywhere from local machines to cloud clusters to embedded robotics systems.