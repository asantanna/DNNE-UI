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
class LinearLayerExporter(BaseNodeExporter):
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

ComfyUI's pipeline corrupts slot indices to 0. The exporter:
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
class CustomExporter(BaseNodeExporter):
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