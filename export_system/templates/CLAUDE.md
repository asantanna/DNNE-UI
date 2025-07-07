# DNNE Template System

The template system is the **code generation engine** of DNNE's export system. It converts visual node graphs into executable Python code using a sophisticated template-based approach with variable substitution.

## Overview

Templates transform **visual node configurations** into **production-ready Python code**:
- **Node Templates**: Generate individual node classes
- **Base Templates**: Provide async queue framework  
- **Variable Substitution**: Fill templates with UI widget values
- **Connection Routing**: Wire node inputs/outputs together

## Directory Structure

```
templates/
├── base/                   # Core framework templates
│   ├── queue_framework.py  # QueueNode, SensorNode base classes
│   └── graph_runner.py     # Workflow orchestration
└── nodes/                  # Node-specific templates
    ├── mnist_dataset_queue.py      # ML data loading
    ├── network_queue.py            # Neural network execution
    ├── training_step_queue.py      # Training coordination
    ├── isaac_gym_env_queue.py      # RL environment setup
    ├── isaac_gym_step_queue.py     # RL simulation stepping
    └── or_node_queue.py            # State routing utility
```

## Template Architecture

### Template Variable System

#### **Variable Declaration**
Every template starts with variable declarations:
```python
# Template variables - replaced during export
template_vars = {
    "NODE_ID": "network_1",
    "CLASS_NAME": "NetworkNode", 
    "HIDDEN_SIZE": 128,
    "LEARNING_RATE": 0.001,
    "DEVICE": "cuda"
}
```

#### **Variable Substitution**
Variables are substituted using Python string formatting:
```python
# Template code
class {CLASS_NAME}_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.hidden_size = {HIDDEN_SIZE}
        self.learning_rate = {LEARNING_RATE}
        self.device = "{DEVICE}"

# Generated code
class NetworkNode_network_1(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.hidden_size = 128
        self.learning_rate = 0.001
        self.device = "cuda"
```

### Template Categories

#### 1. **Queue Templates** (Current Standard)
All modern templates use async queue-based execution:
```python
class {CLASS_NAME}_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input1"], optional=["input2"])
        self.setup_outputs(["output1", "output2"])
        
    async def compute(self, input1, input2=None) -> Dict[str, Any]:
        # Async computation logic
        result = await self.process_data(input1, input2)
        return {"output1": result["data"], "output2": result["metadata"]}
```

#### 2. **Legacy Standard Templates** (Deprecated)
Older synchronous templates (being phased out):
```python
class {CLASS_NAME}_{NODE_ID}:
    def compute(self, input1, input2):
        # Synchronous computation
        return (result1, result2)
```

## Template Patterns

### 1. Basic Node Template

#### **Structure**
```python
# templates/nodes/example_queue.py

# Template variables
template_vars = {
    "NODE_ID": "example_1",
    "CLASS_NAME": "ExampleNode",
    "PARAM1": 42,
    "PARAM2": "default_value"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Description of what this node does"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        
        # Setup async queues
        self.setup_inputs(required=["required_input"], optional=["optional_input"])
        self.setup_outputs(["output1", "output2"])
        
        # Initialize with template parameters
        self.param1 = {PARAM1}
        self.param2 = "{PARAM2}"
        
    async def compute(self, required_input, optional_input=None) -> Dict[str, Any]:
        """Main computation logic"""
        
        # Process inputs
        result = self.process_logic(required_input, optional_input)
        
        # Return outputs as dictionary
        return {
            "output1": result["primary"],
            "output2": result["secondary"]
        }
    
    def process_logic(self, input1, input2):
        """Helper method for computation"""
        # Node-specific processing
        return {"primary": processed_data, "secondary": metadata}
```

### 2. Trigger-Based Template (ML Training)

#### **Training Coordination Pattern**
```python
# templates/nodes/training_step_queue.py

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["loss", "optimizer"])
        self.setup_outputs(["ready"])
        
        # Training parameters
        self.learning_rate = {LEARNING_RATE}
        self.gradient_clip = {GRADIENT_CLIP}
        
    async def compute(self, loss, optimizer) -> Dict[str, Any]:
        """Perform training step and signal completion"""
        
        # Backpropagation
        loss.backward()
        
        # Gradient clipping if enabled
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 
                                         self.gradient_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Generate completion signal
        ready_signal = {
            "signal_type": "ready",
            "timestamp": time.time(), 
            "source_node": self.node_id,
            "metadata": {
                "phase": "training_complete",
                "loss_value": loss.item()
            }
        }
        
        return {"ready": ready_signal}
```

### 3. State Caching Template (RL Training)

#### **Dual-Mode Execution Pattern**
```python
# templates/nodes/isaac_gym_step_queue.py

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["sim_handle", "actions"], optional=["trigger"])
        self.setup_outputs(["observations", "rewards", "done", "info", "next_observations"])
        
        # State caching for RL synchronization
        self.cached_observations = None
        self.cached_rewards = None
        self.cached_done = None
        self.cached_info = None
        
    async def compute(self, sim_handle, actions, trigger=None) -> Dict[str, Any]:
        """Dual-mode execution: normal step vs triggered output"""
        
        if trigger is not None:
            # Triggered mode: Output cached state
            next_observations = self.cached_observations if self.cached_observations is not None else torch.zeros(1, 21)
            return {
                "observations": torch.zeros(1, 21),      # dummy
                "rewards": torch.zeros(1),               # dummy
                "done": torch.zeros(1, dtype=torch.bool), # dummy
                "info": {},                              # dummy
                "next_observations": next_observations   # cached state
            }
        
        # Normal mode: Step simulation and cache results
        observations = self._step_simulation(sim_handle, actions)
        rewards = self._compute_rewards(sim_handle)
        done = self._check_done(sim_handle)
        info = self._get_info(sim_handle)
        
        # Cache for later trigger-based output
        self.cached_observations = observations
        self.cached_rewards = rewards
        self.cached_done = done
        self.cached_info = info
        
        return {
            "observations": observations,
            "rewards": rewards,
            "done": done,
            "info": info,
            "next_observations": torch.zeros(1, 21)  # empty until triggered
        }
```

### 4. Sensor Template (Time-Based)

#### **Continuous Data Generation**
```python
# templates/nodes/imu_sensor_queue.py

class {CLASS_NAME}_{NODE_ID}(SensorNode):
    """Time-based sensor data generation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, update_rate={SAMPLE_RATE})
        self.setup_outputs(["acceleration", "angular_velocity", "orientation"])
        
        # Sensor parameters
        self.sample_rate = {SAMPLE_RATE}
        self.add_noise = {ADD_NOISE}
        self.noise_std = {NOISE_STD}
        
    async def compute(self) -> Dict[str, Any]:
        """Generate sensor data at specified rate"""
        
        # Simulate sensor readings
        acceleration = self._generate_acceleration()
        angular_velocity = self._generate_angular_velocity()
        orientation = self._generate_orientation()
        
        # Add noise if enabled
        if self.add_noise:
            acceleration = self._add_noise(acceleration, self.noise_std)
            angular_velocity = self._add_noise(angular_velocity, self.noise_std * 0.1)
        
        return {
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "orientation": orientation
        }
```

## Variable Substitution Rules

### Data Type Handling

#### **Numeric Values**
```python
# Template
self.learning_rate = {LEARNING_RATE}
self.batch_size = {BATCH_SIZE}

# Variables
{"LEARNING_RATE": 0.001, "BATCH_SIZE": 32}

# Generated
self.learning_rate = 0.001
self.batch_size = 32
```

#### **String Values**
```python
# Template (note quotes)
self.device = "{DEVICE}"
self.mode = "{TRAINING_MODE}"

# Variables  
{"DEVICE": "cuda", "TRAINING_MODE": "train"}

# Generated
self.device = "cuda"
self.mode = "train"
```

#### **Boolean Values**
```python
# Template
self.use_dropout = {USE_DROPOUT}
self.headless = {HEADLESS}

# Variables
{"USE_DROPOUT": True, "HEADLESS": False}

# Generated  
self.use_dropout = True
self.headless = False
```

#### **List/Dict Values**
```python
# Template
self.layer_sizes = {LAYER_SIZES}
self.config = {CONFIG_DICT}

# Variables
{"LAYER_SIZES": [128, 64, 32], "CONFIG_DICT": {"lr": 0.01, "momentum": 0.9}}

# Generated
self.layer_sizes = [128, 64, 32]
self.config = {"lr": 0.01, "momentum": 0.9}
```

### Special Substitution Patterns

#### **Conditional Code Generation**
```python
# Template with conditional logic
{%if USE_CUDA%}
self.device = torch.device("cuda")
{%else%}
self.device = torch.device("cpu")  
{%endif%}

# Note: This pattern is not currently implemented but could be added
```

#### **Loop-Based Generation**
```python
# Template for multiple layers
{%for i, size in enumerate(LAYER_SIZES)%}
self.layer_{i} = nn.Linear({size[0]}, {size[1]})
{%endfor%}

# Note: This pattern is not currently implemented but could be added
```

## Base Templates

### Queue Framework (`base/queue_framework.py`)

#### **QueueNode Base Class**
```python
class QueueNode:
    """Base class for all async queue-based nodes"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.input_queues = {}
        self.output_queues = {}
        self.logger = logging.getLogger(f"node_{node_id}")
        
    def setup_inputs(self, required=None, optional=None):
        """Create input queues"""
        required = required or []
        optional = optional or []
        
        for input_name in required + optional:
            self.input_queues[input_name] = asyncio.Queue(maxsize=2)
            
    def setup_outputs(self, outputs):
        """Create output queues"""
        for output_name in outputs:
            self.output_queues[output_name] = asyncio.Queue(maxsize=2)
    
    async def run(self):
        """Main execution loop"""
        while True:
            try:
                # Wait for required inputs
                inputs = await self._gather_inputs()
                
                # Execute compute method
                outputs = await self.compute(**inputs)
                
                # Send outputs to queues
                await self._send_outputs(outputs)
                
            except Exception as e:
                self.logger.error(f"Node {self.node_id} error: {e}")
                await asyncio.sleep(0.1)
```

#### **SensorNode Base Class**
```python
class SensorNode(QueueNode):
    """Base class for time-based sensor nodes"""
    
    def __init__(self, node_id: str, update_rate: float = 10.0):
        super().__init__(node_id)
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        
    async def run(self):
        """Time-based execution loop"""
        while True:
            try:
                # Generate sensor data
                outputs = await self.compute()
                
                # Send outputs
                await self._send_outputs(outputs)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Sensor {self.node_id} error: {e}")
                await asyncio.sleep(self.update_interval)
```

### Graph Runner (`base/graph_runner.py`)

#### **Workflow Orchestration**
```python
class GraphRunner:
    """Orchestrates execution of node graph workflows"""
    
    def __init__(self, nodes: Dict[str, QueueNode], connections: List[Tuple]):
        self.nodes = nodes
        self.connections = connections
        self.running = False
        
    async def run(self):
        """Start workflow execution"""
        
        # Setup connections between nodes
        self._setup_connections()
        
        # Start all nodes
        tasks = []
        for node in self.nodes.values():
            task = asyncio.create_task(node.run())
            tasks.append(task)
            
        # Wait for completion or cancellation
        self.running = True
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Workflow interrupted")
        finally:
            self.running = False
            
    def _setup_connections(self):
        """Wire node outputs to inputs based on connection list"""
        for from_node, from_output, to_node, to_input in self.connections:
            from_queue = self.nodes[from_node].output_queues[from_output]
            to_queue = self.nodes[to_node].input_queues[to_input]
            
            # Create connection task
            task = asyncio.create_task(
                self._pipe_queues(from_queue, to_queue, f"{from_node}.{from_output} -> {to_node}.{to_input}")
            )
```

## Template Development Guidelines

### Creating New Templates

#### 1. **Follow Naming Convention**
- Queue templates: `{node_type}_queue.py`
- Legacy templates: `{node_type}_template.py` (deprecated)
- Base templates: descriptive names in `base/`

#### 2. **Template Structure**
```python
# 1. Template variables declaration
template_vars = {
    "NODE_ID": "default_id",
    "CLASS_NAME": "DefaultClass",
    # ... all variables with defaults
}

# 2. Class definition with variable substitution
class {CLASS_NAME}_{NODE_ID}(QueueNode):
    
    # 3. Constructor with queue setup
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[...], optional=[...])
        self.setup_outputs([...])
        
    # 4. Async compute method
    async def compute(self, ...inputs...) -> Dict[str, Any]:
        # 5. Implementation logic
        return {"output1": result1, "output2": result2}
```

#### 3. **Variable Naming**
- Use **UPPER_CASE** for template variables
- Provide **sensible defaults** in template_vars
- Use **descriptive names**: `LEARNING_RATE` not `LR`
- Group related variables: `INPUT_SIZE`, `OUTPUT_SIZE`, `HIDDEN_SIZE`

#### 4. **Error Handling**
```python
async def compute(self, input_data):
    try:
        # Main computation
        result = self.process_data(input_data)
        return {"output": result}
        
    except Exception as e:
        self.logger.error(f"Error in {self.node_id}: {e}")
        # Return safe default or re-raise
        return {"output": torch.zeros_like(input_data)}
```

### Template Testing

#### **Manual Template Testing**
```python
# Test template variable substitution
template_content = open("template.py").read()
variables = {"NODE_ID": "test_1", "LEARNING_RATE": 0.001}

# Simple string formatting test
generated_code = template_content.format(**variables)
print(generated_code)

# Check for syntax errors
try:
    compile(generated_code, "test_template", "exec")
    print("✓ Template generates valid Python")
except SyntaxError as e:
    print(f"✗ Template syntax error: {e}")
```

#### **Integration Testing**
```python
# Test generated node execution
exec(generated_code)  # Execute generated class

# Create node instance
node_class = locals()[f"{variables['CLASS_NAME']}_{variables['NODE_ID']}"]
node = node_class("test_node")

# Test compute method
result = await node.compute(test_input)
assert "output" in result
```

## Common Template Patterns

### 1. **Device Management**
```python
# Template pattern for device handling
self.device = "{DEVICE}"
if self.device == "auto":
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

# Move tensors to device
def ensure_device(self, tensor):
    return tensor.to(self.device)
```

### 2. **Parameter Validation**
```python
# Template pattern for parameter validation
def __init__(self, node_id: str):
    super().__init__(node_id)
    
    # Validate parameters
    assert {BATCH_SIZE} > 0, "Batch size must be positive"
    assert 0 <= {DROPOUT_RATE} <= 1, "Dropout rate must be between 0 and 1"
    assert "{DEVICE}" in ["cpu", "cuda", "auto"], "Invalid device"
    
    self.batch_size = {BATCH_SIZE}
    self.dropout_rate = {DROPOUT_RATE}
```

### 3. **Logging Integration**
```python
# Template pattern for consistent logging
async def compute(self, input_data):
    self.logger.debug(f"Processing input shape: {input_data.shape}")
    
    start_time = time.time()
    result = self.process_data(input_data)
    elapsed = time.time() - start_time
    
    self.logger.info(f"Computation completed in {elapsed:.3f}s")
    return {"output": result}
```

### 4. **Configuration Flexibility**
```python
# Template pattern for optional features
def __init__(self, node_id: str):
    super().__init__(node_id)
    
    # Optional features based on template variables
    self.use_dropout = {USE_DROPOUT}
    self.use_batch_norm = {USE_BATCH_NORM}
    self.activation = "{ACTIVATION}"
    
    # Build layers based on configuration
    layers = []
    layers.append(nn.Linear({INPUT_SIZE}, {HIDDEN_SIZE}))
    
    if self.use_batch_norm:
        layers.append(nn.BatchNorm1d({HIDDEN_SIZE}))
        
    if self.activation == "relu":
        layers.append(nn.ReLU())
    elif self.activation == "tanh":
        layers.append(nn.Tanh())
        
    if self.use_dropout:
        layers.append(nn.Dropout({DROPOUT_RATE}))
        
    self.model = nn.Sequential(*layers)
```

## Troubleshooting Templates

### Common Issues

#### **Variable Substitution Errors**
```python
# ❌ Problem: Missing quotes for string values
self.device = {DEVICE}  # Generates: self.device = cuda (syntax error)

# ✅ Solution: Add quotes in template
self.device = "{DEVICE}"  # Generates: self.device = "cuda"
```

#### **Type Mismatches**
```python
# ❌ Problem: Boolean as string
self.enabled = "{ENABLED}"  # Generates: self.enabled = "True" (string, not bool)

# ✅ Solution: No quotes for booleans
self.enabled = {ENABLED}  # Generates: self.enabled = True
```

#### **Missing Variables**
```python
# ❌ Problem: Template uses undefined variable
self.size = {UNDEFINED_VAR}  # KeyError during substitution

# ✅ Solution: Define all variables in template_vars
template_vars = {
    "NODE_ID": "default",
    "UNDEFINED_VAR": 128  # Provide default
}
```

### Debugging Strategies

#### **Template Validation**
```python
# Check template for common issues
def validate_template(template_path):
    content = open(template_path).read()
    
    # Find all template variables
    import re
    variables = re.findall(r'\{(\w+)\}', content)
    
    # Check for missing template_vars
    if 'template_vars' in content:
        # Extract declared variables
        declared_vars = extract_template_vars(content)
        missing = set(variables) - set(declared_vars.keys())
        if missing:
            print(f"Missing variables: {missing}")
```

The template system provides the foundation for DNNE's code generation capabilities, enabling the transformation of visual workflows into production-ready Python applications.