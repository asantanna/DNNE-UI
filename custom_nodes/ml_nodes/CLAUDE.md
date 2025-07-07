# DNNE ML Nodes - Supervised Learning Architecture

This directory contains machine learning nodes for supervised learning tasks in DNNE. The ML nodes implement a **trigger-based training coordination system** that enables visual construction of complete training pipelines.

## Overview

The ML nodes provide:
- **Dataset Loading**: MNIST, custom datasets with batch sampling
- **Neural Networks**: PyTorch model construction and execution
- **Training Components**: Optimizers, loss functions, training steps
- **Progress Tracking**: Epoch management and statistics monitoring
- **Trigger Coordination**: Synchronous training loop execution

## Core Training Architecture

### Trigger-Based Training Loop

DNNE's supervised learning uses a **trigger-based coordination system** that ensures proper sequential execution:

```
GetBatch → Network → Loss → TrainingStep → [ready_trigger] → GetBatch (next batch)
   ↑                                          ↓
   └─────────── trigger signal ──────────────┘
```

**Key Insight**: The training loop is **reactive** - each component waits for its inputs and triggers the next component when complete.

### Why Trigger-Based?

**Problem**: Async queue systems can cause race conditions in training:
- Network might process new batch before gradients are applied
- Multiple batches might be processed simultaneously
- Training statistics become inconsistent

**Solution**: Explicit synchronization through trigger signals:
- TrainingStep completes → sends "ready" signal → triggers GetBatch
- GetBatch generates new batch only when triggered
- Creates natural **barrier synchronization** between training steps

## Node Categories

### 1. Data Nodes

#### **MNISTDataset** - Dataset Container
```python
RETURN_TYPES = ("DATASET", "INT", "INT", "INT")
RETURN_NAMES = ("dataset", "num_classes", "input_size", "num_samples")

# Features
- Loads MNIST training/test data
- Configurable train/test split
- Device placement (CPU/CUDA)
- Data normalization and preprocessing
```

#### **BatchSampler** - Batch Generation  
```python
RETURN_TYPES = ("TENSOR", "TENSOR", "SYNC")
RETURN_NAMES = ("images", "labels", "ready")

# Features  
- Triggered batch sampling (waits for training completion)
- Configurable batch size
- Shuffling and reproducible sampling
- Automatic epoch boundary detection
```

**Trigger Pattern**:
```python
async def compute(self, dataset, trigger=None):
    # Wait for trigger before generating next batch
    if trigger is None:
        # Initial batch or error state
        return generate_initial_batch()
    
    # Generate next batch only after trigger received
    return generate_next_batch()
```

### 2. Neural Network Nodes

#### **LinearLayer** - Dense Layer Construction
```python
RETURN_TYPES = ("NN_MODULE", "INT")  
RETURN_NAMES = ("layer", "output_size")

# Features
- Configurable input/output dimensions
- Weight initialization strategies
- Bias control
- Device placement
```

#### **Network** - Model Container
```python
RETURN_TYPES = ("TENSOR", "NN_MODULE")
RETURN_NAMES = ("output", "model")

# Features
- Sequential layer composition
- Forward pass execution
- Gradient computation support
- Training/eval mode management
```

**Layer Consolidation Pattern**:
The Network node **consolidates multiple LinearLayer nodes** into a single PyTorch Sequential model:
```python
# Multiple LinearLayer nodes become:
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)
```

### 3. Training Components

#### **SGDOptimizer** - Gradient Descent
```python
RETURN_TYPES = ("OPTIMIZER",)
RETURN_NAMES = ("optimizer",)

# Features
- Configurable learning rate and momentum
- Parameter group management
- Learning rate scheduling support
- Integration with PyTorch optimizers
```

#### **CrossEntropyLoss** - Loss Computation
```python
RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR")
RETURN_NAMES = ("loss", "accuracy", "predictions")

# Features
- Standard classification loss
- Accuracy computation
- Batch-wise statistics
- Device-aware computation
```

#### **TrainingStep** - Gradient Updates
```python
RETURN_TYPES = ("SYNC",)
RETURN_NAMES = ("ready",)

# Features
- Backpropagation execution
- Optimizer step coordination
- Gradient clipping support
- Training completion signaling
```

**Training Step Coordination**:
```python
async def compute(self, loss, optimizer):
    # Perform backpropagation
    loss.backward()
    
    # Update parameters
    optimizer.step()
    optimizer.zero_grad()
    
    # Signal training completion
    ready_signal = {
        "signal_type": "ready", 
        "timestamp": time.time(),
        "metadata": {"loss": loss.item()}
    }
    return (ready_signal,)
```

### 4. Control & Monitoring

#### **EpochTracker** - Training Progress
```python
RETURN_TYPES = ("INT", "FLOAT", "BOOL", "DICT")
RETURN_NAMES = ("current_epoch", "avg_loss", "training_complete", "stats")

# Features
- Epoch boundary detection
- Loss averaging and statistics
- Training completion monitoring
- Progress reporting
```

#### **SetMode** - Training/Eval Mode
```python
RETURN_TYPES = ("SYNC",)
RETURN_NAMES = ("sync",)

# Features
- Global training/eval mode setting
- Context-based model state management
- Batch norm / dropout behavior control
```

## Context System

### Implicit Context Architecture

After the **context removal migration**, DNNE uses **implicit global context**:

```python
# Before: Explicit context passing
def compute(self, input_data, context):
    model = context.get("model")
    
# After: Implicit context access  
def compute(self, input_data):
    model = get_global_context().get("model")
```

### Context Usage Patterns

#### **Model Storage**
```python
# Store models in global context
context.store("model_network", pytorch_model)
context.store("optimizer_sgd", torch_optimizer)

# Access from other nodes
model = context.retrieve("model_network") 
optimizer = context.retrieve("optimizer_sgd")
```

#### **Training State Management**
```python
# Track training progress
context.episode_count = current_epoch
context.step_count = batch_number  
context.total_reward = cumulative_loss

# Episode boundaries
if epoch_complete:
    context.episode_count += 1
    context.step_count = 0
```

#### **Parameter Coordination**
```python
# SGD Optimizer accesses all stored models
stored_models = context.get_all_models()
for model_name, model in stored_models.items():
    optimizer.add_param_group({'params': model.parameters()})
```

## Training Loop Architecture

### Complete MNIST Training Flow

```
MNISTDataset → BatchSampler → Network → CrossEntropyLoss → TrainingStep
    ↓              ↑             ↓           ↓               ↓
  dataset    [trigger]      predictions   loss          ready_signal  
                ↑                                           ↓
                └─────────────── trigger ──────────────────┘
```

### Execution Sequence

1. **Dataset Initialization**: MNISTDataset loads and prepares data
2. **Initial Batch**: BatchSampler generates first batch (no trigger needed)
3. **Forward Pass**: Network processes batch → produces predictions
4. **Loss Computation**: CrossEntropyLoss computes loss and accuracy
5. **Backpropagation**: TrainingStep performs gradient updates
6. **Trigger Generation**: TrainingStep sends "ready" signal
7. **Next Batch**: BatchSampler receives trigger → generates next batch
8. **Loop Continues**: Steps 3-7 repeat until training complete

### Synchronization Guarantees

✅ **Sequential Execution**: No batch processed until previous training step completes  
✅ **Consistent State**: Model parameters updated before next forward pass  
✅ **Proper Statistics**: Loss/accuracy computed on correct model state  
✅ **Epoch Boundaries**: Clean transitions between training epochs  

## Design Patterns

### 1. Startup Signal Pattern

**Problem**: How to start the training loop?
**Solution**: TrainingStep sends initial "ready" signal during startup

```python
class TrainingStep_node_X(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self._startup_complete = False
    
    async def startup(self):
        # Send initial ready signal to kickstart training
        if not self._startup_complete:
            await self.send_startup_signal()
            self._startup_complete = True
```

### 2. Dual Input Pattern

**Problem**: Nodes need both configuration (optimizer) and data (loss) inputs
**Solution**: Distinguish between configuration and flow inputs

```python
async def compute(self, loss, optimizer=None):
    # Configuration input (provided once)
    if optimizer is not None:
        self.stored_optimizer = optimizer
        
    # Flow input (provided each iteration)  
    return self.process_training_step(loss, self.stored_optimizer)
```

### 3. Context Bridge Pattern

**Problem**: How to share models between distant nodes?
**Solution**: Use global context as communication bridge

```python
# Network stores model in context
context.store(f"model_{self.node_id}", self.model)

# SGD Optimizer retrieves all models
all_models = context.get_stored_models()
for model_name, model in all_models.items():
    self.optimizer.add_param_group({'params': model.parameters()})
```

## Node Implementation Guidelines

### Creating New ML Nodes

#### 1. **Inherit from Base Classes**
```python
from custom_nodes.ml_nodes.base_nodes import MLNodeBase

class CustomTrainingNode(MLNodeBase):
    CATEGORY = "ml/training"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"input_data": ("TENSOR",)},
            "optional": {"config": ("CONFIG",)}
        }
```

#### 2. **Handle Context Integration**
```python
def compute(self, input_data, config=None):
    # Store in context for other nodes
    context = get_global_context()
    context.store("processed_data", result)
    
    # Retrieve shared state
    shared_model = context.retrieve("shared_model")
    return (processed_result,)
```

#### 3. **Support Trigger Patterns**
```python
async def compute(self, data_input, trigger=None):
    # Handle triggered execution
    if trigger is not None:
        # Process data only when triggered
        return self.process_triggered(data_input)
    else:
        # Initial or configuration setup
        return self.process_initial(data_input)
```

### Template Integration

#### Export Template Structure
```python
# templates/nodes/custom_node_queue.py
class {CLASS_NAME}_{NODE_ID}(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input1"], optional=["trigger"])
        self.setup_outputs(["output1", "ready"])
        
        # Template variables
        self.param1 = {PARAM1}
        self.param2 = "{PARAM2}"
    
    async def compute(self, input1, trigger=None):
        # Node-specific logic with template variables
        result = self.process(input1, self.param1, self.param2)
        
        # Generate ready signal if needed
        ready_signal = self.create_ready_signal()
        return {"output1": result, "ready": ready_signal}
```

## Performance Considerations

### Queue Size Optimization

**Small Queues** (size=1):
- ✅ Lower memory usage
- ✅ Tighter synchronization
- ❌ Potential bottlenecks

**Large Queues** (size=10+):
- ✅ Better throughput
- ✅ Handles timing variations
- ❌ Higher memory usage
- ❌ Looser synchronization

**Recommended**: Start with size=2 for training nodes, size=1 for trigger signals.

### Device Management

#### Automatic Device Placement
```python
def ensure_device_consistency(self, tensor_data):
    """Ensure all tensors are on the same device"""
    target_device = self.device or "cuda" if torch.cuda.is_available() else "cpu"
    
    if isinstance(tensor_data, torch.Tensor):
        return tensor_data.to(target_device)
    elif isinstance(tensor_data, dict):
        return {k: v.to(target_device) if isinstance(v, torch.Tensor) else v 
                for k, v in tensor_data.items()}
```

### Memory Management

#### Gradient Accumulation
```python
class TrainingStep_node_X(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.accumulate_steps = {ACCUMULATE_STEPS}
        self.current_step = 0
        
    async def compute(self, loss, optimizer):
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulate_steps
        scaled_loss.backward()
        
        self.current_step += 1
        
        # Update parameters when accumulation complete
        if self.current_step % self.accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        return self.create_ready_signal()
```

## Troubleshooting

### Common Issues

#### **Training Loop Hangs**
- **Cause**: Missing trigger connection from TrainingStep to BatchSampler
- **Fix**: Verify trigger connection in workflow
- **Debug**: Add logging to trigger receive/send events

#### **Memory Growth**  
- **Cause**: Gradients not cleared between steps
- **Fix**: Ensure `optimizer.zero_grad()` in TrainingStep
- **Debug**: Monitor GPU memory usage during training

#### **Inconsistent Loss**
- **Cause**: Multiple batches processed simultaneously
- **Fix**: Check trigger coordination is working
- **Debug**: Log batch IDs and processing order

#### **Device Mismatches**
- **Cause**: Tensors on different devices (CPU vs CUDA)
- **Fix**: Ensure consistent device placement in nodes
- **Debug**: Add device checking in compute methods

### Debugging Strategies

#### Enable Debug Logging
```python
# In generated code
import logging
logging.basicConfig(level=logging.DEBUG)

class NetworkNode_node_2(QueueNode):
    async def compute(self, input_data):
        self.logger.debug(f"Processing batch shape: {input_data.shape}")
        self.logger.debug(f"Model device: {next(self.model.parameters()).device}")
        # ... computation
```

#### Monitor Queue States
```python
# Add queue monitoring
async def monitor_queues(self):
    while True:
        for name, queue in self.input_queues.items():
            self.logger.info(f"Queue {name}: size={queue.qsize()}")
        await asyncio.sleep(1)
```

## Integration with Export System

### Template Variables

Common ML node template variables:
```python
template_vars = {
    "NODE_ID": node_id,
    "CLASS_NAME": "NetworkNode", 
    "LEARNING_RATE": 0.001,
    "BATCH_SIZE": 32,
    "HIDDEN_SIZE": 128,
    "DEVICE": "cuda",
    "DROPOUT_RATE": 0.1
}
```

### Generated Training Scripts

Exported ML workflows become standalone training scripts:
```python
# Generated runner.py structure
async def main():
    # Create nodes
    nodes = {
        "1": MNISTDataset_node_1("1"),
        "2": BatchSampler_node_2("2"),  
        "3": Network_node_3("3"),
        "4": TrainingStep_node_4("4")
    }
    
    # Setup trigger-based connections
    connections = [
        ("1", "dataset", "2", "dataset"),
        ("2", "images", "3", "input"),
        ("3", "output", "4", "predictions"),
        ("4", "ready", "2", "trigger")  # Trigger connection
    ]
    
    # Run training loop
    runner = GraphRunner(nodes, connections)
    await runner.run()
```

The ML nodes provide a robust foundation for supervised learning workflows that can be visually constructed and exported as production-ready training scripts.