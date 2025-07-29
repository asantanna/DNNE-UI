# Logging Guidelines for DNNE Exported Code

This document defines the logging standards for DNNE exported code (generated Python scripts). These guidelines ensure consistent, useful output across all exported workflows.

## Overview

DNNE uses a two-tier output system:
- **`print()`** - For important user-facing messages
- **`self.logger`** - For diagnostic and debug information

## Command-Line Switches

The exported runner.py supports flexible logging control:

```bash
# Quiet mode (default) - Only warnings, errors, and user messages
python runner.py

# Verbose mode - Adds INFO level logging for all subsystems
python runner.py --verbose

# Debug mode - Shows all logging including DEBUG level for all subsystems
python runner.py --debug

# Subsystem-specific verbose mode
python runner.py --verbose yield,ppo

# Subsystem-specific debug mode
python runner.py --debug yield

# Combine: verbose for all, debug for specific subsystems
python runner.py --verbose --debug yield,ppo
```

When specifying subsystems, use these standard names:
- `yield` - Adaptive yielding system
- `ppo` - PPO/RL training
- `queue` - Queue framework
- `checkpoint` - Checkpoint operations
- `mnist` - MNIST data handling
- `node` - All node operations
- `balancing` - Execution balance reports (periodic balance between subgraphs)

## Logger Initialization

DNNE provides a custom logging wrapper that automatically prepends "dnne." to logger names for proper namespacing:

```python
# Import standard logging for general use
import logging

# Import DNNE logging wrapper for subsystem loggers
from framework.globals import dnne_logging

# General logger (for modules without specific subsystem)
logger = logging.getLogger(__name__)

# Subsystem loggers (automatic "dnne." prefix)
yield_logger = dnne_logging.getLogger("yield")      # Creates "dnne.yield"
ppo_logger = dnne_logging.getLogger("ppo")          # Creates "dnne.ppo"
queue_logger = dnne_logging.getLogger("queue")      # Creates "dnne.queue"
checkpoint_logger = dnne_logging.getLogger("checkpoint")  # Creates "dnne.checkpoint"

# Node loggers (in node classes)
class MyNode(QueueNode):
    def __init__(self, node_id):
        super().__init__(node_id)
        # Use node_logger to distinguish from general logger
        self.node_logger = dnne_logging.getLogger(f"node.{node_id}")
```

### Variable Naming Standards

- **`logger`** - General module logging (uses standard Python logging)
- **`{subsystem}_logger`** - Subsystem-specific logging (e.g., `yield_logger`, `ppo_logger`)
- **`self.node_logger`** - Node instance logging (makes it clear this is node-specific)

## Logging Levels

### User-Facing Output (print)

Always use `print()` for messages that users should always see:

```python
# Training milestones
print(f"🚀 Training starting... ({total_epochs} epochs)")
print(f"📊 Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")
print(f"🎯 Training complete! Final accuracy: {accuracy:.2%}")

# Important state changes
print(f"💾 Checkpoint saved to {checkpoint_path}")
print(f"⚠️  No checkpoint found, starting fresh")
```

### DEBUG Level (self.logger.debug)

Detailed diagnostic information for troubleshooting:

```python
# Tensor shapes and data flow
self.logger.debug(f"Input tensor shape: {x.shape}, dtype: {x.dtype}")
self.logger.debug(f"Queue depth: {self.input_queue.qsize()}")

# Computation details
self.logger.debug(f"Batch {batch_id} processing time: {elapsed:.3f}s")
self.logger.debug(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### INFO Level (self.logger.info)

General progress and state information:

```python
# Node initialization
self.logger.info(f"Node {self.node_id} initialized with {num_layers} layers")
self.logger.info(f"Using device: {device}")

# Major operations
self.logger.info(f"Starting epoch {epoch}")
self.logger.info(f"Loaded checkpoint from iteration {iteration}")

# Configuration details
self.logger.info(f"Learning rate: {lr}, batch size: {batch_size}")
```

### WARNING Level (self.logger.warning)

Potential issues that don't prevent execution:

```python
# Performance concerns
self.logger.warning(f"GPU utilization low: {gpu_util}%")
self.logger.warning(f"Training loss increasing for {epochs} epochs")

# Configuration issues
self.logger.warning(f"Batch size {batch_size} may be too large for GPU memory")
self.logger.warning(f"No seed specified - results may not be reproducible")
```

### ERROR Level (self.logger.error)

Serious problems that affect functionality:

```python
# Failed operations
self.logger.error(f"Failed to save checkpoint: {e}")
self.logger.error(f"Cannot load model weights from {path}")

# Missing requirements
self.logger.error(f"Required input '{input_name}' not connected")
self.logger.error(f"CUDA out of memory: {e}")
```

### CRITICAL Level (self.logger.critical)

System failures requiring immediate attention:

```python
# Fatal errors
self.logger.critical("Isaac Gym library not found - cannot continue")
self.logger.critical(f"Unrecoverable CUDA error: {e}")
```

## Best Practices

### 1. Respect Verbosity Settings

```python
# Always visible (important milestones)
print(f"📊 Epoch {epoch} complete - Loss: {loss:.4f}")

# Only in verbose mode
if g.verbose:
    self.logger.info(f"Batch {batch}/{total_batches} processed")

# Only in debug mode  
if g.debug:
    self.logger.debug(f"Gradient norm: {grad_norm:.6f}")
```

### 2. Use Appropriate Emojis for User Messages

```python
# Standard emoji conventions
print("🚀 Starting...")        # Launch/start
print("📊 Statistics...")      # Data/metrics  
print("🎯 Complete!")         # Success/completion
print("💾 Saving...")         # File operations
print("⚠️  Warning...")       # Warnings
print("❌ Error...")          # Errors
print("🔄 Loading...")        # Loading/resuming
```

### 3. Include Context in Log Messages

```python
# Good - includes context
self.logger.info(f"Node {self.node_id}: Processed {count} items in {elapsed:.2f}s")

# Bad - missing context  
self.logger.info(f"Processed items")
```

### 4. Format Numbers Appropriately

```python
# Loss values - 4 decimal places
print(f"Loss: {loss:.4f}")

# Percentages - 1-2 decimal places
print(f"Accuracy: {accuracy:.2%}")  # 0.9234 -> "92.34%"

# Time - appropriate units
if elapsed < 1:
    print(f"Time: {elapsed*1000:.0f}ms")
else:
    print(f"Time: {elapsed:.1f}s")
```

### 5. Batch Progress Updates

```python
# Show progress periodically, not every batch
if batch_num % 10 == 0 and g.verbose:
    self.logger.info(f"Progress: {batch_num}/{total_batches}")

# Summary after completion
print(f"📊 Processed {total_batches} batches in {elapsed:.1f}s")
```

## Node-Specific Guidelines

### Training Nodes
- Always print epoch summaries
- Log batch progress only in verbose mode
- Show final results prominently

### Data Loading Nodes  
- Print dataset info on first load
- Log batch preparation in debug mode
- Warn about data issues

### Model Nodes
- Print architecture summary once
- Log forward pass details in debug mode
- Show checkpoint operations

### Environment Nodes
- Print environment initialization
- Log step details in debug mode
- Show episode summaries

## Example Implementation

```python
from framework.globals import dnne_logging

# Module-level logger for training subsystem
training_logger = dnne_logging.getLogger("training")

class MyTrainingNode(QueueNode):
    def __init__(self, node_id):
        super().__init__(node_id)
        # Node-specific logger
        self.node_logger = dnne_logging.getLogger(f"node.{node_id}")
        self.node_logger.info(f"Initializing {self.__class__.__name__}")
        
    async def compute(self, inputs):
        # User-facing message
        if self.first_batch:
            print(f"🚀 Starting training with batch size {batch_size}")
            
        # Debug information using node logger
        self.node_logger.debug(f"Input shape: {inputs.shape}")
        
        # Subsystem-specific logging
        training_logger.debug(f"Computing gradients for batch {self.batch_num}")
        
        # Process batch...
        
        # Conditional progress
        if g.verbose and self.batch_num % 10 == 0:
            self.node_logger.info(f"Batch {self.batch_num}: loss={loss:.4f}")
            
        # Always show epoch complete
        if epoch_complete:
            print(f"📊 Epoch {epoch} complete - Loss: {avg_loss:.4f}")
```

## Subsystem Usage Examples

### Adaptive Yielding System
```python
from framework.globals import dnne_logging
yield_logger = dnne_logging.getLogger("yield")

def sync_adaptive_yield():
    yield_logger.debug("Starting adaptive yield")
    # ... yield logic ...
    yield_logger.debug(f"Yield completed in {duration}ms")
```

### PPO Training
```python
from framework.globals import dnne_logging
ppo_logger = dnne_logging.getLogger("ppo")

class PPOTrainer:
    def train_step(self):
        ppo_logger.info("Starting PPO training iteration")
        ppo_logger.debug(f"Batch size: {batch_size}, LR: {lr}")
```

### Queue Framework
```python
from framework.globals import dnne_logging
queue_logger = dnne_logging.getLogger("queue")

class QueueManager:
    def process(self):
        queue_logger.debug(f"Queue depth: {self.queue.qsize()}")
```

### Execution Balance Reporting
```python
from framework.globals import dnne_logging
balancing_logger = dnne_logging.getLogger("balancing")

# Periodic balance reports (controlled by --verbose balancing)
def print_balance_report():
    stats = calculate_balance_stats()
    balancing_logger.info(f"Execution balance: PPO {stats['ppo_pct']:.1f}% / MNIST {stats['mnist_pct']:.1f}%")
```

## Testing Your Logging

Test all logging modes to ensure appropriate output:

```bash
# Quiet mode - should see only milestones
python runner.py --timeout 30s

# Verbose - should add progress updates  
python runner.py --timeout 30s --verbose

# Debug - should show detailed diagnostics
python runner.py --timeout 30s --debug

# Debug specific subsystems only
python runner.py --timeout 30s --debug yield,ppo

# Verbose globally, debug for yield system
python runner.py --timeout 30s --verbose --debug yield
```