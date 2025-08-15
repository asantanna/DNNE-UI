# TrainingStep Node

## Overview
The TrainingStep node executes a complete training iteration including forward pass, loss calculation, backpropagation, and parameter updates.

## Properties

- **Category**: `ml`
- **Color Scheme**: Training nodes
- **Implementation**: `custom_nodes/training_step_visnode.py`

## Inputs

### Required Inputs
- **network** (NETWORK)
  - Neural network model to train
  - From Network or individual layer nodes

- **inputs** (TENSOR)
  - Input batch data
  - Shape: [batch_size, ...input_dimensions]

- **targets** (TENSOR)
  - Ground truth labels/targets
  - Shape: [batch_size] for classification

- **loss_fn** (LOSS_FN)
  - Loss function (e.g., CrossEntropyLoss)
  - Defines the optimization objective

- **optimizer** (OPTIMIZER)
  - Optimizer (e.g., SGDOptimizer)
  - Updates model parameters

### Optional Inputs
- **trigger** (TRIGGER)
  - Signal to execute training step
  - Used for synchronization in workflows

## Outputs

- **loss** (FLOAT)
  - Scalar loss value for the batch
  - Used for monitoring training progress

- **predictions** (TENSOR)
  - Model predictions/logits
  - Shape: [batch_size, num_classes]

- **completion** (TRIGGER)
  - Signal that training step completed
  - Triggers dependent nodes

## Functionality

1. **Forward Pass**: Computes network output from inputs
2. **Loss Calculation**: Computes loss between predictions and targets
3. **Backpropagation**: Calculates gradients via `loss.backward()`
4. **Parameter Update**: Optimizer updates weights via `optimizer.step()`
5. **Gradient Reset**: Clears gradients via `optimizer.zero_grad()`

## Training Loop Sequence
```python
# For each batch:
1. optimizer.zero_grad()       # Clear previous gradients
2. predictions = network(inputs)  # Forward pass
3. loss = loss_fn(predictions, targets)  # Compute loss
4. loss.backward()             # Compute gradients
5. optimizer.step()            # Update parameters
```

## Usage Example

### In Visual Workflow
1. Connect Network output to network input
2. Connect GetBatch outputs to inputs and targets
3. Connect CrossEntropyLoss to loss_fn
4. Connect SGDOptimizer to optimizer
5. Connect loss output to EpochTracker for monitoring

### Complete Training Pipeline
```
GetBatch → inputs, targets
    ↓         ↓
Network    targets
    ↓         ↓
TrainingStep (with loss_fn, optimizer)
    ↓
loss → EpochTracker
predictions → Accuracy
```

### Exported Python Code
```python
class TrainingStepNode(QueueNode):
    async def process(self, network, inputs, targets, loss_fn, optimizer):
        # Training mode
        network.train()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = network(inputs)
        
        # Compute loss
        loss = loss_fn(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        await self.output_queue.put({
            "loss": loss.item(),
            "predictions": predictions,
            "completion": True
        })
```

## Best Practices

1. **Batch Size**: Use appropriate batch sizes (32, 64, 128) for stable training
2. **Learning Rate**: Start with standard rates (0.01 for SGD, 0.001 for Adam)
3. **Gradient Clipping**: Consider clipping for RNNs or unstable training
4. **Mixed Precision**: Use AMP for faster training on modern GPUs

## Training Modes

- **Training Mode**: Enables dropout, batch norm training behavior
- **Evaluation Mode**: Disable via separate evaluation workflow
- **Gradient Accumulation**: Multiple forward passes before optimizer step

## Common Issues

- **NaN Loss**: Learning rate too high or numerical instability
- **Exploding Gradients**: Need gradient clipping or lower learning rate
- **Memory Errors**: Batch size too large for GPU memory
- **Slow Convergence**: Learning rate too low or poor initialization

## Performance Optimization

1. **GPU Utilization**: Ensure tensors are on CUDA device
2. **Data Loading**: Use parallel data loaders
3. **Mixed Precision**: Enable automatic mixed precision (AMP)
4. **Gradient Checkpointing**: For very deep networks

## Related Nodes

- [SGDOptimizer](sgd_optimizer.md) - Parameter optimization
- [CrossEntropyLoss](cross_entropy_loss.md) - Loss calculation
- [Network](network.md) - Model definition
- [EpochTracker](epoch_tracker.md) - Training monitoring
- [Accuracy](accuracy.md) - Performance metrics