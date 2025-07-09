# ML Training and Optimization Nodes

## Overview

Training nodes handle the optimization process, loss computation, and training orchestration. These nodes work together to train neural networks effectively.

---

## SGDOptimizerNode

### Purpose
Implements Stochastic Gradient Descent optimization with optional momentum.

### Category
`ML/Training`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| model | nn.Module | Neural network to optimize |
| loss | Tensor | Computed loss to minimize |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| learning_rate | float | 0.01 | Step size for updates |
| momentum | float | 0.0 | Momentum factor (0-1) |
| weight_decay | float | 0.0 | L2 regularization strength |
| nesterov | boolean | false | Use Nesterov momentum |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| optimizer | Optimizer | Configured optimizer object |

### Usage Example
```
Network.model → SGDOptimizer(lr=0.1, momentum=0.9) → TrainingStep
```

Common configurations:
- Basic SGD: `momentum=0, lr=0.01`
- SGD with momentum: `momentum=0.9, lr=0.01`
- Fine-tuning: Lower lr (0.001), high momentum
- With regularization: `weight_decay=0.0001`

### Export Support
✅ Full support with queue template

### Notes
- Updates parameters in-place
- Momentum accelerates convergence
- Weight decay prevents overfitting
- Learning rate scheduling supported

---

## CrossEntropyLossNode

### Purpose
Computes cross-entropy loss for classification tasks, combining LogSoftmax and NLLLoss.

### Category
`ML/Training`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| predictions | Tensor | Raw logits from network (before softmax) |
| targets | Tensor | Ground truth class labels |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| reduction | string | "mean" | How to reduce loss: "mean", "sum", "none" |
| label_smoothing | float | 0.0 | Label smoothing factor (0-1) |
| ignore_index | int | -100 | Index to ignore in loss computation |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| loss | Tensor | Computed loss value |
| accuracy | float | Batch accuracy (added feature) |

### Usage Example
```
Network → CrossEntropyLoss ← Targets
              ↓
          Optimizer
```

Use cases:
- Multi-class classification
- Outputs raw logits (no softmax needed)
- Supports class weights for imbalanced data

### Export Support
✅ Full support with enhanced metrics

### Notes
- Numerically stable computation
- Automatically applies softmax
- Returns both loss and accuracy
- Handles batch processing

---

## MSELossNode *(Coming Soon)*

### Purpose
Computes Mean Squared Error loss for regression tasks.

### Category
`ML/Training`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| predictions | Tensor | Network predictions |
| targets | Tensor | Ground truth values |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| reduction | string | "mean" | How to reduce: "mean", "sum", "none" |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| loss | Tensor | Computed MSE loss |

### Usage Example
```
Network → MSELoss ← Targets
            ↓
        Optimizer
```

Use cases:
- Regression problems
- Autoencoder reconstruction
- Continuous value prediction

### Export Support
✅ Full support with queue template

### Notes
- Sensitive to outliers
- Scale targets appropriately
- Consider MAE for robust alternative

---

## TrainingStepNode

### Purpose
Executes a complete training step: forward pass, loss computation, backward pass, and parameter update.

### Category
`ML/Training`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| optimizer | Optimizer | Configured optimizer |
| loss | Tensor | Computed loss value |
| trigger | Any | Optional trigger for timing |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| gradient_clip | float | 0.0 | Max gradient norm (0=disabled) |
| accumulation_steps | int | 1 | Gradient accumulation steps |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| step_complete | Trigger | Signal that step finished |
| loss_value | float | Loss value for logging |

### Usage Example
```
Loss → TrainingStep → EpochTracker
  ↑                        ↓
Optimizer              GetBatch
```

Training loop flow:
1. Compute loss
2. Backward pass
3. Clip gradients (optional)
4. Update parameters
5. Zero gradients

### Export Support
✅ Full support with async execution

### Notes
- Handles gradient computation
- Supports mixed precision training
- Includes gradient clipping
- Thread-safe for async execution

---

## EpochTrackerNode

### Purpose
Manages epoch progression and provides training statistics.

### Category
`ML/Training`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| step_complete | Trigger | Signal from TrainingStep |
| loss | float | Current loss value |
| accuracy | float | Current accuracy (optional) |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| total_epochs | int | 10 | Number of epochs to train |
| log_interval | int | 100 | Steps between logs |
| save_interval | int | 1 | Epochs between checkpoints |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| epoch_trigger | Trigger | Triggers new epoch |
| training_complete | boolean | True when done |
| epoch_stats | dict | Epoch statistics |

### Usage Example
```
TrainingStep → EpochTracker → GetBatch
                    ↓
              ModelCheckpoint
```

Features:
- Tracks training progress
- Computes epoch statistics
- Logs metrics periodically
- Triggers model saving

### Export Support
✅ Full support with comprehensive logging

### Notes
- Maintains running averages
- Supports early stopping
- Integrates with logging systems
- Provides training summaries

---

## AccuracyMetricNode *(Coming Soon)*

*Note: Accuracy computation is currently integrated into CrossEntropyLossNode*

### Purpose
Computes classification accuracy from predictions and labels.

### Category
`ML/Training`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| predictions | Tensor | Network outputs/logits |
| targets | Tensor | True labels |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| top_k | int | 1 | Compute top-k accuracy |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| accuracy | float | Accuracy percentage (0-100) |

### Usage Example
```
Network → AccuracyMetric ← Targets
              ↓
          Logging/Display
```

Metrics computed:
- Top-1 accuracy (default)
- Top-5 accuracy for ImageNet-style tasks
- Per-class accuracy available

### Export Support
✅ Full support with queue template

### Notes
- Handles multi-class classification
- Works with raw logits or probabilities
- Batch or epoch-level computation

---

## RunEpochNode *(Coming Soon)*

### Purpose
Orchestrates a complete epoch of training including all batches.

### Category
`ML/Training`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| sampler | BatchSampler | Data sampler |
| model | nn.Module | Model to train |
| optimizer | Optimizer | Optimizer to use |
| epoch_num | int | Current epoch number |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| training | boolean | true | Training or evaluation mode |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| epoch_complete | Trigger | Signals completion |
| epoch_loss | float | Average epoch loss |
| epoch_metrics | dict | All epoch metrics |

### Usage Example
```
[Sampler, Model, Optimizer] → RunEpoch → Logger
```

Manages:
- Batch iteration
- Loss accumulation
- Metric computation
- Mode switching (train/eval)

### Export Support
✅ Full support with queue template

---

## Training Patterns

### Basic Training Loop
```
GetBatch → Network → Loss → Optimizer → TrainingStep → EpochTracker
    ↑                                                        ↓
    └────────────────────────────────────────────────────────┘
```

### With Validation
```
TrainSampler → GetBatch → Network → Loss → TrainingStep
                              ↓
ValSampler → GetBatch → Network → Loss → Metrics
```

### Multi-Loss Training
```
Network → [Loss1, Loss2] → CombineLoss → Optimizer → TrainingStep
```

## Best Practices

### Learning Rate Selection
- Start with standard values (0.01 for SGD, 0.001 for Adam)
- Use learning rate finder
- Reduce on plateau
- Consider warmup for large batches

### Loss Function Choice
- CrossEntropy for classification
- MSE for regression
- Custom losses for specific tasks
- Weight losses for imbalanced data

### Training Stability
- Clip gradients for RNNs
- Use batch normalization
- Monitor gradient norms
- Start with small learning rates

### Optimization Tips
- Momentum helps escape local minima
- Weight decay prevents overfitting
- Gradient accumulation for large batches
- Mixed precision for speed

## Common Issues

### Issue: Loss not decreasing
**Solution**: Check learning rate, verify data flow, ensure gradients flow

### Issue: Loss exploding
**Solution**: Reduce learning rate, add gradient clipping, check for NaN

### Issue: Overfitting
**Solution**: Add regularization, reduce model size, increase data

### Issue: Slow training
**Solution**: Increase batch size, use better optimizer, check data loading

## Advanced Features

### Learning Rate Scheduling
```python
# Reduce LR on plateau
scheduler = ReduceLROnPlateau(optimizer, 'min')

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

### Gradient Accumulation
```python
# Effective batch size = batch_size × accumulation_steps
accumulation_steps = 4
```

### Mixed Precision Training
```python
# Automatic mixed precision for faster training
with autocast():
    output = model(input)
    loss = criterion(output, target)
```

## Future Enhancements

- Adam and AdamW optimizers
- Learning rate schedulers
- Advanced metrics (F1, AUC)
- Distributed training support
- Automatic hyperparameter tuning