# ML Data Processing Nodes

## Overview

Data processing nodes handle loading datasets, creating batches, and preparing data for training. These nodes form the foundation of any machine learning workflow in DNNE.

---

## MNISTDatasetNode

### Purpose
Loads the MNIST handwritten digit dataset for training and testing neural networks.

### Category
`ML/Data`

### Inputs
None (Sensor node)

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| train | boolean | true | Load training set (true) or test set (false) |
| download | boolean | true | Download dataset if not present |
| data_path | string | "./data" | Directory to store/load dataset |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| dataset | MNISTDataset | MNIST dataset object |

### Usage Example
```
MNISTDatasetNode → BatchSamplerNode → GetBatchNode → Network
```

Load MNIST training data:
- Set `train=true` for 60,000 training images
- Set `train=false` for 10,000 test images
- Data automatically downloads on first use

### Export Support
✅ Full support with queue template

### Notes
- Images are 28x28 grayscale
- Labels are integers 0-9
- Normalized to [0,1] range
- Cached after first download

---

## BatchSamplerNode

### Purpose
Creates batches from a dataset with configurable sampling strategy.

### Category
`ML/Data`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| dataset | Dataset | PyTorch dataset to sample from |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| batch_size | int | 32 | Number of samples per batch |
| shuffle | boolean | true | Randomize sample order |
| drop_last | boolean | false | Drop incomplete final batch |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| sampler | BatchSampler | Sampler object for retrieving batches |

### Usage Example
```
Dataset → BatchSamplerNode(batch_size=64) → GetBatchNode
```

Common configurations:
- Training: `shuffle=true, drop_last=true`
- Validation: `shuffle=false, drop_last=false`
- Large datasets: smaller batch_size to fit memory

### Export Support
✅ Full support with queue template

### Notes
- Sampler maintains internal state
- Iterates through dataset once per epoch
- Thread-safe for async operations

---

## GetBatchNode

### Purpose
Retrieves the next batch from a sampler and formats it for training.

### Category
`ML/Data`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| sampler | BatchSampler | Batch sampler to draw from |
| trigger | Any | Optional trigger to control timing |

### Parameters
None

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| inputs | Tensor | Batch of input data (e.g., images) |
| targets | Tensor | Batch of labels/targets |
| batch_idx | int | Current batch number |

### Usage Example
```
BatchSampler → GetBatchNode → [Network, Loss]
                    ↑
              EpochTracker (trigger)
```

Typical connections:
- Connect inputs to network input
- Connect targets to loss function
- Use batch_idx for logging

### Export Support
✅ Full support with enhanced features:
- Automatic epoch tracking
- Batch statistics logging
- Synchronized with training loop

### Notes
- Automatically handles epoch boundaries
- Reshapes data for neural network input
- Moves tensors to appropriate device

---

## TrainTestSplitNode *(Coming Soon)*

### Purpose
Splits a dataset into training and validation sets.

### Category
`ML/Data`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| dataset | Dataset | Full dataset to split |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| train_ratio | float | 0.8 | Fraction for training (0-1) |
| random_seed | int | 42 | Seed for reproducible splits |
| stratify | boolean | false | Maintain class distribution |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| train_dataset | Dataset | Training subset |
| val_dataset | Dataset | Validation subset |

### Usage Example
```
FullDataset → TrainTestSplitNode → [TrainSampler, ValSampler]
```

Best practices:
- Use 80/20 or 70/30 splits typically
- Enable stratify for imbalanced datasets
- Keep consistent seed for reproducibility

### Export Support
✅ Full support with queue template

### Notes
- Preserves dataset properties
- No data duplication (uses indices)
- Supports custom datasets

---

## Data Pipeline Patterns

### Basic Training Pipeline
```
MNISTDataset(train=true) → BatchSampler(batch_size=32) → GetBatch → Network
```

### Train/Validation Pipeline
```
MNISTDataset → TrainTestSplit ─→ TrainSampler → GetBatch → Network
                              └→ ValSampler → GetBatch → Evaluate
```

### Multi-Epoch Training
```
BatchSampler → GetBatch → Network → Loss
       ↑           ↓
  EpochTracker ← TrainingStep
```

## Best Practices

### Batch Size Selection
- Start with 32 or 64
- Larger batches: faster training, more memory
- Smaller batches: better generalization, less memory
- Power of 2 for GPU efficiency

### Data Loading Performance
- Enable shuffle for training
- Disable shuffle for validation
- Use drop_last to maintain consistent batch sizes
- Consider data augmentation nodes (future)

### Memory Management
- Monitor GPU memory with large batches
- Reduce batch size if OOM errors
- Use gradient accumulation for effective larger batches

## Common Issues

### Issue: Dataset not found
**Solution**: Ensure data_path exists and is writable

### Issue: Batch size too large
**Solution**: Reduce batch_size parameter or use smaller dataset

### Issue: Inconsistent batch dimensions
**Solution**: Enable drop_last to avoid partial batches

## Future Enhancements

- Custom dataset loaders (CSV, images)
- Data augmentation nodes
- Distributed data loading
- Streaming datasets for large data
- Cache management utilities