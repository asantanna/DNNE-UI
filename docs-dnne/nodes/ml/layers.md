# ML Neural Network Layer Nodes

## Overview

Layer nodes are the building blocks of neural networks in DNNE. They provide various types of layers from simple linear transformations to advanced architectures.

---

## LinearLayerNode

### Purpose
Creates a fully connected (dense) layer that applies a linear transformation to input data.

### Category
`ML/Layers`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| input | Tensor | Input tensor to transform |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input_size | int | Required | Number of input features |
| output_size | int | Required | Number of output features |
| bias | boolean | true | Include bias parameters |
| activation | string | "none" | Activation function to apply |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| output | Tensor | Transformed tensor |
| layer | nn.Module | The layer module (for NetworkNode) |

### Usage Example
```
Input(784) → LinearLayer(784→128) → ReLU → LinearLayer(128→10) → Output
```

Common configurations:
- Hidden layers: Use ReLU or ELU activation
- Output layer: Usually no activation (raw logits)
- Feature extraction: Large input_size → smaller output_size

### Export Support
✅ Full support with queue template

### Notes
- Weights initialized using Xavier/He initialization
- Total parameters: input_size × output_size + output_size (if bias)
- Supports batch processing automatically

---

## Conv2DLayerNode

### Purpose
Applies 2D convolution over input images, essential for computer vision tasks.

### Category
`ML/Layers` (registered as "Conv2DLayer")

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| input | Tensor | Input tensor (N, C, H, W) |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| in_channels | int | Required | Number of input channels |
| out_channels | int | Required | Number of output filters |
| kernel_size | int/tuple | 3 | Size of convolution kernel |
| stride | int/tuple | 1 | Stride of convolution |
| padding | int/tuple | 0 | Padding added to input |
| bias | boolean | true | Include bias parameters |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| output | Tensor | Convolved features |
| layer | nn.Module | The layer module |

### Usage Example
```
Image → Conv2D(3→32, k=3) → ReLU → MaxPool → Conv2D(32→64, k=3)
```

Typical CNN architecture:
- Early layers: Fewer channels, capture edges
- Deeper layers: More channels, capture complex features
- Use padding to maintain spatial dimensions

### Export Support
✅ Full support with queue template

### Notes
- Input format: (batch, channels, height, width)
- Output size calculation: (input + 2×padding - kernel) / stride + 1
- Supports various padding modes: 'valid', 'same'

---

## NetworkNode

### Purpose
Acts as an aggregator that creates a chain of LinearLayer nodes and combines them into a single PyTorch Sequential module. LinearLayers are always used within this Network node pattern.

### Category
`ML/Layers`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| input | Tensor | Input data to the network |
| to_output | Tensor | Output from the last LinearLayer in the chain |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | string | "network" | Name for the network module |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| layers | Tensor | Pass-through to first LinearLayer |
| output | Tensor | Final network output |
| model | nn.Sequential | Combined network module for optimizer |

### Usage Example
```
                ┌──────────────────────────────────┐
                ↓                                  │
GetBatch → NetworkNode                             │
              ↓ layers                             │
         LinearLayer1 → LinearLayer2 → LinearLayer3
                                           ↓ to_output
```

Connection Pattern:
1. Network's `layers` output → First LinearLayer's input
2. LinearLayers chain together sequentially
3. Last LinearLayer's output → Network's `to_output` input
4. Network provides final `output` and `model`

Benefits:
- Visual organization of layer chains
- Automatic collection into PyTorch Sequential
- Single model reference for optimizers
- Clean code generation
- Efficient backpropagation

### Export Support
✅ Full support with automatic layer collection
- Identifies all LinearLayers in the chain
- Creates optimized nn.Sequential module
- Preserves layer order and parameters

### Notes
- LinearLayers must form a connected chain
- The Network node acts as entry/exit points
- Supports any number of LinearLayers
- Compatible with all PyTorch optimizers
- Enables visual grouping in the UI

---

## DropoutNode

### Purpose
Applies dropout regularization during training to prevent overfitting.

### Category
`ML/Layers`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| input | Tensor | Input tensor |
| training | boolean | Training mode flag |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dropout_rate | float | 0.5 | Probability of dropping units (0-1) |
| inplace | boolean | false | Modify input in-place |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| output | Tensor | Tensor with dropout applied |

### Usage Example
```
LinearLayer → Dropout(0.2) → LinearLayer → Dropout(0.5) → Output
```

Best practices:
- Use lower rates (0.1-0.3) for small networks
- Use higher rates (0.5-0.7) for large networks
- Place after activation functions
- Disable during evaluation

### Export Support
✅ Full support with automatic training mode handling

### Notes
- Automatically disabled during inference
- Scales remaining values to maintain expected sum
- Different random drops each forward pass

---

## BatchNormNode

### Purpose
Normalizes inputs across the batch dimension to stabilize training.

### Category
`ML/Layers`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| input | Tensor | Input tensor to normalize |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_features | int | Required | Number of features to normalize |
| eps | float | 1e-5 | Small constant for numerical stability |
| momentum | float | 0.1 | Momentum for running statistics |
| affine | boolean | true | Learn scale and shift parameters |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| output | Tensor | Normalized tensor |
| layer | nn.Module | BatchNorm module |

### Usage Example
```
Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU
```

Common patterns:
- Place before or after activation
- Use with deep networks
- Helps with gradient flow

### Export Support
✅ Full support with queue template

### Notes
- Maintains running statistics during training
- Uses different behavior for train/eval modes
- Can enable faster training with higher learning rates

---

## Layer Combination Patterns

### Simple Feedforward Network
```
Input → Linear(784→128) → ReLU → Dropout(0.2) → Linear(128→10)
```

### Deep Network with BatchNorm
```
Input → Linear → BatchNorm → ReLU → Dropout →
        Linear → BatchNorm → ReLU → Dropout →
        Linear → Output
```

### CNN Architecture
```
Image → Conv2D → BatchNorm → ReLU → MaxPool →
        Conv2D → BatchNorm → ReLU → MaxPool →
        Flatten → Linear → Dropout → Linear
```

### Network Node Pattern
```
Input → NetworkNode ←────────────────────────┐
            ↓ layers                         │
       Linear1(784→256) → Linear2(256→128) → Linear3(128→10)
                                                ↓ to_output
            ↓ model
        Optimizer
```

## Best Practices

### Layer Sizing
- Gradually decrease layer sizes in encoder
- Use power of 2 sizes for GPU efficiency
- Match input/output sizes to data dimensions

### Activation Placement
- ReLU/ELU after linear layers
- No activation on final output layer
- Experiment with activation types

### Regularization Strategy
- Dropout: Start with 0.2-0.5
- BatchNorm: Use in deep networks
- Combine both for best results

### Initialization
- Default initialization usually works
- He initialization for ReLU networks
- Xavier initialization for tanh/sigmoid

## Common Issues

### Issue: Gradient explosion
**Solution**: Add BatchNorm or reduce learning rate

### Issue: Overfitting
**Solution**: Add Dropout layers or reduce model size

### Issue: Dimension mismatch
**Solution**: Check layer input/output sizes match

### Issue: Training instability
**Solution**: Add BatchNorm and check initialization

## Performance Tips

- Larger batches work better with BatchNorm
- Dropout adds minimal computational cost
- Linear layers are memory-bound on GPU
- Conv2D layers are compute-bound

## Future Enhancements

- Attention layers
- Recurrent layers (LSTM, GRU)
- Transformer blocks
- Custom layer creation
- Layer pruning and quantization