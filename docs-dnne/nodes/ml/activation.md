# ML Activation Function Nodes

## Overview

Activation nodes introduce non-linearity into neural networks, enabling them to learn complex patterns. DNNE provides a unified activation node with multiple function options.

---

## ActivationNode

### Purpose
Applies non-linear activation functions to tensors, essential for deep learning.

### Category
`ML/Activation`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| input | Tensor | Input tensor to activate |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| function | string | "relu" | Activation function name |
| inplace | boolean | false | Modify input tensor directly |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| output | Tensor | Activated tensor |

### Supported Functions

#### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
```
- **Use case**: Default for hidden layers
- **Pros**: Simple, efficient, no vanishing gradient
- **Cons**: Can cause dead neurons

#### ELU (Exponential Linear Unit)
```
f(x) = x if x > 0 else α(e^x - 1)
```
- **Use case**: Alternative to ReLU
- **Pros**: Smooth, handles negative values
- **Cons**: Slightly more expensive

#### LeakyReLU
```
f(x) = x if x > 0 else α×x
```
- **Use case**: Prevents dead neurons
- **Pros**: Allows small negative gradients
- **Parameter**: α = 0.01 (default)

#### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
```
- **Use case**: Binary classification, gates
- **Pros**: Outputs in [0,1]
- **Cons**: Vanishing gradient

#### Tanh (Hyperbolic Tangent)
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- **Use case**: Hidden layers in RNNs
- **Pros**: Zero-centered, outputs in [-1,1]
- **Cons**: Vanishing gradient

#### Softmax
```
f(x_i) = e^(x_i) / Σ(e^(x_j))
```
- **Use case**: Multi-class output layer
- **Pros**: Outputs sum to 1 (probabilities)
- **Note**: Usually combined with CrossEntropyLoss

#### GELU (Gaussian Error Linear Unit)
```
f(x) = x × Φ(x)
```
- **Use case**: Transformer models
- **Pros**: Smooth, probabilistic
- **Cons**: More expensive computation

#### Swish/SiLU
```
f(x) = x × sigmoid(x)
```
- **Use case**: Modern architectures
- **Pros**: Smooth, self-gated
- **Cons**: More computation

### Usage Examples

#### Standard Hidden Layer
```
LinearLayer → Activation(relu) → LinearLayer
```

#### Output Layer Patterns
```
# Classification
LinearLayer → Activation(none) → CrossEntropyLoss

# Binary classification
LinearLayer → Activation(sigmoid) → BCELoss

# Regression
LinearLayer → Activation(none) → MSELoss
```

#### Deep Network
```
Input → Linear → Activation(elu) → Dropout →
        Linear → Activation(elu) → Dropout →
        Linear → Output
```

### Export Support
✅ Full support with all activation functions

### Notes
- Inplace operations save memory but prevent gradient computation on input
- Some activations (ReLU, ELU) support inplace mode
- Activation choice significantly impacts training dynamics

---

## Activation Selection Guide

### For Hidden Layers

| Activation | When to Use | Characteristics |
|------------|-------------|-----------------|
| ReLU | Default choice | Fast, simple, effective |
| ELU | Need smoothness | Better gradient flow |
| LeakyReLU | Dead neuron issues | Allows negative values |
| GELU | Transformers | State-of-the-art |

### For Output Layers

| Task | Activation | Loss Function |
|------|------------|---------------|
| Multi-class | None | CrossEntropyLoss |
| Binary class | Sigmoid | BCELoss |
| Multi-label | Sigmoid | BCEWithLogitsLoss |
| Regression | None | MSELoss |
| Probability | Softmax | NLLLoss |

### Network Depth Considerations

- **Shallow (1-3 layers)**: Any activation works
- **Medium (4-10 layers)**: Prefer ReLU/ELU
- **Deep (10+ layers)**: Use ReLU with BatchNorm
- **Very Deep (50+ layers)**: ResNet-style connections

## Common Patterns

### Standard MLP
```
Input → Linear(784→256) → ReLU →
        Linear(256→128) → ReLU →
        Linear(128→10) → Output
```

### CNN Architecture
```
Conv2D → BatchNorm → ReLU → MaxPool →
Conv2D → BatchNorm → ReLU → MaxPool →
Flatten → Linear → ReLU → Linear
```

### Modern Architecture
```
Linear → LayerNorm → GELU → Dropout →
Linear → LayerNorm → GELU → Dropout
```

## Performance Comparison

| Function | Speed | Memory | Gradient Flow |
|----------|-------|---------|---------------|
| ReLU | ★★★★★ | ★★★★★ | ★★★☆☆ |
| ELU | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| GELU | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| Sigmoid | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| Tanh | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |

## Best Practices

### Activation Placement
- After linear/conv layers
- Before dropout (usually)
- Not on final output (for raw logits)

### Avoiding Issues
- **Vanishing gradient**: Use ReLU family
- **Exploding gradient**: Use gradient clipping
- **Dead neurons**: Use LeakyReLU or ELU
- **Saturation**: Avoid sigmoid/tanh in deep networks

### Initialization
- ReLU: He initialization
- Tanh/Sigmoid: Xavier initialization
- Modern: Default PyTorch initialization

## Common Issues

### Issue: All neurons outputting zero
**Cause**: Dead ReLU problem
**Solution**: Use LeakyReLU or reduce learning rate

### Issue: Gradient vanishing
**Cause**: Sigmoid/Tanh saturation
**Solution**: Switch to ReLU family

### Issue: Training instability
**Cause**: Activation scale mismatch
**Solution**: Use BatchNorm or careful initialization

## Advanced Usage

### Custom Activation Parameters
```python
# LeakyReLU with custom slope
Activation("leakyrelu", negative_slope=0.2)

# ELU with custom alpha
Activation("elu", alpha=1.5)
```

### Combining Activations
```python
# Swish = x * sigmoid(x)
Linear → [Identity, Sigmoid] → Multiply
```

### Activation Regularization
```python
# Sparse activations
ReLU → L1Penalty → NextLayer
```

## Future Enhancements

- Parametric activations (PReLU)
- Adaptive activations
- Learned activation functions
- Activation visualization
- Hardware-optimized variants