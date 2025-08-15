# LinearLayer Node

## Overview
The LinearLayer node implements a fully connected (dense) neural network layer, performing a linear transformation: `output = input @ weight.T + bias`.

## Properties

- **Category**: `ml`
- **Color Scheme**: Layer nodes
- **Implementation**: `custom_nodes/linear_layer_visnode.py`

## Inputs

### Required Parameters
- **in_features** (INT)
  - Default: `784`
  - Number of input features/neurons
  - For MNIST flattened images: 784 (28*28)

- **out_features** (INT)
  - Default: `128`
  - Number of output features/neurons
  - Determines the layer's output dimension

- **bias** (BOOLEAN)
  - Default: `True`
  - Whether to include a learnable bias term
  - Set to False for certain architectures or when using BatchNorm

### Optional Inputs
- **input** (TENSOR)
  - Input tensor from previous layer or data source
  - Shape: [batch_size, in_features]

## Outputs

- **layer** (LAYER)
  - PyTorch Linear layer object
  - Can be connected to Network node or used standalone

- **output** (TENSOR)
  - Output tensor after linear transformation
  - Shape: [batch_size, out_features]

## Functionality

1. **Weight Initialization**: Uses PyTorch's default initialization (Kaiming uniform)
2. **Forward Pass**: Computes `y = xW^T + b`
3. **Gradient Support**: Fully differentiable for backpropagation
4. **Device Handling**: Automatically moves to appropriate device (CPU/CUDA)

## Usage Example

### In Visual Workflow
1. Add LinearLayer node
2. Set in_features to match previous layer's output
3. Set out_features for desired dimension
4. Connect to activation functions or other layers

### Common Architectures
```
# Simple MLP for MNIST
Input (784) → LinearLayer(784, 128) → ReLU → LinearLayer(128, 10)

# Hidden layers
LinearLayer(128, 256) → LinearLayer(256, 128) → LinearLayer(128, 10)
```

### Exported Python Code
```python
class LinearLayerNode(QueueNode):
    def __init__(self, in_features=784, out_features=128, bias=True):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias=bias)
    
    async def process(self, input_tensor):
        output = self.layer(input_tensor)
        await self.output_queue.put({
            "layer": self.layer,
            "output": output
        })
```

## Best Practices

1. **Dimension Matching**: Ensure in_features matches the output size of the previous layer
2. **Activation Functions**: Usually followed by non-linear activation (ReLU, Sigmoid, etc.)
3. **Initialization**: Consider custom weight initialization for deep networks
4. **Regularization**: Combine with Dropout or BatchNorm for better generalization

## Common Issues

- **Dimension Mismatch**: Error when in_features doesn't match input tensor size
- **Memory Usage**: Large layers (e.g., 4096x4096) consume significant memory
- **Gradient Vanishing**: Very deep networks may need careful initialization

## Related Nodes

- [Network](network.md) - Combine multiple layers
- [Activation](activation.md) - Add non-linearity
- [BatchNorm](batchnorm.md) - Normalize layer outputs
- [Dropout](dropout.md) - Add regularization