# Convolutional Neural Network Support

## Priority
Medium

## Description
Add convolutional layer nodes for image processing tasks, enabling CNN architectures in DNNE.

## Motivation
- CNNs are essential for modern computer vision
- Current Conv2DLayer node needs companions (pooling, etc.)
- Users need to build vision models
- MNIST can achieve 99%+ accuracy with CNNs

## Implementation Notes
### Required Nodes
- Conv2D (already exists as Conv2DLayerNode)
- MaxPool2D node
- AvgPool2D node  
- BatchNorm2D node
- Flatten node for conv-to-linear transition

### Technical Requirements
- Proper shape propagation through conv layers
- Handle padding calculations
- Support different data formats (NCHW vs NHWC)
- Efficient memory usage for feature maps

## Example Use Case
CNN-based MNIST classifier:
```
Input → Conv2D → ReLU → MaxPool2D → 
        Conv2D → ReLU → MaxPool2D →
        Flatten → Linear → Output
```

## Dependencies
- Current Conv2DLayerNode implementation
- Shape inference system

## Estimated Effort
Medium

## Success Metrics
- Build complete CNN for MNIST
- Achieve 99%+ accuracy
- Support common CNN architectures (LeNet, VGG-style)