# ML Nodes CLAUDE.md

Machine learning nodes for supervised learning in DNNE.

## Overview

ML nodes implement trigger-based training coordination for visual neural network construction.

## Node Categories

- **Data Nodes**: MNISTDataset, BatchSampler, GetBatch
- **Layer Nodes**: LinearLayer, Conv2D, Dropout, BatchNorm
- **Network Nodes**: Network (sequential model container)
- **Training Nodes**: SGDOptimizer, CrossEntropyLoss, TrainingStep
- **Utility Nodes**: AccuracyMetrics, EpochTracker
- **Activation Nodes**: ReLU, Tanh, Sigmoid, Softmax

## Key Patterns

- **Trigger-Based Training**: TrainingStep → trigger → GetBatch
- **Network Composition**: Stack LinearLayer nodes → Network node
- **Device Management**: All nodes support CPU/CUDA placement

## Documentation

For detailed information, see:
- **Node Reference**: `docs-dnne/nodes/ml/`
- **Training Workflows**: `docs-dnne/ML/training_workflow.md`
- **Examples**: `docs-dnne/examples/mnist_classification.md`