# ML Nodes Documentation

Machine learning nodes for building neural networks, handling datasets, and training models in DNNE.

## Dataset Nodes

### [MNISTDataset](mnist_dataset.md)
Loads the MNIST handwritten digit dataset (28x28 grayscale images, 10 classes).

### [CIFAR10Dataset](cifar10_dataset.md)
Loads the CIFAR-10 image classification dataset (32x32 color images, 10 classes).

## Layer Nodes

### [LinearLayer](linear_layer.md)
Fully connected (dense) neural network layer with configurable input/output dimensions.

### [Conv2DLayer](conv2d_layer.md)
2D convolutional layer for image processing with configurable kernel size, stride, and padding.

### [BatchNorm](batchnorm.md)
Batch normalization layer for stabilizing training and improving convergence.

### [Dropout](dropout.md)
Dropout regularization layer to prevent overfitting during training.

### [Flatten](flatten.md)
Flattens multi-dimensional tensors into 1D vectors for fully connected layers.

### [Activation](activation.md)
Applies non-linear activation functions (ReLU, Sigmoid, Tanh, etc.) to tensors.

### [Network](network.md)
Composite node that combines multiple layers into a sequential neural network model.

## Training Nodes

### [SGDOptimizer](sgd_optimizer.md)
Stochastic Gradient Descent optimizer with momentum support.

### [CrossEntropyLoss](cross_entropy_loss.md)
Calculates cross-entropy loss for classification tasks.

### [TrainingStep](training_step.md)
Executes a single training iteration including forward pass, loss calculation, and backpropagation.

### [Accuracy](accuracy.md)
Calculates classification accuracy metrics during training and evaluation.

## Data Handling Nodes

### [BatchSampler](batch_sampler.md)
Creates a batch sampler for iterating through datasets in mini-batches.

### [GetBatch](get_batch.md)
Retrieves the next batch of data from a sampler with automatic triggering.

## Monitoring Nodes

### [EpochTracker](epoch_tracker.md)
Tracks training epochs, loss values, and other metrics over time.

### [TensorVisualizer](tensor_visualizer.md)
Visualizes tensor data as images or plots for debugging and analysis.

## Node Categories by Function

### Data Loading
- MNISTDataset - MNIST digit dataset
- CIFAR10Dataset - CIFAR-10 image dataset

### Neural Network Layers
- LinearLayer - Fully connected layer
- Conv2DLayer - 2D convolution
- BatchNorm - Batch normalization
- Dropout - Dropout regularization
- Flatten - Tensor flattening
- Activation - Non-linear activations
- Network - Sequential model composition

### Optimization & Training
- SGDOptimizer - SGD with momentum
- CrossEntropyLoss - Classification loss
- TrainingStep - Training iteration
- Accuracy - Accuracy metrics

### Data Processing
- BatchSampler - Mini-batch creation
- GetBatch - Batch retrieval

### Monitoring & Visualization
- EpochTracker - Training progress tracking
- TensorVisualizer - Data visualization

## Common Workflows

### Basic MNIST Classification
1. MNISTDataset → BatchSampler → GetBatch
2. Network (with LinearLayer nodes) 
3. CrossEntropyLoss + SGDOptimizer → TrainingStep
4. Accuracy for evaluation
5. EpochTracker for monitoring

### CNN for CIFAR-10
1. CIFAR10Dataset → BatchSampler → GetBatch
2. Conv2DLayer → BatchNorm → Activation → Conv2DLayer (repeated)
3. Flatten → LinearLayer → Activation → LinearLayer
4. CrossEntropyLoss + SGDOptimizer → TrainingStep
5. Accuracy + EpochTracker for monitoring

## Export Behavior

All ML nodes export to PyTorch-based Python code with:
- Async queue-based execution for real-time applications
- Proper device handling (CPU/CUDA)
- Automatic gradient computation
- State management for training/evaluation modes

## Implementation Details

- **Base Class**: All ML nodes inherit from `RoboticsNodeBase`
- **Location**: `/home/asantanna/DNNE/DNNE-UI/custom_nodes/*_visnode.py`
- **Templates**: `/home/asantanna/DNNE/DNNE-UI/export_system/templates/nodes/*_queue.py`
- **Export**: Generates standalone Python modules with queue-based async execution