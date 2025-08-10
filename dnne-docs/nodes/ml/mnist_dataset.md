# MNISTDataset Node

## Overview
The MNISTDataset node loads the MNIST handwritten digit dataset, a standard benchmark dataset for image classification containing 70,000 grayscale images of handwritten digits (0-9).

## Properties

- **Category**: `ml`
- **Color Scheme**: Data nodes (defined by node_colors)
- **Implementation**: `custom_nodes/mnist_dataset_visnode.py`

## Inputs

### Required Parameters
- **data_path** (STRING)
  - Default: `"./data"`
  - Directory path where MNIST dataset will be stored or loaded from
  - Creates the directory if it doesn't exist

- **train** (BOOLEAN)
  - Default: `True`
  - Whether to load training set (True) or test set (False)
  - Training set: 60,000 samples
  - Test set: 10,000 samples

- **download** (BOOLEAN)
  - Default: `True`
  - Whether to automatically download the MNIST dataset if not found
  - Set to False if dataset is already downloaded

## Outputs

- **dataset** (DATASET)
  - PyTorch Dataset object containing MNIST data
  - Can be used with BatchSampler for mini-batch processing

- **schema** (SCHEMA)
  - Dataset metadata including:
    - `num_samples`: Total number of samples
    - `input_shape`: Shape of input tensors [1, 28, 28]
    - `num_classes`: 10 (digits 0-9)
    - `class_names`: List of class names ["0", "1", ..., "9"]

## Functionality

1. **Data Loading**: Uses torchvision to load MNIST dataset
2. **Preprocessing**: Applies ToTensor and normalization transforms
3. **Caching**: Downloads dataset once and reuses cached version
4. **Context Integration**: Stores dataset in global context for access by other nodes

## Transform Pipeline
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])
```

## Usage Example

### In Visual Workflow
1. Add MNISTDataset node
2. Set `train=True` for training data
3. Connect to BatchSampler for mini-batch creation
4. Use schema output for network architecture configuration

### Exported Python Code
```python
class MNISTDatasetNode(QueueNode):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.schema = None
    
    async def process(self, data_path, train, download):
        dataset = datasets.MNIST(
            root=data_path,
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        
        schema = {
            "num_samples": len(dataset),
            "input_shape": [1, 28, 28],
            "num_classes": 10,
            "class_names": [str(i) for i in range(10)]
        }
        
        await self.output_queue.put({
            "dataset": dataset,
            "schema": schema
        })
```

## Best Practices

1. **Data Path**: Use consistent data_path across training and evaluation
2. **Download Once**: Set `download=False` after initial download to speed up loading
3. **Train/Test Split**: Use separate nodes for training (`train=True`) and test (`train=False`) datasets
4. **Normalization**: Default normalization values are pre-calculated for MNIST

## Common Issues

- **Download Failures**: Ensure internet connection for first-time download
- **Path Permissions**: Verify write permissions for data_path directory
- **Memory Usage**: Full dataset loads into memory (~50MB)

## Related Nodes

- [BatchSampler](batch_sampler.md) - Create mini-batches from dataset
- [GetBatch](get_batch.md) - Retrieve batches for training
- [CIFAR10Dataset](cifar10_dataset.md) - Similar node for CIFAR-10 dataset