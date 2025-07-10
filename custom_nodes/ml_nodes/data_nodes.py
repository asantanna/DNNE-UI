"""
Data loading and batching nodes
"""

import torch
from torch.utils.data import DataLoader
from inspect import cleandoc
from .base import RoboticsNodeBase, get_context


class MNISTDatasetNode(RoboticsNodeBase):
    """MNIST Dataset Node
    Loads the MNIST handwritten digit dataset for training or testing."""
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_path": ("STRING", {
                    "default": "./data",
                    "tooltip": "Directory path where MNIST dataset will be stored or loaded from. Creates the directory if it doesn't exist."
                }),
                "train": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to load training set (True) or test set (False). Training set has 60,000 samples, test set has 10,000 samples."
                }),
                "download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to automatically download the MNIST dataset if not found in data_path. Set to False if dataset is already downloaded."
                }),
            }
        }

    RETURN_TYPES = ("DATASET", "SCHEMA")
    RETURN_NAMES = ("dataset", "schema")
    FUNCTION = "load_dataset"
    CATEGORY = "ml"

    def load_dataset(self, data_path, train, download):
        # Import here to avoid dependency if not used
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )
        
        # Create schema describing the dataset
        schema = {
            "outputs": {
                "images": {
                    "type": "tensor",
                    "shape": (28, 28),
                    "flattened_size": 784,
                    "dtype": "float32"
                },
                "labels": {
                    "type": "tensor", 
                    "shape": (),
                    "num_classes": 10,
                    "dtype": "int64"
                }
            },
            "num_samples": len(dataset)
        }

        return (dataset, schema)


class BatchSamplerNode(RoboticsNodeBase):
    """Batch Sampler Node
    Creates a DataLoader that provides batched samples from a dataset with configurable batch size and shuffling."""
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {
                    "tooltip": "Input dataset to create batches from. Should be a PyTorch dataset object (e.g., from MNISTDatasetNode)."
                }),
                "schema": ("SCHEMA", {
                    "tooltip": "Dataset schema containing metadata about data shapes, types, and structure. Used for validation and downstream processing."
                }),
                "batch_size": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 512,
                    "tooltip": "Number of samples per batch. Larger batches use more memory but can improve training stability. Common values: 16, 32, 64, 128."
                }),
                "shuffle": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to randomly shuffle the dataset order each epoch. Generally True for training, False for evaluation."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "tooltip": "Random seed for reproducible shuffling. Set to -1 for random seed, or any positive integer for deterministic shuffling."
                }),
            }
        }

    RETURN_TYPES = ("DATALOADER", "SCHEMA")
    RETURN_NAMES = ("dataloader", "schema")
    FUNCTION = "create_dataloader"
    CATEGORY = "ml"

    def create_dataloader(self, dataset, schema, batch_size, shuffle, seed):
        # Set seed if specified
        generator = None
        if seed >= 0:
            generator = torch.Generator()
            generator.manual_seed(seed)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Pass through the schema unchanged
        return (dataloader, schema)


class GetBatchNode(RoboticsNodeBase):
    """Get Batch Node
    Retrieves the next batch from a DataLoader and tracks epoch progress and statistics."""
    
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataloader": ("DATALOADER", {
                    "tooltip": "Configured PyTorch DataLoader that provides batched data. Should come from BatchSamplerNode."
                }),
                "schema": ("SCHEMA", {
                    "tooltip": "Dataset schema with metadata about batch structure and data types. Used for output validation and downstream nodes."
                }),
                "trigger": ("SYNC", {
                    "tooltip": "Synchronization trigger that controls when to fetch the next batch. Connect to training loop or other control nodes."
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "BOOLEAN", "DICT")
    RETURN_NAMES = ("images", "labels", "epoch_complete", "epoch_stats")
    FUNCTION = "get_batch"
    CATEGORY = "ml"

    def get_batch(self, dataloader, schema, trigger):
        context = get_context()
        
        # Initialize tracking variables
        if "dataloader_iter" not in context.memory:
            context.memory["dataloader_iter"] = iter(dataloader)
            context.memory["epoch_complete"] = False
            context.memory["current_epoch"] = 0
            context.memory["batch_in_epoch"] = 0
            context.memory["total_batches_per_epoch"] = len(dataloader)

        try:
            images, labels = next(context.memory["dataloader_iter"])
            context.memory["batch_in_epoch"] += 1
            epoch_complete = False
        except StopIteration:
            # End of epoch
            context.memory["dataloader_iter"] = iter(dataloader)
            images, labels = next(context.memory["dataloader_iter"])
            epoch_complete = True
            context.memory["current_epoch"] += 1
            context.memory["batch_in_epoch"] = 1
            context.episode_count += 1

        # Create epoch stats
        epoch_stats = {
            "epoch": context.memory["current_epoch"],
            "batch": context.memory["batch_in_epoch"],
            "total_batches": context.memory["total_batches_per_epoch"],
            "progress": context.memory["batch_in_epoch"] / context.memory["total_batches_per_epoch"],
            "completed": epoch_complete
        }

        return (images, labels, epoch_complete, epoch_stats)