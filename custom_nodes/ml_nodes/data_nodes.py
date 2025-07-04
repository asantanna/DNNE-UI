"""
Data loading and batching nodes
"""

import torch
from torch.utils.data import DataLoader
from .base import RoboticsNodeBase, get_context


class MNISTDatasetNode(RoboticsNodeBase):
    """Load MNIST dataset"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_path": ("STRING", {"default": "./data"}),
                "train": ("BOOLEAN", {"default": True}),
                "download": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("DATASET", "SCHEMA")
    RETURN_NAMES = ("dataset", "schema")
    FUNCTION = "load_dataset"
    CATEGORY = "ml/data"

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
    """Create batches from dataset"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET",),
                "schema": ("SCHEMA",),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 512}),
                "shuffle": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1}),  # -1 means random
            }
        }

    RETURN_TYPES = ("DATALOADER", "SCHEMA")
    RETURN_NAMES = ("dataloader", "schema")
    FUNCTION = "create_dataloader"
    CATEGORY = "ml/data"

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
    """Get next batch from dataloader"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataloader": ("DATALOADER",),
                "schema": ("SCHEMA",)
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "BOOLEAN", "DICT")
    RETURN_NAMES = ("images", "labels", "epoch_complete", "epoch_stats")
    FUNCTION = "get_batch"
    CATEGORY = "ml/data"

    def get_batch(self, dataloader, schema):
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