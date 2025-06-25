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

    RETURN_TYPES = ("DATASET", "INT", "INT")
    RETURN_NAMES = ("dataset", "num_samples", "num_classes")
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

        return (dataset, len(dataset), 10)  # MNIST has 10 classes


class BatchSamplerNode(RoboticsNodeBase):
    """Create batches from dataset"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET",),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 512}),
                "shuffle": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1}),  # -1 means random
            }
        }

    RETURN_TYPES = ("DATALOADER",)
    RETURN_NAMES = ("dataloader",)
    FUNCTION = "create_dataloader"
    CATEGORY = "ml/data"

    def create_dataloader(self, dataset, batch_size, shuffle, seed):
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

        return (dataloader,)


class GetBatchNode(RoboticsNodeBase):
    """Get next batch from dataloader"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataloader": ("DATALOADER",)
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "BOOLEAN")
    RETURN_NAMES = ("images", "labels", "epoch_complete")
    FUNCTION = "get_batch"
    CATEGORY = "ml/data"

    def get_batch(self, dataloader):
        context = get_context()
        
        # Get or create iterator in context
        if "dataloader_iter" not in context.memory:
            context.memory["dataloader_iter"] = iter(dataloader)
            context.memory["epoch_complete"] = False

        try:
            images, labels = next(context.memory["dataloader_iter"])
            epoch_complete = False
        except StopIteration:
            # Reset iterator for next epoch
            context.memory["dataloader_iter"] = iter(dataloader)
            images, labels = next(context.memory["dataloader_iter"])
            epoch_complete = True
            context.episode_count += 1

        return (images, labels, epoch_complete)