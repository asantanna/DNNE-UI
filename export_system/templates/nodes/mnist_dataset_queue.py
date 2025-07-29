# Template variables - replaced during export
from framework.globals import dnne_logging

# MNIST subsystem logger
mnist_logger = dnne_logging.getLogger("mnist")

template_vars = {
    "NODE_ID": "mnist_1",
    "CLASS_NAME": "MNISTDatasetNode",
    "DATA_PATH": "./data",
    "TRAIN": True,
    "DOWNLOAD": True,
    "BATCH_SIZE": 32,
    "EMIT_RATE": 10.0
}

class {CLASS_NAME}_{NODE_ID}(SensorNode):
    """MNIST dataset loader that emits batches at fixed rate"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, update_rate={EMIT_RATE})
        self.setup_outputs(["batch_data", "batch_labels"])
        
        # Setup dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.dataset = datasets.MNIST(
            root="{DATA_PATH}",
            train={TRAIN},
            download={DOWNLOAD},
            transform=transform
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size={BATCH_SIZE},
            shuffle=True,
            num_workers=0
        )
        
        self.data_iter = iter(self.dataloader)
        self.epoch = 0
    
    async def compute(self) -> Dict[str, Any]:
        try:
            images, labels = next(self.data_iter)
        except StopIteration:
            # Reset iterator at end of epoch
            self.epoch += 1
            self.data_iter = iter(self.dataloader)
            images, labels = next(self.data_iter)
            mnist_logger.info(f"Starting epoch {{self.epoch}}")
        
        return {{
            "batch_data": images,
            "batch_labels": labels
        }}
