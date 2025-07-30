# Template variables - replaced during export

class CIFAR10DatasetNode_{NODE_ID}(QueueNode):
    """CIFAR-10 dataset loader"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])  # No inputs
        self.setup_outputs(["dataset", "schema"])
        
        # Setup dataset with CIFAR-10 specific normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                               std=(0.2023, 0.1994, 0.2010))
        ])
        
        self.dataset = datasets.CIFAR10(
            root="{DATA_PATH}",
            train={TRAIN},
            download={DOWNLOAD},
            transform=transform
        )
        
        # Create schema describing the dataset
        # CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        self.schema = {
            "outputs": {
                "images": {
                    "type": "tensor",
                    "shape": (3, 32, 32),
                    "flattened_size": 3072,
                    "dtype": "float32"
                },
                "labels": {
                    "type": "tensor", 
                    "shape": (),
                    "num_classes": 10,
                    "dtype": "int64"
                }
            },
            "num_samples": len(self.dataset)
        }
        
    async def compute(self) -> Dict[str, Any]:
        # Return dataset and its schema
        return {
            "dataset": self.dataset,
            "schema": self.schema
        }
    
    async def run(self):
        """Override run to emit dataset once"""
        self.running = True
        self.node_logger.info(f"Starting node {self.node_id}")
        
        try:
            # Emit dataset once
            outputs = await self.compute()
            for output_name, value in outputs.items():
                await self.send_output(output_name, value)
            
            # Keep running but don't emit again
            while self.running:
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            self.node_logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False