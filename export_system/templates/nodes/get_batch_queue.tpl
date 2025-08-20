# Template variables - replaced during export

import torch
from framework.globals import Global as g

class GetBatchNode_{NODE_ID}(QueueNode):
    """Get batch from dataloader as fast as possible"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Only trigger is a repeated input, dataloader and schema are one-time configs
        self.setup_inputs(required=[])
        self.setup_outputs(["images", "labels", "epoch_complete", "epoch_stats"])
        
        # Manually create queues for all inputs
        from asyncio import Queue
        self.input_queues["dataloader"] = Queue(maxsize=1)  # One-time config
        self.input_queues["schema"] = Queue(maxsize=1)  # One-time config
        self.input_queues["trigger"] = Queue(maxsize=10)  # Repeated trigger signals
        self.dataloader = None
        self.schema = None
        self.data_iter = None
        self.epoch = 0
        self.batch_in_epoch = 0
        self.total_batches_per_epoch = 0
        
    async def run(self):
        """Override run to wait for dataloader, schema, and triggers"""
        self.running = True
        self.node_logger.info(f"Starting node {self.node_id}")
        
        try:
            # Wait for dataloader and schema
            self.dataloader = await self.input_queues["dataloader"].get()
            self.schema = await self.input_queues["schema"].get()
            self.data_iter = iter(self.dataloader)
            self.total_batches_per_epoch = len(self.dataloader)
            
            # Log schema info for debugging
            if "outputs" in self.schema and "images" in self.schema["outputs"]:
                img_info = self.schema["outputs"]["images"]
                self.node_logger.info(f"Received dataloader with image shape: {img_info.get('shape')}, flattened_size: {img_info.get('flattened_size')}")
            
            self.node_logger.info(f"Received dataloader with {self.total_batches_per_epoch} batches per epoch, waiting for trigger signals")
            
            # Always wait for trigger signals before generating batches
            while self.running:
                # Wait for trigger signal
                trigger_signal = await self.input_queues["trigger"].get()
                # Too noisy - commenting out per-batch trigger logging
                # self.node_logger.info(f"Received trigger signal: {trigger_signal.get('signal_type', 'unknown')}")
                
                # Generate batch when triggered
                outputs = await self.compute()
                if outputs:
                    for output_name, value in outputs.items():
                        await self.send_output(output_name, value)
            
        except asyncio.CancelledError:
            self.node_logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def compute(self) -> Dict[str, Any]:
        if self.data_iter is None:
            return {}
            
        epoch_complete = False
        epoch_stats = None
        
        try:
            images, labels = next(self.data_iter)
            self.batch_in_epoch += 1
        except StopIteration:
            # End of epoch - create stats before resetting
            epoch_stats = {
                "epoch": self.epoch,
                "total_batches": self.batch_in_epoch,
                "completed": True
            }
            
            # Reset for next epoch
            self.epoch += 1
            self.batch_in_epoch = 1  # Start at 1 for the batch we're about to return
            epoch_complete = True
            self.data_iter = iter(self.dataloader)
            images, labels = next(self.data_iter)
            self.node_logger.info(f"📊 Completed epoch {epoch_stats['epoch']} ({epoch_stats['total_batches']} batches)")
            self.node_logger.info(f"🚀 Starting epoch {self.epoch}")
        
        # Create batch progress info
        if not epoch_stats:
            epoch_stats = {
                "epoch": self.epoch,
                "batch": self.batch_in_epoch,
                "total_batches": self.total_batches_per_epoch,
                "progress": self.batch_in_epoch / self.total_batches_per_epoch,
                "completed": False
            }
        
        # TENSOR DIMENSION STANDARDS:
        # Images should be [batch_size, channels, height, width] or [batch_size, features]
        # Labels should be [batch_size]
        # DataLoader already handles this correctly
        
        # Move to configured device and set gradient requirements
        device = torch.device(g.get_device())
        images = images.to(device)
        labels = labels.to(device)
        
        # Enable gradients for training mode
        if not g.inference_mode:
            images.requires_grad_(True)
        
        return {
            "images": images,
            "labels": labels,
            "epoch_complete": epoch_complete,
            "epoch_stats": epoch_stats
        }