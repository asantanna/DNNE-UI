# Template variables - replaced during export

class GetBatchNode_{NODE_ID}(SensorNode):
    """Get batch from dataloader at fixed rate"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, update_rate=10.0)  # 10 batches per second
        self.setup_inputs(required=["dataloader", "schema"])
        self.setup_outputs(["images", "labels", "epoch_complete", "epoch_stats"])
        self.dataloader = None
        self.schema = None
        self.data_iter = None
        self.epoch = 0
        self.batch_in_epoch = 0
        self.total_batches_per_epoch = 0
        
    async def run(self):
        """Override run to wait for dataloader and schema first"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            # Wait for dataloader and schema
            self.dataloader = await self.input_queues["dataloader"].get()
            self.schema = await self.input_queues["schema"].get()
            self.data_iter = iter(self.dataloader)
            self.total_batches_per_epoch = len(self.dataloader)
            
            # Log schema info for debugging
            if "outputs" in self.schema and "images" in self.schema["outputs"]:
                img_info = self.schema["outputs"]["images"]
                self.logger.info(f"Received dataloader with image shape: {img_info.get('shape')}, flattened_size: {img_info.get('flattened_size')}")
            
            self.logger.info(f"Received dataloader with {self.total_batches_per_epoch} batches per epoch, starting batch generation")
            
            # Now run the sensor loop
            await super().run()
            
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
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
            self.logger.info(f"📊 Completed epoch {epoch_stats['epoch']} ({epoch_stats['total_batches']} batches)")
            self.logger.info(f"🚀 Starting epoch {self.epoch}")
        
        # Create batch progress info
        if not epoch_stats:
            epoch_stats = {
                "epoch": self.epoch,
                "batch": self.batch_in_epoch,
                "total_batches": self.total_batches_per_epoch,
                "progress": self.batch_in_epoch / self.total_batches_per_epoch,
                "completed": False
            }
        
        return {
            "images": images,
            "labels": labels,
            "epoch_complete": epoch_complete,
            "epoch_stats": epoch_stats
        }