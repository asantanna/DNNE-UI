# Template variables - replaced during export

class GetBatchNode_{NODE_ID}(SensorNode):
    """Get batch from dataloader at fixed rate"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, update_rate=10.0)  # 10 batches per second
        self.setup_inputs(required=["dataloader", "schema"])
        self.setup_outputs(["images", "labels", "epoch_complete"])
        self.dataloader = None
        self.schema = None
        self.data_iter = None
        self.epoch = 0
        
    async def run(self):
        """Override run to wait for dataloader and schema first"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            # Wait for dataloader and schema
            self.dataloader = await self.input_queues["dataloader"].get()
            self.schema = await self.input_queues["schema"].get()
            self.data_iter = iter(self.dataloader)
            
            # Log schema info for debugging
            if "outputs" in self.schema and "images" in self.schema["outputs"]:
                img_info = self.schema["outputs"]["images"]
                self.logger.info(f"Received dataloader with image shape: {img_info.get('shape')}, flattened_size: {img_info.get('flattened_size')}")
            
            self.logger.info("Received dataloader and schema, starting batch generation")
            
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
        try:
            images, labels = next(self.data_iter)
        except StopIteration:
            # Reset iterator at end of epoch
            self.epoch += 1
            epoch_complete = True
            self.data_iter = iter(self.dataloader)
            images, labels = next(self.data_iter)
            self.logger.info(f"Starting epoch {self.epoch}")
        
        return {
            "images": images,
            "labels": labels,
            "epoch_complete": epoch_complete
        }