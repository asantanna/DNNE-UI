# Template variables - replaced during export

class BatchSamplerNode_{NODE_ID}(QueueNode):
    """Batch sampler that wraps a dataset and emits batches"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["dataset", "schema"])
        self.setup_outputs(["dataloader", "schema"])
        
        # Sampler parameters
        self.batch_size = {BATCH_SIZE}
        self.shuffle = {SHUFFLE}
        self.seed = {SEED}
        
    async def compute(self, dataset, schema) -> Dict[str, Any]:
        # Create dataloader from dataset
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed) if self.shuffle else None
        )
        
        self.logger.info(f"Created dataloader with batch_size={self.batch_size}, shuffle={self.shuffle}")
        
        # Pass through the schema unchanged
        return {
            "dataloader": dataloader,
            "schema": schema
        }