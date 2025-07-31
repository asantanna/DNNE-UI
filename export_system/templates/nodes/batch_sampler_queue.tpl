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
        self.seed_control = {SEED_CONTROL}
        
        # Initialize seed tracking
        self.initial_seed = self.seed
        self.current_seed = self.initial_seed
        self.epoch_count = 0
        
    async def compute(self, dataset, schema) -> Dict[str, Any]:
        # Apply seed control logic
        if self.seed_control == "randomize" and self.shuffle:
            # Generate a new random seed for each epoch
            import random
            self.current_seed = random.randint(0, 2**32 - 1)
            self.node_logger.info(f"Randomizing seed for epoch {{self.epoch_count}}: {{self.current_seed}}")
        # else: 'fixed' - keep using initial_seed/current_seed
        
        self.epoch_count += 1
        
        # Create dataloader from dataset
        generator = None
        if self.shuffle and self.seed >= 0:
            generator = torch.Generator()
            generator.manual_seed(self.current_seed)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
            generator=generator
        )
        
        self.node_logger.info(f"Created dataloader with batch_size={{self.batch_size}}, shuffle={{self.shuffle}}, seed={{self.current_seed if self.shuffle else 'N/A'}}")
        
        # Pass through the schema unchanged
        return {
            "dataloader": dataloader,
            "schema": schema
        }