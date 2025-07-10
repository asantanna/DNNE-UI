# Template variables - replaced during export
template_vars = {
    "NODE_ID": "network_1",
    "NETWORK_LAYERS": [],
    "LAYER_DEFINITIONS": "",
    "NUM_LAYERS": 3,
    "INPUT_SIZE": 784,
    "OUTPUT_SIZE": 10,
    "CHECKPOINT_ENABLED": False,
    "CHECKPOINT_TRIGGER_TYPE": "epoch",
    "CHECKPOINT_TRIGGER_VALUE": "50",
    "CHECKPOINT_LOAD_ON_START": False
}

class NetworkNode_{NODE_ID}(QueueNode):
    """Neural network with multiple layers"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input"])
        self.setup_outputs(["layers", "output", "model"])
        
        # Build network from detected layers: {NETWORK_LAYERS}
        layers = []
{LAYER_DEFINITIONS}
        
        self.network = nn.Sequential(*layers)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self.network.to(self.device)
        
        # Checkpoint configuration
        self.checkpoint_enabled = {CHECKPOINT_ENABLED}
        self.checkpoint_trigger_type = "{CHECKPOINT_TRIGGER_TYPE}"
        self.checkpoint_trigger_value = "{CHECKPOINT_TRIGGER_VALUE}"
        self.checkpoint_load_on_start = {CHECKPOINT_LOAD_ON_START}
        self.checkpoint_manager = None
        
        # Initialize checkpoint manager if enabled
        if self.checkpoint_enabled:
            from run_utils import CheckpointManager, validate_checkpoint_config
            
            # Validate checkpoint configuration
            checkpoint_config = {{
                'enabled': self.checkpoint_enabled,
                'trigger_type': self.checkpoint_trigger_type,
                'trigger_value': self.checkpoint_trigger_value
            }}
            
            try:
                validate_checkpoint_config(checkpoint_config)
                # Get checkpoint directory from command line args (set by runner.py)
                try:
                    import builtins
                    save_checkpoint_dir = getattr(builtins, 'SAVE_CHECKPOINT_DIR', None)
                    load_checkpoint_dir = getattr(builtins, 'LOAD_CHECKPOINT_DIR', None)
                except:
                    save_checkpoint_dir = None
                    load_checkpoint_dir = None
                    
                self.checkpoint_manager = CheckpointManager(
                    node_id=node_id,
                    checkpoint_dir=save_checkpoint_dir
                )
                self.logger.info(f"Checkpoint manager initialized: {self.checkpoint_trigger_type} trigger")
                
                # Load checkpoint on start if requested
                if self.checkpoint_load_on_start and load_checkpoint_dir:
                    self.load_checkpoint(load_checkpoint_dir)
                    
            except ValueError as e:
                self.logger.error(f"Checkpoint configuration error: {e}")
                self.checkpoint_enabled = False
        
        self.logger.info(f"Created network with {NUM_LAYERS} layers: {INPUT_SIZE} -> {OUTPUT_SIZE}")
        
    def get_parameters(self):
        """Return network parameters for optimizer"""
        return self.network.parameters()
    
    def save_checkpoint(self, trigger_type="external", trigger_value=None, 
                       current_epoch=None, current_metric=None, metadata=None):
        """
        Save model checkpoint
        
        Args:
            trigger_type: Type of trigger ('epoch', 'time', 'best_metric', 'external')
            trigger_value: Value for the trigger (depends on type)
            current_epoch: Current epoch number (for epoch-based triggers)
            current_metric: Current metric value (for best metric triggers)
            metadata: Additional metadata to include
            
        Returns:
            str: Path to saved checkpoint file, or None if not saved
        """
        if not self.checkpoint_manager:
            self.logger.warning("No checkpoint manager initialized")
            return None
        
        # Check if we should checkpoint
        should_checkpoint = self.checkpoint_manager.should_checkpoint(
            trigger_type, trigger_value, current_epoch, current_metric
        )
        
        if should_checkpoint:
            # Prepare metadata with model information
            checkpoint_metadata = {{
                'trigger_type': trigger_type,
                'trigger_value': trigger_value,
                'current_epoch': current_epoch,
                'current_metric': current_metric,
                'model_type': type(self.network).__name__,
                'architecture': {{
                    'num_layers': {NUM_LAYERS},
                    'input_size': {INPUT_SIZE},
                    'output_size': {OUTPUT_SIZE}
                }},
                'model_info': {{
                    'num_parameters': sum(p.numel() for p in self.network.parameters()),
                    'trainable_parameters': sum(p.numel() for p in self.network.parameters() if p.requires_grad)
                }}
            }}
            
            if metadata:
                checkpoint_metadata.update(metadata)
            
            # Save checkpoint (only model weights + metadata)
            success = self.checkpoint_manager.save_checkpoint(
                self.network.state_dict(), metadata=checkpoint_metadata
            )
            return success
        
        return None
    
    def load_checkpoint(self, load_checkpoint_dir=None):
        """
        Load model checkpoint
        
        Args:
            load_checkpoint_dir: Override load directory (from command line)
            
        Returns:
            bool: True if checkpoint loaded successfully
        """
        if not self.checkpoint_manager:
            self.logger.warning("No checkpoint manager initialized")
            return False
        
        # Load checkpoint from command line directory or default
        checkpoint_data = self.checkpoint_manager.load_checkpoint(load_checkpoint_dir)
        if not checkpoint_data:
            self.logger.warning("No checkpoint found to load")
            return False
        
        try:
            # Load model state
            self.network.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Print loaded info
            metadata = checkpoint_data.get('metadata', {{}})
            epoch = metadata.get('current_epoch', 'unknown')
            metric = metadata.get('current_metric', 'unknown')
            
            self.logger.info(f"Model checkpoint loaded - epoch: {epoch}, metric: {metric}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model checkpoint: {e}")
            return False
        
    async def run(self):
        """Override run to emit model reference once at startup"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            # Emit model reference once for optimizer
            await self.send_output("model", self)
            
            # Now run the normal compute loop
            while self.running:
                # Gather all required inputs
                inputs = {}
                for input_name in self.required_inputs:
                    value = await self.input_queues[input_name].get()
                    inputs[input_name] = value
                
                # Execute compute
                start_time = time.time()
                outputs = await self.compute(**inputs)
                self.last_compute_time = time.time() - start_time
                self.compute_count += 1
                
                # Send outputs (except model which was already sent)
                for output_name, value in outputs.items():
                    if output_name != "model":
                        await self.send_output(output_name, value)
                        
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def compute(self, input) -> Dict[str, Any]:
        # Ensure input is on correct device
        x = input.to(self.device)
        
        # Flatten if needed (for MNIST)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        
        # Forward pass through the entire network
        output = self.network(x)
        
        return {
            "layers": None,  # This output is just for UI connectivity
            "output": output
        }
