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
    "CHECKPOINT_LOAD_ON_START": False,
    "OPTIMIZER_NODE_ID": "optimizer_1"
}

from pathlib import Path
from framework.globals import Global as g

class NetworkNode_{NODE_ID}(QueueNode):
    """Neural network with multiple layers"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input"])
        self.setup_outputs(["layers", "output"])  # model is virtual, not a queue output
        
        # Build network from detected layers: {NETWORK_LAYERS}
        layers = []
{LAYER_DEFINITIONS}
        
        self.network = nn.Sequential(*layers)
        
        # Move to GPU if available
        self.device = torch.device(g.get_device())
        self.network = self.network.to(self.device)
        
        # Sync checking with optimizer
        self.execution_count = 0
        self.optimizer_node_id = "{OPTIMIZER_NODE_ID}"
        self.optimizer_node = None
        
        # Set network to eval mode if in inference
        if g.inference_mode:
            self.network.eval()
            self.node_logger.info("Network set to evaluation mode for inference")
        
        # Checkpoint configuration
        self.checkpoint_enabled = {CHECKPOINT_ENABLED}
        self.checkpoint_trigger_type = "{CHECKPOINT_TRIGGER_TYPE}"
        self.checkpoint_trigger_value = "{CHECKPOINT_TRIGGER_VALUE}"
        self.checkpoint_load_on_start = {CHECKPOINT_LOAD_ON_START}
        self.checkpoint_save_on_exit = True  # Default to save on exit
        self.checkpoint_manager = None
        
        # Always create checkpoint manager if we need to load OR save checkpoints
        # This allows loading even when checkpoint saving is disabled
        if self.checkpoint_enabled or g.load_checkpoint_dir:
            from framework import CheckpointManager, validate_checkpoint_config
            
            # Only validate checkpoint config if saving is enabled
            if self.checkpoint_enabled:
                # Validate checkpoint configuration
                checkpoint_config = {{
                    'enabled': self.checkpoint_enabled,
                    'trigger_type': self.checkpoint_trigger_type,
                    'trigger_value': self.checkpoint_trigger_value
                }}
                
                try:
                    validate_checkpoint_config(checkpoint_config)
                except ValueError as e:
                    self.node_logger.error(f"Checkpoint configuration error: {e}")
                    self.checkpoint_enabled = False
            
            # Create checkpoint manager for loading/saving
            self.checkpoint_manager = CheckpointManager(
                node_id=node_id,
                checkpoint_dir=str(g.save_checkpoint_dir) if g.save_checkpoint_dir else None
            )
            
            if self.checkpoint_enabled:
                self.node_logger.info(f"Checkpoint manager initialized: {self.checkpoint_trigger_type} trigger")
        
        # Handle checkpoint loading - always load if --load-checkpoint-dir is specified
        if g.load_checkpoint_dir:
            # Try to load checkpoint
            if self.checkpoint_manager:
                checkpoint_path = Path(g.load_checkpoint_dir) / f"node_{node_id}"
                if (checkpoint_path / "model.pt").exists():
                    self.node_logger.info(f"Loading checkpoint {checkpoint_path} for node {node_id}")
                    if not self.load_checkpoint(str(g.load_checkpoint_dir)):
                        # Fail-fast: checkpoint exists but couldn't be loaded
                        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}. The checkpoint file exists but could not be loaded.")
                else:
                    self.node_logger.info(f"No checkpoint found for node {node_id}, initializing normally")
                    if g.inference_mode:
                        raise RuntimeError(f"No checkpoint found at {checkpoint_path}. Inference mode requires a saved checkpoint.")
        elif self.checkpoint_enabled and self.checkpoint_load_on_start:
            # Legacy behavior - load on start if configured (when no explicit load dir)
            self.node_logger.debug("No explicit load directory specified, checkpoint_load_on_start ignored")
        
        self.node_logger.info(f"Created network with {NUM_LAYERS} layers: {INPUT_SIZE} -> {OUTPUT_SIZE}")
        
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
            self.node_logger.warning("No checkpoint manager initialized")
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
            self.node_logger.warning("No checkpoint manager initialized")
            return False
        
        # Load checkpoint from command line directory or default
        checkpoint_data = self.checkpoint_manager.load_checkpoint(load_checkpoint_dir)
        if not checkpoint_data:
            self.node_logger.warning("No checkpoint found to load")
            return False
        
        try:
            # Load model state
            self.network.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Print loaded info
            metadata = checkpoint_data.get('metadata', {{}})
            epoch = metadata.get('current_epoch', 'unknown')
            metric = metadata.get('current_metric', 'unknown')
            
            self.node_logger.info(f"Model checkpoint loaded - epoch: {epoch}, metric: {metric}")
            return True
            
        except Exception as e:
            self.node_logger.error(f"Error loading model checkpoint: {e}")
            return False
    
    async def save_checkpoint_on_exit(self, exit_reason: str) -> bool:
        """
        Save checkpoint on exit if enabled
        
        Args:
            exit_reason: Reason for exit ('timeout', 'training_complete', 'keyboard_interrupt', etc.)
            
        Returns:
            bool: True if checkpoint saved successfully
        """
        if not self.checkpoint_enabled or not self.checkpoint_save_on_exit or not self.checkpoint_manager:
            return False
            
        try:
            import time
            
            # Prepare exit metadata
            metadata = {
                'exit_type': 'on_exit',
                'exit_reason': exit_reason,
                'timestamp': time.time(),
                'model_type': type(self.network).__name__,
                'architecture': {
                    'num_layers': {NUM_LAYERS},
                    'input_size': {INPUT_SIZE},
                    'output_size': {OUTPUT_SIZE}
                },
                'model_info': {
                    'num_parameters': sum(p.numel() for p in self.network.parameters()),
                    'trainable_parameters': sum(p.numel() for p in self.network.parameters() if p.requires_grad)
                }
            }
            
            # Save checkpoint with exit metadata
            success = self.checkpoint_manager.save_checkpoint(
                self.network.state_dict(), metadata=metadata
            )
            
            if success:
                self.node_logger.info(f"💾 Exit checkpoint saved for Network node {self.node_id}")
                return True
            else:
                # Only warn if we were actually trying to save (had a save directory)
                if g.save_checkpoint_dir:
                    self.node_logger.warning(f"⚠️ Failed to save exit checkpoint for Network node {self.node_id}")
                return False
                
        except Exception as e:
            self.node_logger.error(f"Error saving exit checkpoint: {e}")
            return False
        
    async def run(self):
        """Standard run method - no model emission needed (virtual connection)"""
        self.running = True
        self.node_logger.info(f"Starting node {self.node_id}")
        
        try:
            # Resolve optimizer connection for sync checking
            if self.optimizer_node_id and self.optimizer_node_id != "None":
                self.optimizer_node = g.graph_runner.get_node(self.optimizer_node_id)
                if self.optimizer_node:
                    if g.disable_sync_check:
                        self.node_logger.info(f"Network({self.node_id}) sync checking disabled by user")
                    else:
                        self.node_logger.info(f"Network({self.node_id}) sync checking enabled with Optimizer({self.optimizer_node_id})")
                else:
                    self.node_logger.warning(f"Could not find optimizer node {self.optimizer_node_id} for sync checking")
            
            # Run the normal compute loop using MultiWaiter
            while self.running:
                # Use MultiWaiter to get inputs efficiently
                inputs = await self.input_waiter.get()
                
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
            self.node_logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    def get_execution_count(self) -> int:
        """Get current execution count for sync checking"""
        return self.execution_count
    
    async def compute(self, input) -> Dict[str, Any]:
        # Sync check with optimizer (enabled by default, disable with --disable-sync-check)
        if self.optimizer_node and not g.inference_mode and not g.disable_sync_check:
            optimizer_count = self.optimizer_node.get_execution_count()
            if self.execution_count != optimizer_count:
                raise RuntimeError(
                    f"🔥 SYNC VIOLATION DETECTED! 🔥\n"
                    f"Network execution count: {self.execution_count}\n"
                    f"Optimizer execution count: {optimizer_count}\n"
                    f"Network is running ahead of optimizer - this will corrupt training!\n"
                    f"Check your dataflow connections or use --disable-sync-check to bypass."
                )
        
        # Increment execution count after sync check
        self.execution_count += 1
        
        # Ensure input is on correct device
        x = input.to(self.device)
        
        # TENSOR DIMENSION STANDARDS:
        # Expected input: [batch_size, features, ...] where dim 0 = batch, dim 1 = features
        # Networks expect 2D input: [batch_size, features]
        
        # Flatten spatial dimensions if present (e.g., images)
        if x.dim() > 2:
            # Preserve batch dimension, flatten everything else into features
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        elif x.dim() == 1:
            # FAIL-FAST: 1D tensors not allowed except for scalar losses
            raise ValueError(
                f"Network received 1D tensor with shape {x.shape}. "
                f"Expected 2D+ tensor with [batch_size, features, ...]"
            )
        
        # Forward pass through the entire network
        # Gradients are preserved if input requires_grad (handled by PyTorch autograd)
        # In inference mode, torch.no_grad() is already applied by GraphRunner
        output = self.network(x)
        
        return {
            "layers": None,  # This output is just for UI connectivity
            "output": output
        }
