# Template variables - replaced during export
template_vars = {
    "NODE_ID": "optimizer_1",
    "LEARNING_RATE": 0.01,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0,
    "BATCH_SIZE": 1,  # Gradient accumulation batch size
    "MODEL_NODE_IDS": '["33"]'  # List of connected model node IDs (can be one or many)
}

from framework.globals import Global as g

class SGDOptimizerNode_{NODE_ID}(QueueNode):
    """SGD Optimizer node that performs training steps"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Loss is the main data input (model connection is virtual)
        self.setup_inputs(required=["loss"])
        self.setup_outputs(["step_complete"])
        
        # Optimizer parameters
        self.learning_rate = {LEARNING_RATE}
        self.momentum = {MOMENTUM}
        self.weight_decay = {WEIGHT_DECAY}
        self.batch_size = {BATCH_SIZE}  # Gradient accumulation batch size
        self.enable_bootstrap = {ENABLE_BOOTSTRAP}
        
        # Virtual connections to model nodes - will be resolved in run()
        self.model_node_ids = {MODEL_NODE_IDS}  # Always a list (can be 1 or many)
        self.model_nodes = []
        self.optimizers = []  # One optimizer per model
        
        # Gradient accumulation state
        self.curr_batch_idx = 0
        self.accumulated_loss = 0.0
        
        # Sync checking with network
        self.execution_count = 0
        
    async def run(self):
        """Override run to setup optimizer(s) with network node(s)"""
        self.running = True
        self.node_logger.info(f"Starting node {self.node_id}")
        
        try:
            # Resolve virtual connections now that all nodes are created
            for model_node_id in self.model_node_ids:
                model_node = g.graph_runner.get_node(model_node_id)
                if not model_node:
                    self.node_logger.error(f"FATAL: Model node {model_node_id} not found in graph")
                    raise RuntimeError(f"Model node {model_node_id} not found - check workflow connections")
                
                if not hasattr(model_node, 'get_parameters'):
                    self.node_logger.error(f"FATAL: Model node {model_node_id} has no get_parameters() method")
                    raise RuntimeError(f"Model node {model_node_id} is not a valid model node")
                
                self.model_nodes.append(model_node)
                
                # Create optimizer for this model
                all_params = list(model_node.get_parameters())
                if not all_params:
                    self.node_logger.error(f"FATAL: Model node {model_node_id} has no trainable parameters")
                    raise RuntimeError(f"Model node {model_node_id} returned no parameters to optimize")
                
                optimizer = optim.SGD(
                    all_params,
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )
                self.optimizers.append(optimizer)
                self.node_logger.info(f"Created SGD optimizer for model {model_node_id} with {len(all_params)} parameter groups")
            
            # FAIL-FAST: Must have exactly one optimizer per model
            if len(self.optimizers) != len(self.model_node_ids):
                self.node_logger.error(f"FATAL: Expected {len(self.model_node_ids)} optimizers but created {len(self.optimizers)}")
                raise RuntimeError(f"Failed to create optimizers for all models - check connections")
            
            num_models = len(self.optimizers)
            if num_models > 1:
                self.node_logger.info(f"Managing {num_models} models with shared loss - using single backward pass")
            else:
                self.node_logger.info(f"Managing 1 model: lr={self.learning_rate}, momentum={self.momentum}")
            
            # Send initial step_complete signal to start the training loop if enabled
            if self.enable_bootstrap:
                import time
                step_signal = {
                    "signal_type": "step_complete",
                    "timestamp": time.time(),
                    "source_node": self.node_id,
                    "metadata": {"phase": "startup"}
                }
                await self.send_output("step_complete", step_signal)
                self.node_logger.info(f"Sent startup step_complete signal")
            else:
                self.node_logger.info(f"Bootstrap trigger disabled by widget setting")
            
            # Now run normal compute loop for loss inputs
            await super().run()
                
        except asyncio.CancelledError:
            self.node_logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    def get_execution_count(self) -> int:
        """Get current execution count for sync checking"""
        return self.execution_count
    
    def get_parameters(self):
        """Return all managed parameters for requires_grad toggling by TrainingSequencer"""
        params = []
        for model_node in self.model_nodes:
            params.extend(model_node.get_parameters())
        return params
    
    def zero_grad_only(self):
        """Zero gradients without backward - used by TrainingSequencer"""
        if not g.inference_mode:
            for optimizer in self.optimizers:
                optimizer.zero_grad()
    
    def backward_only(self, loss, retain_graph=False):
        """Perform backward without step - used by TrainingSequencer"""
        if not g.inference_mode:
            # Scale loss for gradient accumulation
            loss_scaled = loss / self.batch_size
            loss_scaled.backward(retain_graph=retain_graph)
            self.accumulated_loss += loss.item() if hasattr(loss, 'item') else float(loss)
    
    async def step_only(self):
        """Step optimizer without backward - used by TrainingSequencer"""
        # Increment batch index
        self.curr_batch_idx += 1
        
        # Only step optimizer at end of batch
        if self.curr_batch_idx >= self.batch_size:
            if not g.inference_mode:
                for optimizer in self.optimizers:
                    optimizer.step()
                
                if g.verbose and self.batch_size > 1:
                    avg_loss = self.accumulated_loss / self.batch_size
                    self.node_logger.info(f"Batch complete: avg_loss={avg_loss:.4f}, batch_size={self.batch_size}")
            
            # Reset for next batch
            self.curr_batch_idx = 0
            self.accumulated_loss = 0.0
        
        # ALWAYS increment execution count for sync checking (regardless of step)
        if not g.inference_mode:
            self.execution_count += 1
        
        # Always send step_complete signal for workflow synchronization
        import time
        step_signal = {
            "signal_type": "step_complete",
            "timestamp": time.time(),
            "source_node": self.node_id,
            "metadata": {
                "phase": "training_sequencer_step",
                "batch_progress": f"{self.curr_batch_idx}/{self.batch_size}"
            }
        }
        await self.send_output("step_complete", step_signal)
        
        if g.verbose:
            self.node_logger.debug(f"Sent step_complete from step_only() [batch {self.curr_batch_idx}/{self.batch_size}]")
    
    async def compute(self, loss) -> Dict[str, Any]:
        """Perform training step when loss is received"""
        if not self.optimizers:
            raise RuntimeError(
                f"SGDOptimizerNode {self.node_id}: No optimizers created. "
                f"Check that Network nodes are connected and working properly."
            )
            
        # Perform training step (skip in inference mode)
        if not g.inference_mode:
            # Zero gradients only at start of batch
            if self.curr_batch_idx == 0:
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                self.accumulated_loss = 0.0
            
            # Check for retain_graph override (useful for complex architectures)
            # Can be set via: --override all:retain_graph=True
            retain_graph = g.get_node_config(self.node_id, 'retain_graph', False)
            if retain_graph:
                self.node_logger.debug(f"Using retain_graph=True for backward pass")
            
            # Scale loss for gradient accumulation (automatic averaging)
            loss_scaled = loss / self.batch_size
            loss_scaled.backward(retain_graph=retain_graph)
            
            # Track accumulated loss for logging
            self.accumulated_loss += loss.item() if hasattr(loss, 'item') else float(loss)
            
            # Increment batch counter
            self.curr_batch_idx += 1
            
            # Step all optimizers only at end of batch
            if self.curr_batch_idx >= self.batch_size:
                for optimizer in self.optimizers:
                    optimizer.step()
                
                # Log batch completion if using accumulation
                if self.batch_size > 1 and g.verbose:
                    avg_loss = self.accumulated_loss / self.batch_size
                    self.node_logger.info(f"Batch complete: avg_loss={avg_loss:.4f}, batch_size={self.batch_size}")
                
                # Reset for next batch
                self.curr_batch_idx = 0
                self.accumulated_loss = 0.0
            
            # ALWAYS increment execution count for sync checking (regardless of step)
            self.execution_count += 1
        
        # Send step_complete signal for next batch
        import time
        step_signal = {
            "signal_type": "step_complete",
            "timestamp": time.time(),
            "source_node": self.node_id,
            "metadata": {
                "phase": "training_sequencer_step",
                "loss_value": loss.item() if hasattr(loss, 'item') else float(loss)
            }
        }
        
        # Only log in verbose mode - EpochTracker will show summaries
        if g.verbose:
            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
            self.node_logger.info(f"Training step completed. Loss: {loss_val:.4f}")
        
        return {"step_complete": step_signal}