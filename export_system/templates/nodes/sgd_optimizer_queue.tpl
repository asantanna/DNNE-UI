# Template variables - replaced during export
template_vars = {
    "NODE_ID": "optimizer_1",
    "LEARNING_RATE": 0.01,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0,
    "NETWORK_NODE_ID": "network_1"  # Virtual connection to network node
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
        self.enable_bootstrap = {ENABLE_BOOTSTRAP}
        self.optimizer = None
        
        # Virtual connection to network node - will be resolved in run()
        self.model_node_id = "{NETWORK_NODE_ID}"
        self.model_node = None
        
        # Sync checking with network
        self.execution_count = 0
        
    async def run(self):
        """Override run to setup optimizer with network node"""
        self.running = True
        self.node_logger.info(f"Starting node {self.node_id}")
        
        try:
            # Resolve virtual connection now that all nodes are created
            self.model_node = g.graph_runner.get_node(self.model_node_id)
            
            # Create optimizer using the connected model node's parameters
            if self.model_node and hasattr(self.model_node, 'get_parameters'):
                all_params = list(self.model_node.get_parameters())
                
                self.optimizer = optim.SGD(
                    all_params,
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )
                self.node_logger.info(f"Created SGD optimizer with {len(all_params)} parameter groups: lr={self.learning_rate}, momentum={self.momentum}")
                
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
            else:
                self.node_logger.error("No model node received - cannot create optimizer")
                
        except asyncio.CancelledError:
            self.node_logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    def get_execution_count(self) -> int:
        """Get current execution count for sync checking"""
        return self.execution_count
    
    async def compute(self, loss) -> Dict[str, Any]:
        """Perform training step when loss is received"""
        if self.optimizer is None:
            raise RuntimeError(
                f"SGDOptimizerNode {self.node_id}: No optimizer created. "
                f"Check that Network node is connected and working properly."
            )
            
        # Perform training step (skip in inference mode)
        if not g.inference_mode:
            # Standard training step - let PyTorch fail-fast if gradients missing
            self.optimizer.zero_grad()
            
            # Check for retain_graph override (for multiple optimizers sharing loss)
            # Can be set via: --override all:retain_graph=True
            retain_graph = g.get_node_config(self.node_id, 'retain_graph', False)
            if retain_graph:
                self.node_logger.debug(f"Using retain_graph=True for backward pass")
            loss.backward(retain_graph=retain_graph)
            
            self.optimizer.step()
            
            # Increment execution count after successful step
            self.execution_count += 1
        
        # Send step_complete signal for next batch
        import time
        step_signal = {
            "signal_type": "step_complete",
            "timestamp": time.time(),
            "source_node": self.node_id,
            "metadata": {
                "phase": "training_complete",
                "loss_value": loss.item() if hasattr(loss, 'item') else float(loss)
            }
        }
        
        # Only log in verbose mode - EpochTracker will show summaries
        if g.verbose:
            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
            self.node_logger.info(f"Training step completed. Loss: {loss_val:.4f}")
        
        return {"step_complete": step_signal}