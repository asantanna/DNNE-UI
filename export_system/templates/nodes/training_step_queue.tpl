# Template variables - replaced during export
template_vars = {
    "NODE_ID": "21",
    "CLASS_NAME": "TrainingStepNode"
}

from framework.globals import Global as g

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Training step node that performs backpropagation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Loss is the main data input
        self.setup_inputs(required=["loss"])
        self.setup_outputs(["ready", "step_complete"])
        
        # Manually create optimizer queue for config input (handled via get_config_inputs)
        from asyncio import Queue
        self.input_queues["optimizer"] = Queue(maxsize=1)  # Config input - only one optimizer
        
        self.optimizer = None
        
    async def run(self):
        """Override run to get optimizer once, then process loss inputs"""
        self.running = True
        self.node_logger.info(f"Starting node {{self.node_id}}")
        
        try:
            # First, get optimizer as a configuration input
            config = await self.get_config_inputs(["optimizer"])
            self.optimizer = config["optimizer"]
            self.node_logger.info(f"Received optimizer for training")
            
            # Send initial ready signal to start the training loop
            import time
            ready_signal = {
                "signal_type": "ready",
                "timestamp": time.time(),
                "source_node": self.node_id,
                "metadata": {"phase": "startup"}
            }
            await self.send_output("ready", ready_signal)
            self.node_logger.info(f"Sent startup ready signal")
            
            # Now run normal compute loop for loss inputs
            # MultiWaiter already knows loss is the only required input
            await super().run()
            
        except asyncio.CancelledError:
            self.node_logger.info(f"Node {{self.node_id}} cancelled")
            raise
        finally:
            self.running = False
        
    async def compute(self, loss) -> Dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError(
                f"TrainingStepNode {self.node_id}: No optimizer received. "
                f"Check that SGDOptimizer node is connected and working properly."
            )
            
        # Perform backpropagation (skip in inference mode)
        if not g.inference_mode:
            # Standard training step - let PyTorch fail-fast if gradients missing
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Send ready signal for next batch after training step completes
        import time
        ready_signal = {
            "signal_type": "ready",
            "timestamp": time.time(),
            "source_node": self.node_id,
            "metadata": {
                "phase": "training_complete",
                "loss_value": loss.item()
            }
        }
        
        # Only log in verbose mode - EpochTracker will show summaries
        if g.verbose:
            self.node_logger.info(f"Training step completed. Loss: {{loss.item():.4f}}")
        
        return {
            "ready": ready_signal,
            "step_complete": True
        }