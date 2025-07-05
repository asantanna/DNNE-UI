"""Node implementation for TrainingStep (ID: 45)"""
import asyncio
import time
from typing import Dict, Any
import torch
import asyncio
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class TrainingStepNode_45(QueueNode):
    """Training step node that performs backpropagation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Set up both inputs initially so queues are created for connections
        self.setup_inputs(required=["loss", "optimizer"])
        self.setup_outputs(["ready", "step_complete"])
        self.optimizer = None
        
    async def run(self):
        """Override run to get optimizer once, then process loss inputs"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            # First, wait for optimizer (configuration)
            self.optimizer = await self.input_queues["optimizer"].get()
            self.logger.info(f"Received optimizer for training")
            
            # Send initial ready signal to start the training loop
            import time
            ready_signal = {
                "signal_type": "ready",
                "timestamp": time.time(),
                "source_node": self.node_id,
                "metadata": {"phase": "startup"}
            }
            await self.send_output("ready", ready_signal)
            self.logger.info(f"Sent startup ready signal")
            
            # Now change required inputs to only loss (data flow)
            self.required_inputs = ["loss"]
            
            # Run normal compute loop for loss inputs only
            await super().run()
            
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
        
    async def compute(self, loss) -> Dict[str, Any]:
        if self.optimizer is None:
            self.logger.error("No optimizer available for training step")
            return {"ready": None, "step_complete": False}
            
        # Perform backpropagation
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
        import builtins
        if hasattr(builtins, 'VERBOSE') and builtins.VERBOSE:
            self.logger.info(f"Training step completed. Loss: {loss.item():.4f}")
        
        return {
            "ready": ready_signal,
            "step_complete": True
        }
