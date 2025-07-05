"""Node implementation for TrainingStep (ID: 45)"""
import asyncio
import torch
from typing import Dict, Any
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class TrainingStepNode_45(QueueNode):
    """Training step node that performs backpropagation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Set up both inputs initially so queues are created for connections
        self.setup_inputs(required=["loss", "optimizer"])
        self.setup_outputs(["step_complete"])
        self.optimizer = None
        
    async def run(self):
        """Override run to get optimizer once, then process loss inputs"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            # First, wait for optimizer (configuration)
            self.optimizer = await self.input_queues["optimizer"].get()
            self.logger.info(f"Received optimizer for training")
            
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
            return {"step_complete": False}
            
        # Enable anomaly detection to find the inplace operation
        torch.autograd.set_detect_anomaly(True)
        
        try:
            # Perform backpropagation with gradient computation synchronization
            self.optimizer.zero_grad()
            loss.backward()
            
            # Ensure gradient computation is complete before the optimizer step
            # This prevents race conditions in high-speed async processing
            if loss.device.type == 'cuda':
                torch.cuda.synchronize()
            
            self.optimizer.step()
            
        except RuntimeError as e:
            self.logger.error(f"Gradient computation error: {e}")
            # Re-raise to get the detailed traceback
            raise
        finally:
            torch.autograd.set_detect_anomaly(False)
        
        # Add small delay to prevent overwhelming the async queue system
        # This ensures gradient computation is fully complete before next forward pass
        await asyncio.sleep(0.001)  # 1ms delay
        
        # Only log in verbose mode - EpochTracker will show summaries
        import builtins
        if hasattr(builtins, 'VERBOSE') and builtins.VERBOSE:
            self.logger.info(f"Training step completed. Loss: {loss.item():.4f}")
        
        return {"step_complete": True}
