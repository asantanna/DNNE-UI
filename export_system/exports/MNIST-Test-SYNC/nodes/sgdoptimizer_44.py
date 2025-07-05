"""Node implementation for SGDOptimizer (ID: 44)"""
import asyncio
from typing import Dict, Any
import torch.optim as optim
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class SGDOptimizerNode_44(QueueNode):
    """SGD Optimizer node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["network"])  # Connection from network node
        self.setup_outputs(["optimizer"])
        
        # Optimizer parameters
        self.learning_rate = 0.10000000000000002
        self.momentum = 0.9
        self.weight_decay = 0.0
        self.optimizer = None
        
    async def run(self):
        """Override run to wait for model connection first"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            # Wait for network connection (network node will send itself)
            network_node = await self.input_queues["network"].get()
            
            # Create optimizer using the connected network node's parameters
            if network_node and hasattr(network_node, 'get_parameters'):
                all_params = list(network_node.get_parameters())
                
                self.optimizer = optim.SGD(
                    all_params,
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )
                self.logger.info(f"Created SGD optimizer with {len(all_params)} parameter groups: lr={self.learning_rate}, momentum={self.momentum}")
                
                # Emit optimizer
                await self.send_output("optimizer", self.optimizer)
                
                # Keep running but don't emit again
                while self.running:
                    await asyncio.sleep(1.0)
            else:
                self.logger.error("No network node received - cannot create optimizer")
                
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Abstract method implementation - not used since we override run()"""
        return {}
