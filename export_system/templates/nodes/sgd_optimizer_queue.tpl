# Template variables - replaced during export
template_vars = {
    "NODE_ID": "optimizer_1",
    "LEARNING_RATE": 0.01,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0
}

class SGDOptimizerNode_{NODE_ID}(QueueNode):
    """SGD Optimizer node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # No setup_inputs since we manually handle the model queue
        self.setup_inputs(required=[])
        self.setup_outputs(["optimizer"])
        
        # Manually create model queue for one-time config
        from asyncio import Queue
        self.input_queues["model"] = Queue(maxsize=1)
        
        # Optimizer parameters
        self.learning_rate = {LEARNING_RATE}
        self.momentum = {MOMENTUM}
        self.weight_decay = {WEIGHT_DECAY}
        self.optimizer = None
        
    async def run(self):
        """Override run to wait for model connection first"""
        self.running = True
        self.node_logger.info(f"Starting node {self.node_id}")
        
        try:
            # Wait for model connection (network node will send itself)
            model_node = await self.input_queues["model"].get()
            
            # Create optimizer using the connected model node's parameters
            if model_node and hasattr(model_node, 'get_parameters'):
                all_params = list(model_node.get_parameters())
                
                self.optimizer = optim.SGD(
                    all_params,
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )
                self.node_logger.info(f"Created SGD optimizer with {len(all_params)} parameter groups: lr={self.learning_rate}, momentum={self.momentum}")
                
                # Emit optimizer
                await self.send_output("optimizer", self.optimizer)
                
                # Keep running but don't emit again
                while self.running:
                    await asyncio.sleep(1.0)
            else:
                self.node_logger.error("No model node received - cannot create optimizer")
                
        except asyncio.CancelledError:
            self.node_logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Abstract method implementation - not used since we override run()"""
        return {}