# Template variables - replaced during export
template_vars = {
    "NODE_ID": "20",
    "CLASS_NAME": "SGDOptimizerNode",
    "LEARNING_RATE": 0.01,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """SGD Optimizer node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["model_params"])
        self.setup_outputs(["optimizer"])
        
        # Optimizer parameters
        self.learning_rate = {LEARNING_RATE}
        self.momentum = {MOMENTUM}
        self.weight_decay = {WEIGHT_DECAY}
        self.optimizer = None
        
    async def compute(self, model_params) -> Dict[str, Any]:
        # Create optimizer if not exists
        if self.optimizer is None:
            self.optimizer = optim.SGD(
                model_params,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
            self.logger.info(f"Created SGD optimizer: lr={self.learning_rate}, momentum={self.momentum}")
        
        return {"optimizer": self.optimizer}