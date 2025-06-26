# Template variables - replaced during export
template_vars = {
    "NODE_ID": "21",
    "CLASS_NAME": "TrainingStepNode"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Training step node that performs backpropagation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["loss", "optimizer"])
        self.setup_outputs(["step_complete"])
        
    async def compute(self, loss, optimizer) -> Dict[str, Any]:
        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.logger.info(f"Training step completed. Loss: {loss.item():.4f}")
        
        return {"step_complete": True}