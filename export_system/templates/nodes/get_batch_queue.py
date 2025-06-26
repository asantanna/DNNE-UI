# Template variables - replaced during export
template_vars = {
    "NODE_ID": "15",
    "CLASS_NAME": "GetBatchNode"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Get batch from dataloader"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["dataloader"])
        self.setup_outputs(["batch_data", "batch_labels"])
        
    async def compute(self, dataloader) -> Dict[str, Any]:
        # In a real implementation, this would coordinate with the dataloader
        # For now, just pass through the data
        if isinstance(dataloader, tuple) and len(dataloader) == 2:
            images, labels = dataloader
            return {
                "batch_data": images,
                "batch_labels": labels
            }
        else:
            self.logger.warning("Unexpected dataloader format")
            return {
                "batch_data": dataloader,
                "batch_labels": None
            }