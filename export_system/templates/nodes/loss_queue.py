# Template variables - replaced during export
template_vars = {
    "NODE_ID": "loss_1",
    "CLASS_NAME": "LossNode",
    "LOSS_TYPE": "cross_entropy"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Loss computation node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["predictions", "labels"])
        self.setup_outputs(["loss", "accuracy"])
        
        # Setup loss function
        self.loss_type = "{LOSS_TYPE}"
        if self.loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()
    
    async def compute(self, predictions, labels) -> Dict[str, Any]:
        # Ensure tensors are on the same device
        labels = labels.to(predictions.device)
        
        # Compute loss
        loss = self.criterion(predictions, labels)
        
        # Compute accuracy for classification
        accuracy = 0.0
        if self.loss_type == "cross_entropy":
            _, predicted = torch.max(predictions, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total if total > 0 else 0.0
        
        self.logger.info(f"Loss: {{loss.item():.4f}}, Accuracy: {{accuracy:.2%}}")
        
        return {{
            "loss": loss,
            "accuracy": accuracy
        }}
