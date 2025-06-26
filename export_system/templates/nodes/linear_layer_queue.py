# Template variables - replaced during export
template_vars = {
    "NODE_ID": "linear_1",
    "CLASS_NAME": "LinearLayerNode",
    "INPUT_SIZE": 784,
    "OUTPUT_SIZE": 128,
    "ACTIVATION": "relu",
    "DROPOUT": 0.0,
    "BIAS": True
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Linear layer with activation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input_tensor"])
        self.setup_outputs(["output_tensor"])
        
        # Create layer
        self.linear = nn.Linear({INPUT_SIZE}, {OUTPUT_SIZE}, bias={BIAS_VALUE})
        self.dropout = nn.Dropout({DROPOUT}) if {DROPOUT} > 0 else None
        self.activation = "{ACTIVATION_VALUE}"
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = self.linear.to(self.device)
        
    async def compute(self, input_tensor) -> Dict[str, Any]:
        # Ensure input is on correct device
        x = input_tensor.to(self.device)
        
        # Flatten if needed (for MNIST)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass
        x = self.linear(x)
        
        # Activation
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        elif self.activation == "sigmoid":
            x = torch.sigmoid(x)
        
        # Dropout if training
        if self.dropout is not None:
            x = self.dropout(x)
        
        return {{"output_tensor": x}}
