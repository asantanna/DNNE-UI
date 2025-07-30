# Template variables - replaced during export
template_vars = {
    "NODE_ID": "linear_1",
    "CLASS_NAME": "LinearLayerNode",
    "INPUT_SIZE": 784,
    "OUTPUT_SIZE": 128,
    "ACTIVATION": "relu",
    "DROPOUT": 0.0,
    "BIAS": True,
    "WEIGHT_INIT": "auto"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Linear layer with activation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input"])
        self.setup_outputs(["output"])
        
        # Create layer
        self.linear = nn.Linear({INPUT_SIZE}, {OUTPUT_SIZE}, bias={BIAS_VALUE})
        self.dropout = nn.Dropout({DROPOUT}) if {DROPOUT} > 0 else None
        self.activation = "{ACTIVATION_VALUE}"
        
        # Initialize weights
        self._initialize_weights("{WEIGHT_INIT}")
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear = self.linear.to(self.device)
        
    def get_parameters(self):
        """Return model parameters for optimizer"""
        return self.linear.parameters()
        
    async def compute(self, input) -> Dict[str, Any]:
        # Ensure input is on correct device
        x = input.to(self.device)
        
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
        elif self.activation == "leaky_relu":
            x = F.leaky_relu(x, negative_slope=0.01)
        
        # Dropout if training
        if self.dropout is not None:
            x = self.dropout(x)
        
        return {{"output": x}}
    
    def _initialize_weights(self, init_method):
        """Initialize layer weights based on method and activation function"""
        
        # If auto mode, determine best initialization based on activation
        if init_method == "auto":
            if self.activation in ["relu", "leaky_relu"]:
                init_method = "kaiming_normal"
            elif self.activation in ["tanh", "sigmoid"]:
                init_method = "xavier_normal"
            else:
                init_method = "kaiming_normal"  # Default for linear/none
            
            self.node_logger.info(f"Auto weight init: activation={self.activation}, using {init_method}")
        
        # Apply initialization
        if init_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(self.linear.weight)
        elif init_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.linear.weight)
        elif init_method == "normal":
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        elif init_method == "uniform":
            nn.init.uniform_(self.linear.weight, a=-0.1, b=0.1)
        elif init_method == "none":
            pass  # Keep PyTorch defaults
        
        # Initialize bias to zero
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
