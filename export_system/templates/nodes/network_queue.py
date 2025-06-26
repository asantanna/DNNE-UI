# Template variables - replaced during export

class NetworkNode_{NODE_ID}(QueueNode):
    """Neural network with multiple layers"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input"])
        self.setup_outputs(["layers", "network_output"])
        
        # Build network from detected layers: {NETWORK_LAYERS}
        layers = []
        {LAYER_DEFINITIONS}
        
        self.network = nn.Sequential(*layers)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self.network.to(self.device)
        
        self.logger.info(f"Created network with {NUM_LAYERS} layers: {INPUT_SIZE} -> {OUTPUT_SIZE}")
        
    def get_parameters(self):
        """Return network parameters for optimizer"""
        return self.network.parameters()
        
    async def compute(self, input) -> Dict[str, Any]:
        # Ensure input is on correct device
        x = input.to(self.device)
        
        # Flatten if needed (for MNIST)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass through the entire network
        output = self.network(x)
        
        return {
            "layers": None,  # This output is just for UI connectivity
            "network_output": output
        }
