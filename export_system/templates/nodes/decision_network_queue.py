# Template variables - replaced during export
template_vars = {
    "NODE_ID": "decision_1",
    "CLASS_NAME": "DecisionNetworkNode",
    "NUM_INPUTS": 2,
    "ACTION_DIM": 6,
    "HIDDEN_SIZE": 256,
    "DEVICE": "cuda"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Decision network that waits for all inputs before processing"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # This node waits for both vision and sound features
        self.setup_inputs(required=["vision_features", "sound_features"])
        self.setup_outputs(["action", "confidence"])
        
        # Setup device
        self.device = torch.device("{DEVICE}" if torch.cuda.is_available() else "cpu")
        
        # Build decision network
        # Assumes vision_features and sound_features are flattened vectors
        self.fusion_layer = nn.Linear(512 + 256, {HIDDEN_SIZE})  # Adjust based on actual sizes
        self.hidden_layer = nn.Linear({HIDDEN_SIZE}, {HIDDEN_SIZE})
        self.action_layer = nn.Linear({HIDDEN_SIZE}, {ACTION_DIM})
        self.confidence_layer = nn.Linear({HIDDEN_SIZE}, 1)
        
        # Move to device
        self.fusion_layer = self.fusion_layer.to(self.device)
        self.hidden_layer = self.hidden_layer.to(self.device)
        self.action_layer = self.action_layer.to(self.device)
        self.confidence_layer = self.confidence_layer.to(self.device)
        
    async def compute(self, vision_features, sound_features) -> Dict[str, Any]:
        # This method only executes when BOTH inputs are available
        # The queue framework handles the synchronization automatically
        
        # Move inputs to device
        vision = vision_features.to(self.device)
        sound = sound_features.to(self.device)
        
        # Ensure batch dimension
        if vision.dim() == 1:
            vision = vision.unsqueeze(0)
        if sound.dim() == 1:
            sound = sound.unsqueeze(0)
        
        # Fuse multimodal features
        fused = torch.cat([vision, sound], dim=1)
        
        # Forward pass
        x = F.relu(self.fusion_layer(fused))
        x = F.relu(self.hidden_layer(x))
        
        # Generate action and confidence
        action = self.action_layer(x)
        confidence = torch.sigmoid(self.confidence_layer(x))
        
        self.logger.info(f"Decision made with confidence: {{confidence.mean().item():.2f}}")
        
        return {{
            "action": action,
            "confidence": confidence
        }}
