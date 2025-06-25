# Template variables - replaced during export
template_vars = {
    "NODE_ID": "vision_1",
    "CLASS_NAME": "VisionNetworkNode",
    "MODEL_TYPE": "resnet18",
    "PRETRAINED": True,
    "OUTPUT_DIM": 512,
    "DEVICE": "cuda"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Vision processing network"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["camera_data"])
        self.setup_outputs(["vision_features"])
        
        # Setup device
        self.device = torch.device("{DEVICE}" if torch.cuda.is_available() else "cpu")
        
        # Load vision model
        if "{MODEL_TYPE}" == "resnet18":
            from torchvision.models import resnet18
            self.model = resnet18(pretrained={PRETRAINED})
            # Remove final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif "{MODEL_TYPE}" == "mobilenet":
            from torchvision.models import mobilenet_v2
            self.model = mobilenet_v2(pretrained={PRETRAINED})
            self.model.classifier = nn.Identity()
        else:
            # Custom model placeholder
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, {OUTPUT_DIM})
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to eval mode
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    async def compute(self, camera_data) -> Dict[str, Any]:
        # Move to device
        image = camera_data.to(self.device)
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Normalize
        image = self.transform(image)
        
        # Forward pass
        with torch.no_grad():
            features = self.model(image)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        return {{"vision_features": features}}
