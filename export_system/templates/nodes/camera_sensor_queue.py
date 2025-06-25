# Template variables - replaced during export
template_vars = {
    "NODE_ID": "camera_1",
    "CLASS_NAME": "CameraSensorNode",
    "FPS": 30.0,
    "WIDTH": 640,
    "HEIGHT": 480,
    "USE_REAL_CAMERA": False,
    "CAMERA_INDEX": 0
}

class {CLASS_NAME}_{NODE_ID}(SensorNode):
    """Camera sensor that generates image data at fixed FPS"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, update_rate={FPS})
        self.setup_outputs(["image", "timestamp"])
        
        self.width = {WIDTH}
        self.height = {HEIGHT}
        self.channels = 3
        self.frame_count = 0
        self.use_real_camera = {USE_REAL_CAMERA}
        
        if self.use_real_camera:
            import cv2
            self.camera = cv2.VideoCapture({CAMERA_INDEX})
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, {FPS})
        
    async def compute(self) -> Dict[str, Any]:
        if self.use_real_camera:
            # Read from real camera
            ret, frame = self.camera.read()
            if ret:
                # Convert BGR to RGB and normalize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            else:
                # Fallback to random if camera fails
                image = torch.rand(self.channels, self.height, self.width)
        else:
            # Simulated camera data for testing
            image = torch.rand(self.channels, self.height, self.width)
        
        timestamp = time.time()
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            self.logger.info(f"Captured frame {{self.frame_count}}")
        
        return {{
            "image": image,
            "timestamp": timestamp
        }}
    
    def __del__(self):
        """Cleanup camera resources"""
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
