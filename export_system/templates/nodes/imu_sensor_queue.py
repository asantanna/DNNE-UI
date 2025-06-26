# Template variables - replaced during export
template_vars = {
    "NODE_ID": "imu_1",
    "CLASS_NAME": "IMUSensorNode",
    "SAMPLE_RATE": 100.0,
    "ADD_NOISE": True
}

class {CLASS_NAME}_{NODE_ID}(SensorNode):
    """IMU sensor that generates acceleration and gyro data"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, update_rate={SAMPLE_RATE})
        self.setup_outputs(["acceleration", "angular_velocity", "orientation"])
        
        self.sample_count = 0
        self.add_noise = {ADD_NOISE}
        
    async def compute(self) -> Dict[str, Any]:
        import random
        
        # Simulate IMU data without PyTorch
        if self.add_noise:
            # Gravity + noise
            acceleration = [
                random.gauss(0.0, 0.1),
                random.gauss(0.0, 0.1),
                9.81 + random.gauss(0.0, 0.1)
            ]
            angular_velocity = [
                random.gauss(0.0, 0.01),
                random.gauss(0.0, 0.01),
                random.gauss(0.0, 0.01)
            ]
        else:
            # Clean data
            acceleration = [0.0, 0.0, 9.81]
            angular_velocity = [0.0, 0.0, 0.0]
        
        # Simple orientation quaternion (identity = no rotation)
        orientation = [1.0, 0.0, 0.0, 0.0]
        
        self.sample_count += 1
        
        if self.sample_count % 100 == 0:
            self.logger.info(f"IMU sample {{self.sample_count}}")
        
        return {{
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "orientation": orientation
        }}
