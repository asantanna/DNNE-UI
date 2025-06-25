#!/usr/bin/env python3
"""
Create missing template files
"""

import os
from pathlib import Path

templates = {
    "export_system/templates/nodes/imu_sensor_queue.py": '''# Template variables - replaced during export
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
''',

    "export_system/templates/nodes/sound_network_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "sound_1",
    "CLASS_NAME": "SoundNetworkNode",
    "MODEL_TYPE": "wav2vec",
    "OUTPUT_DIM": 256,
    "DEVICE": "cuda"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Sound processing network (placeholder without ML dependencies)"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["audio_data"])
        self.setup_outputs(["sound_features"])
        
        self.model_type = "{MODEL_TYPE}"
        self.output_dim = {OUTPUT_DIM}
        self.logger.info(f"SoundNetwork initialized (placeholder mode)")
        
    async def compute(self, audio_data) -> Dict[str, Any]:
        # Placeholder: generate random features
        import random
        
        # Simulate feature extraction
        features = [random.random() for _ in range(self.output_dim)]
        
        self.logger.debug(f"Processed audio data")
        
        return {{"sound_features": features}}
''',

    "export_system/templates/nodes/robot_controller_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "robot_1",
    "CLASS_NAME": "RobotControllerNode",
    "JOINT_LIMITS_MIN": -3.14,
    "JOINT_LIMITS_MAX": 3.14,
    "CONTROL_TYPE": "position",
    "NUM_JOINTS": 7
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Robot controller that converts actions to joint commands"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["action"])
        self.setup_outputs(["joint_commands", "status"])
        
        self.num_joints = {NUM_JOINTS}
        self.joint_limits = ({JOINT_LIMITS_MIN}, {JOINT_LIMITS_MAX})
        self.control_type = "{CONTROL_TYPE}"
        
    async def compute(self, action) -> Dict[str, Any]:
        # Convert action to joint commands
        # Placeholder implementation
        
        # Ensure action is the right size
        if hasattr(action, '__len__'):
            if len(action) != self.num_joints:
                self.logger.warning(f"Action size {{len(action)}} != num_joints {{self.num_joints}}")
        
        # Clip to joint limits
        joint_commands = []
        for i in range(self.num_joints):
            if hasattr(action, '__getitem__'):
                cmd = max(self.joint_limits[0], min(self.joint_limits[1], float(action[i])))
            else:
                cmd = 0.0  # Default position
            joint_commands.append(cmd)
        
        status = f"Sent {{self.control_type}} commands to {{self.num_joints}} joints"
        
        return {{
            "joint_commands": joint_commands,
            "status": status
        }}
''',

    # Add optimizer template too
    "export_system/templates/nodes/optimizer_queue.py": '''# Template variables - replaced during export
template_vars = {
    "NODE_ID": "optimizer_1",
    "CLASS_NAME": "OptimizerNode",
    "OPTIMIZER_TYPE": "adam",
    "LEARNING_RATE": 0.001,
    "MODEL_NODES": []
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Optimizer node (placeholder for non-ML testing)"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["loss"])
        self.setup_outputs(["step_complete"])
        
        self.optimizer_type = "{OPTIMIZER_TYPE}"
        self.learning_rate = {LEARNING_RATE}
        self.step_count = 0
        
    async def compute(self, loss) -> Dict[str, Any]:
        # Placeholder optimization
        self.step_count += 1
        
        if self.step_count % 10 == 0:
            self.logger.info(f"Optimization step {{self.step_count}}")
        
        return {{"step_complete": True}}
'''
}

def create_templates():
    created = 0
    for filepath, content in templates.items():
        path = Path(filepath)
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        path.write_text(content, encoding='utf-8')
        print(f"✅ Created: {filepath}")
        created += 1
    
    print(f"\n✅ Created {created} template files")

if __name__ == "__main__":
    print("Creating missing template files...")
    create_templates()
