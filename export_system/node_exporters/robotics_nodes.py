#!/usr/bin/env python3
"""
Exporters for robotics nodes using queue-based templates
"""

from ..graph_exporter import ExportableNode

class CameraSensorExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/camera_sensor_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        # Parse resolution
        resolution = params.get("resolution", "640x480")
        width, height = map(int, resolution.split('x'))
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CameraSensorNode",
            "FPS": params.get("fps", 30.0),
            "WIDTH": width,
            "HEIGHT": height,
            "USE_REAL_CAMERA": params.get("use_real_camera", False),
            "CAMERA_INDEX": params.get("camera_index", 0)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import numpy as np",
            "import torch",
        ]


class IMUSensorExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/imu_sensor_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IMUSensorNode",
            "SAMPLE_RATE": params.get("sample_rate", 100.0),
            "ADD_NOISE": params.get("add_noise", True)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import numpy as np",
            "import torch",
        ]


class VisionNetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/vision_network_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "VisionNetworkNode",
            "MODEL_TYPE": params.get("model", "resnet18"),
            "PRETRAINED": params.get("pretrained", True),
            "OUTPUT_DIM": params.get("output_dim", 512),
            "DEVICE": params.get("device", "cuda")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
            "import torchvision.transforms as transforms",
        ]


class SoundNetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/sound_network_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SoundNetworkNode",
            "MODEL_TYPE": params.get("model", "wav2vec"),
            "OUTPUT_DIM": params.get("output_dim", 256),
            "DEVICE": params.get("device", "cuda")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]


class DecisionNetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/decision_network_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        # Count input connections to determine input dimension
        num_inputs = len(connections.get("inputs", {}))
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "DecisionNetworkNode",
            "NUM_INPUTS": num_inputs,
            "ACTION_DIM": params.get("action_dim", 6),
            "HIDDEN_SIZE": params.get("hidden_size", 256),
            "DEVICE": params.get("device", "cuda")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]


class RobotControllerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/robot_controller_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        # Parse joint limits
        joint_limits = params.get("joint_limits", [-3.14, 3.14])
        if isinstance(joint_limits, str):
            joint_limits = eval(joint_limits)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "RobotControllerNode",
            "JOINT_LIMITS_MIN": joint_limits[0],
            "JOINT_LIMITS_MAX": joint_limits[1],
            "CONTROL_TYPE": params.get("control_type", "position"),
            "NUM_JOINTS": params.get("num_joints", 7)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import numpy as np",
            "import torch",
        ]


class IsaacGymEnvExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/isaac_gym_env_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IsaacGymEnvNode",
            "ENV_NAME": params.get("env_name", "Cartpole"),
            "NUM_ENVS": params.get("num_envs", 1),
            "DEVICE": params.get("device", "cuda"),
            "HEADLESS": params.get("headless", False)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import numpy as np",
            "# from isaacgym import gymapi  # Uncomment when Isaac Gym is available",
        ]


# Registration function
def register_robotics_exporters(exporter):
    """Register all robotics node exporters"""
    # Sensors
    exporter.register_node("CameraSensor", CameraSensorExporter)
    exporter.register_node("IMUSensor", IMUSensorExporter)
    exporter.register_node("AudioSensor", CameraSensorExporter)  # Reuse camera template
    
    # Processing networks
    exporter.register_node("VisionNetwork", VisionNetworkExporter)
    exporter.register_node("SoundNetwork", SoundNetworkExporter)
    exporter.register_node("DecisionNetwork", DecisionNetworkExporter)
    
    # Control
    exporter.register_node("RobotController", RobotControllerExporter)
    
    # Isaac Gym
    exporter.register_node("IsaacGymEnv", IsaacGymEnvExporter)
