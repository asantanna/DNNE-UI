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
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'resolution', 'default': '640x480'},
            {'name': 'fps', 'default': 30.0},
            {'name': 'use_real_camera', 'default': False},
            {'name': 'camera_index', 'default': 0}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Parse resolution
        resolution = params['resolution']
        width, height = map(int, resolution.split('x'))
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CameraSensorNode",
            "FPS": params['fps'],
            "WIDTH": width,
            "HEIGHT": height,
            "USE_REAL_CAMERA": params['use_real_camera'],
            "CAMERA_INDEX": params['camera_index']
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
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'sample_rate', 'default': 100.0},
            {'name': 'add_noise', 'default': True}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IMUSensorNode",
            "SAMPLE_RATE": params['sample_rate'],
            "ADD_NOISE": params['add_noise']
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
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'model', 'default': 'resnet18'},
            {'name': 'pretrained', 'default': True},
            {'name': 'output_dim', 'default': 512},
            {'name': 'device', 'default': 'cuda'}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "VisionNetworkNode",
            "MODEL_TYPE": params['model'],
            "PRETRAINED": params['pretrained'],
            "OUTPUT_DIM": params['output_dim'],
            "DEVICE": params['device']
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
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'model', 'default': 'wav2vec'},
            {'name': 'output_dim', 'default': 256},
            {'name': 'device', 'default': 'cuda'}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SoundNetworkNode",
            "MODEL_TYPE": params['model'],
            "OUTPUT_DIM": params['output_dim'],
            "DEVICE": params['device']
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
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
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
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
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
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'env_name', 'default': 'Cartpole'},
            {'name': 'num_envs', 'default': 512},
            {'name': 'isaac_gym_path', 'default': '/home/asantanna/DNNE-LINUX-SUPPORT/isaacgym'},
            {'name': 'isaac_gym_envs_path', 'default': '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs'},
            {'name': 'headless', 'default': True},
            {'name': 'device', 'default': 'cuda'},
            {'name': 'physics_engine', 'default': 'physx'}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Extract parameters
        env_name = params['env_name']
        num_envs = params['num_envs']
        isaac_gym_path = params['isaac_gym_path']
        isaac_gym_envs_path = params['isaac_gym_envs_path']
        headless = params['headless']
        device = params['device']
        physics_engine = params['physics_engine']
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IsaacGymEnvNode",
            "ENV_NAME": env_name,
            "NUM_ENVS": num_envs,
            "ISAAC_GYM_PATH": isaac_gym_path,
            "ISAAC_GYM_ENVS_PATH": isaac_gym_envs_path,
            "HEADLESS": headless,
            "DEVICE": device,
            "PHYSICS_ENGINE": physics_engine
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import isaacgym",  # MUST BE FIRST - before torch
            "import torch",
            "import numpy as np",
            "import os"
        ]
    
    @classmethod
    def get_input_names(cls):
        return []  # No inputs - runs at startup
    
    @classmethod
    def get_output_names(cls):
        return ["sim_handle", "observations"]


class IsaacGymStepExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/isaac_gym_step_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # ComfyUI workflow format uses widgets_values list
        widget_values = node_data.get("widgets_values", [])
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IsaacGymStepNode"
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import numpy as np",
            "# Isaac Gym imports are handled at runtime in the template",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["sim_handle", "actions", "trigger"]
    
    @classmethod
    def get_output_names(cls):
        return ["observations", "rewards", "done", "info", "next_observations"]


class ORNodeExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/or_node_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "ORNode"
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["input_a", "input_b", "input_c"]
    
    @classmethod
    def get_output_names(cls):
        return ["output"]


class CartpoleActionNodeExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/cartpole_action_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # ComfyUI workflow format uses widgets_values list
        widget_values = node_data.get("widgets_values", [10.0])
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CartpoleActionNode",
            "MAX_PUSH_EFFORT": widget_values[0] if len(widget_values) > 0 else 10.0
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["policy"]
    
    @classmethod
    def get_output_names(cls):
        return ["action"]


class CartpoleRewardNodeExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/cartpole_reward_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # ComfyUI workflow format uses widgets_values list
        widget_values = node_data.get("widgets_values", [2.0, True])
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CartpoleRewardNode",
            "RESET_DIST": widget_values[0] if len(widget_values) > 0 else 2.0,
            "INVERT_FOR_LOSS": widget_values[1] if len(widget_values) > 1 else True
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import numpy as np",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["observations"]
    
    @classmethod
    def get_output_names(cls):
        return ["reward_or_loss", "done", "info_dict"]


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
    exporter.register_node("IsaacGymEnvNode", IsaacGymEnvExporter)
    exporter.register_node("IsaacGymStep", IsaacGymStepExporter)
    exporter.register_node("IsaacGymStepNode", IsaacGymStepExporter)
    
    # Utility nodes
    exporter.register_node("ORNode", ORNodeExporter)
    
    # Cartpole RL nodes
    exporter.register_node("CartpoleActionNode", CartpoleActionNodeExporter)
    exporter.register_node("CartpoleRewardNode", CartpoleRewardNodeExporter)
