# robotics_types.py
"""
Core data type definitions for DNNE (Drag-and-Drop Neural Network Environment)
These types replace ComfyUI's image-generation types with robotics-specific ones.
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass, field

# Register these as valid ComfyUI types
ROBOTICS_TYPES = {
    "TENSOR": "TENSOR",
    "ROBOT_STATE": "ROBOT_STATE", 
    "SENSOR_DATA": "SENSOR_DATA",
    "ACTION": "ACTION",
    "CONTEXT": "CONTEXT",
    "SIM_HANDLE": "SIM_HANDLE",  # For Isaac Gym integration
    "SYNC": "SYNC",  # For node synchronization and training coordination
}

# ML-specific types
ML_TYPES = {
    "DATASET": "DATASET",
    "DATALOADER": "DATALOADER",
    "OPTIMIZER": "OPTIMIZER",
    "IMAGE": "IMAGE",
    "SCHEMA": "SCHEMA",
}

# Combine all custom types
ALL_CUSTOM_TYPES = {}
ALL_CUSTOM_TYPES.update(ROBOTICS_TYPES)
ALL_CUSTOM_TYPES.update(ML_TYPES)

@dataclass
class TensorData:
    """Generic tensor container for neural network data"""
    data: Union[torch.Tensor, np.ndarray]
    dtype: str = "float32"
    device: str = "cpu"
    shape_info: Optional[str] = None  # Human-readable shape description
    
    def to_torch(self) -> torch.Tensor:
        if isinstance(self.data, torch.Tensor):
            return self.data
        return torch.from_numpy(self.data)
    
    def to_numpy(self) -> np.ndarray:
        if isinstance(self.data, np.ndarray):
            return self.data
        return self.data.cpu().numpy()

@dataclass
class RobotState:
    """Complete state information for a robot"""
    joint_positions: Optional[torch.Tensor] = None  # [n_joints] or [batch, n_joints]
    joint_velocities: Optional[torch.Tensor] = None
    joint_efforts: Optional[torch.Tensor] = None
    base_position: Optional[torch.Tensor] = None  # [3] or [batch, 3] for x,y,z
    base_orientation: Optional[torch.Tensor] = None  # [4] or [batch, 4] for quaternion
    base_linear_velocity: Optional[torch.Tensor] = None  # [3] or [batch, 3]
    base_angular_velocity: Optional[torch.Tensor] = None  # [3] or [batch, 3]
    
    # Metadata
    joint_names: Optional[List[str]] = None
    timestamp: Optional[float] = None
    frame_id: Optional[str] = None  # Reference frame
    
    def get_dim(self) -> int:
        """Get the dimensionality of the state vector"""
        dim = 0
        if self.joint_positions is not None:
            dim += self.joint_positions.shape[-1]
        if self.joint_velocities is not None:
            dim += self.joint_velocities.shape[-1]
        # Add other components as needed
        return dim
    
    def to_tensor(self) -> torch.Tensor:
        """Flatten to a single tensor for neural network input"""
        components = []
        if self.joint_positions is not None:
            components.append(self.joint_positions)
        if self.joint_velocities is not None:
            components.append(self.joint_velocities)
        if self.base_position is not None:
            components.append(self.base_position)
        if self.base_orientation is not None:
            components.append(self.base_orientation)
        
        if components:
            return torch.cat(components, dim=-1)
        return torch.tensor([])

@dataclass
class SensorData:
    """Container for various sensor readings"""
    sensor_type: str  # "camera", "lidar", "imu", "force_torque", etc.
    data: Union[torch.Tensor, np.ndarray, Dict[str, Any]]
    
    # Common sensor metadata
    timestamp: Optional[float] = None
    frame_id: Optional[str] = None
    
    # Camera-specific
    image_format: Optional[str] = None  # "RGB", "depth", "RGBD"
    resolution: Optional[tuple] = None  # (height, width)
    fov: Optional[float] = None  # Field of view
    
    # IMU-specific
    linear_acceleration: Optional[torch.Tensor] = None
    angular_velocity: Optional[torch.Tensor] = None
    orientation: Optional[torch.Tensor] = None
    
    # Force/Torque sensor
    force: Optional[torch.Tensor] = None  # [3] or [batch, 3]
    torque: Optional[torch.Tensor] = None  # [3] or [batch, 3]
    
    def is_image(self) -> bool:
        return self.sensor_type in ["camera", "rgbd", "depth"]
    
    def is_imu(self) -> bool:
        return self.sensor_type == "imu"

@dataclass
class Action:
    """Robot action/control commands"""
    control_mode: str  # "position", "velocity", "torque", "hybrid"
    
    # Joint commands
    joint_commands: Optional[torch.Tensor] = None  # [n_joints] or [batch, n_joints]
    joint_names: Optional[List[str]] = None
    
    # Base commands (for mobile robots)
    base_linear_velocity: Optional[torch.Tensor] = None  # [3] or [batch, 3]
    base_angular_velocity: Optional[torch.Tensor] = None  # [3] or [batch, 3]
    
    # Gripper/end-effector commands
    gripper_command: Optional[float] = None  # 0.0 to 1.0
    
    # Safety limits
    max_joint_velocity: Optional[torch.Tensor] = None
    max_joint_torque: Optional[torch.Tensor] = None
    
    # Metadata
    timestamp: Optional[float] = None
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for sending to simulator"""
        if self.joint_commands is not None:
            return self.joint_commands
        # Handle other action types
        return torch.tensor([])

@dataclass 
class Context:
    """Shared memory/context for stateful modules"""
    memory: Dict[str, Any] = field(default_factory=dict)
    hidden_states: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Episode/trial information
    episode_count: int = 0
    step_count: int = 0
    total_reward: float = 0.0
    
    # Module communication
    messages: Dict[str, Any] = field(default_factory=dict)
    
    # Training/inference mode
    training: bool = True
    
    def store(self, key: str, value: Any):
        """Store a value in context"""
        self.memory[key] = value
    
    def retrieve(self, key: str, default=None) -> Any:
        """Retrieve a value from context"""
        return self.memory.get(key, default)
    
    def update_hidden(self, module_id: str, hidden: torch.Tensor):
        """Update hidden state for a specific module"""
        self.hidden_states[module_id] = hidden
    
    def get_hidden(self, module_id: str) -> Optional[torch.Tensor]:
        """Get hidden state for a specific module"""
        return self.hidden_states.get(module_id)
    
    def reset(self):
        """Reset context for new episode"""
        self.hidden_states.clear()
        self.messages.clear()
        self.step_count = 0

@dataclass
class SimHandle:
    """Handle to Isaac Gym simulator instance"""
    sim: Any  # Isaac Gym sim object
    envs: List[Any] = field(default_factory=list)  # Environment handles
    viewer: Optional[Any] = None  # Viewer handle if using GUI
    
    # Simulation parameters
    dt: float = 0.01667  # 60 Hz default
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    
    # Device info
    device: str = "cuda:0"
    graphics_device: int = 0
    
    def is_valid(self) -> bool:
        return self.sim is not None

@dataclass
class SyncSignal:
    """Synchronization signal for coordinating node execution"""
    signal_type: str  # "start", "ready", "complete", "epoch_complete", etc.
    timestamp: float
    source_node: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_completion_signal(self) -> bool:
        """Check if this is a completion/ready signal"""
        return self.signal_type in ["ready", "complete", "epoch_complete"]
    
    def __repr__(self) -> str:
        return f"SyncSignal({self.signal_type}, {self.source_node})"

# Type validation functions for ComfyUI integration
def validate_tensor_connection(output_type: str, input_type: str) -> bool:
    """Check if tensor types are compatible for connection"""
    # TENSOR can connect to any robotics type
    if output_type == "TENSOR" or input_type == "TENSOR":
        return True
    
    # Same types can always connect
    if output_type == input_type:
        return True
    
    # Define allowed conversions
    allowed_conversions = {
        ("ROBOT_STATE", "TENSOR"),
        ("SENSOR_DATA", "TENSOR"),
        ("ACTION", "TENSOR"),
        ("TENSOR", "ACTION"),
        ("TENSOR", "ROBOT_STATE"),
        ("SYNC", "SYNC"),  # SYNC can only connect to SYNC
    }
    
    return (output_type, input_type) in allowed_conversions

# Helper functions for type conversion
def robot_state_to_tensor(state: RobotState) -> TensorData:
    """Convert RobotState to generic TensorData"""
    tensor = state.to_tensor()
    return TensorData(
        data=tensor,
        shape_info=f"Robot state: {tensor.shape}"
    )

def tensor_to_action(tensor_data: TensorData, control_mode: str = "position") -> Action:
    """Convert generic tensor to Action"""
    tensor = tensor_data.to_torch()
    return Action(
        control_mode=control_mode,
        joint_commands=tensor
    )

def combine_sensor_data(sensors: List[SensorData]) -> TensorData:
    """Combine multiple sensors into single tensor"""
    tensors = []
    for sensor in sensors:
        if isinstance(sensor.data, (torch.Tensor, np.ndarray)):
            tensor = sensor.data if isinstance(sensor.data, torch.Tensor) else torch.from_numpy(sensor.data)
            tensors.append(tensor.flatten())
    
    combined = torch.cat(tensors) if tensors else torch.tensor([])
    return TensorData(
        data=combined,
        shape_info=f"Combined sensors: {combined.shape}"
    )

# Register types with ComfyUI (called during initialization)
def register_robotics_types():
    """Register our custom types with ComfyUI's type system"""
    # This would be called in your __init__.py or main initialization
    # The exact implementation depends on ComfyUI's type registration system
    
    # Example registration (adjust based on ComfyUI's actual API):
    for type_name in ALL_CUSTOM_TYPES.values():
        # Register type for validation
        # ComfyUI specific registration code here
        pass
    
    print(f"Registered custom types: {list(ALL_CUSTOM_TYPES.values())}")

# Example usage in a node
class ExampleRoboticsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "robot_state": ("ROBOT_STATE",),
                "sensor": ("SENSOR_DATA",),
            },
            "optional": {
                "context": ("CONTEXT",),
            }
        }
    
    RETURN_TYPES = ("ACTION", "CONTEXT")
    FUNCTION = "compute"
    CATEGORY = "robotics/control"
    
    def compute(self, robot_state, sensor, context=None):
        # Your control logic here
        action = Action(
            control_mode="position",
            joint_commands=torch.zeros_like(robot_state.joint_positions)
        )
        
        if context is None:
            context = Context()
        
        context.step_count += 1
        
        return (action, context)