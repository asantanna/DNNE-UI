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
    "SIM_HANDLE": "SIM_HANDLE",  # For Isaac Gym integration (legacy)
    "ENV_HANDLE": "ENV_HANDLE",  # For new Isaac Gym environment handle
    "SYNC": "SYNC",  # For node synchronization and training coordination
    "BALANCING_CONFIG": "BALANCING_CONFIG",  # For system balancing configuration
}

# ML-specific types
ML_TYPES = {
    "DATASET": "DATASET",
    "DATALOADER": "DATALOADER",
    "OPTIMIZER": "OPTIMIZER",
    "IMAGE": "IMAGE",
    "SCHEMA": "SCHEMA",
}

# RL-specific types
RL_TYPES = {
    "ISAAC_ENV_CONFIG": "ISAAC_ENV_CONFIG",
    "PPO_CONFIG": "PPO_CONFIG",
    "RL_METRICS": "RL_METRICS",
}

# Combine all custom types
ALL_CUSTOM_TYPES = {}
ALL_CUSTOM_TYPES.update(ROBOTICS_TYPES)
ALL_CUSTOM_TYPES.update(ML_TYPES)
ALL_CUSTOM_TYPES.update(RL_TYPES)

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
class RobotState(TensorData):
    """Robot joint positions, velocities, and other state information"""
    joint_positions: Optional[torch.Tensor] = None
    joint_velocities: Optional[torch.Tensor] = None
    base_pose: Optional[torch.Tensor] = None  # Position + orientation
    timestamp: float = 0.0
    
@dataclass
class SensorData:
    """Container for various sensor readings"""
    sensor_type: str  # 'camera', 'lidar', 'imu', etc.
    data: Union[torch.Tensor, np.ndarray, Dict[str, Any]]
    timestamp: float = 0.0
    frame_id: Optional[str] = None  # Coordinate frame
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class Action(TensorData):
    """Robot control commands"""
    control_mode: str = "position"  # 'position', 'velocity', 'torque'
    target_values: Optional[torch.Tensor] = None
    constraints: Optional[Dict[str, float]] = None  # min/max limits

@dataclass
class Context:
    """Global context for storing shared state across nodes"""
    memory: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    training_mode: bool = True
    device: str = "cpu"
    
    # Tracking
    episode_count: int = 0
    step_count: int = 0
    total_reward: float = 0.0
    
    def store(self, key: str, value: Any):
        """Store a value in context memory"""
        self.memory[key] = value
    
    def retrieve(self, key: str, default=None):
        """Retrieve a value from context memory"""
        return self.memory.get(key, default)
    
    def clear_episode(self):
        """Clear episode-specific data"""
        self.step_count = 0
        self.total_reward = 0.0
        # Clear episode-specific memory keys
        episode_keys = [k for k in self.memory if k.startswith("episode_")]
        for key in episode_keys:
            del self.memory[key]

# Helper function to convert between types
def tensor_to_robot_state(tensor: torch.Tensor, 
                         joint_dims: int = None) -> RobotState:
    """Convert a flat tensor to RobotState"""
    if joint_dims:
        joint_positions = tensor[:joint_dims]
        joint_velocities = tensor[joint_dims:2*joint_dims] if tensor.shape[0] > joint_dims else None
        return RobotState(
            data=tensor,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
    return RobotState(data=tensor)

def robot_state_to_tensor(state: RobotState) -> torch.Tensor:
    """Convert RobotState to flat tensor"""
    if isinstance(state.data, torch.Tensor):
        return state.data
    return state.to_torch()

# Register custom types with ComfyUI
def register_robotics_types():
    """Call this to register all custom types with ComfyUI's type system"""
    # This would integrate with ComfyUI's type registry
    # For now, it's a placeholder that documents our custom types
    print(f"Registered {len(ALL_CUSTOM_TYPES)} custom DNNE types")
    for type_name in ALL_CUSTOM_TYPES:
        print(f"  - {type_name}")