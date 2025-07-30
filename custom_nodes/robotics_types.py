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
    batch_size: Optional[int] = None
    device: str = "cpu"
    dtype: Optional[torch.dtype] = None
    
    def to(self, device: str):
        """Move tensor to specified device"""
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.to(device)
            self.device = device
        return self
    
    def __repr__(self):
        shape = self.data.shape if hasattr(self.data, 'shape') else 'unknown'
        return f"TensorData(shape={shape}, device={self.device})"

@dataclass
class RobotState:
    """Robot state information from simulation or real robot"""
    position: Union[torch.Tensor, np.ndarray]  # Joint positions or cartesian coordinates
    velocity: Optional[Union[torch.Tensor, np.ndarray]] = None
    acceleration: Optional[Union[torch.Tensor, np.ndarray]] = None
    joint_names: Optional[List[str]] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SensorData:
    """Sensor data container for various sensor types"""
    data: Union[torch.Tensor, np.ndarray, Dict[str, Any]]
    sensor_type: str  # e.g., "camera", "lidar", "force", "imu"
    timestamp: Optional[float] = None
    frame_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Action:
    """Robot action command"""
    command: Union[torch.Tensor, np.ndarray]  # Action values
    action_type: str = "joint_velocity"  # joint_velocity, joint_position, cartesian, etc.
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Context:
    """Context information for decision making"""
    task_description: Optional[str] = None
    goal_state: Optional[RobotState] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Type conversion utilities
def ensure_tensor(data: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
    """Convert data to PyTorch tensor"""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return torch.tensor(data)

def ensure_numpy(data: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    """Convert data to numpy array"""
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)