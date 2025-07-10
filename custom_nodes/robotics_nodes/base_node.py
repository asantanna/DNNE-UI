# robotics_nodes/base_node.py
"""
Base class for all robotics nodes in DNNE
Provides common functionality and standards for robotics modules
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
import numpy as np
from .robotics_types import *

class RoboticsNodeBase:
    """Base class for all robotics nodes"""
    
    # Default category for organization in UI
    CATEGORY = "robotics"
    
    # Common color coding for different node types (optional)
    NODE_COLORS = {
        "sensor": "#4A90E2",      # Blue for sensors
        "controller": "#50E3C2",   # Teal for controllers
        "actuator": "#F5A623",     # Orange for actuators
        "utility": "#7ED321",      # Green for utilities
        "simulation": "#BD10E0",   # Purple for sim nodes
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        """Override this in subclasses to define inputs"""
        return {"required": {}, "optional": {}}
    
    # Default return types (override in subclasses)
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    
    # Function name that will be called (override if needed)
    FUNCTION = "compute"
    
    # For nodes that output data to UI
    OUTPUT_NODE = False
    
    def compute(self, **kwargs):
        """Override this in subclasses to implement node logic"""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement compute() method"
        )
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Called to check if node needs to be re-executed.
        Used for nodes with external state or randomness.
        """
        # By default, only re-execute if inputs change
        # Override for time-dependent or random nodes
        return False
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> bool:
        """
        Validate input connections are compatible.
        Override for custom validation logic.
        """
        return True
    
    # Helper methods for common robotics operations
    
    def ensure_tensor(self, data: Any) -> torch.Tensor:
        """Convert various inputs to torch tensor"""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, (list, tuple)):
            return torch.tensor(data)
        elif isinstance(data, TensorData):
            return data.to_torch()
        else:
            raise ValueError(f"Cannot convert {type(data)} to tensor")
    
    def ensure_batch(self, tensor: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """Ensure tensor has batch dimension"""
        if tensor.dim() == 1 and batch_size is not None:
            return tensor.unsqueeze(0).expand(batch_size, -1)
        return tensor
    
    def get_device(self, *tensors) -> torch.device:
        """Get device from input tensors"""
        for t in tensors:
            if isinstance(t, torch.Tensor):
                return t.device
        return torch.device("cpu")


class SensorNodeBase(RoboticsNodeBase):
    """Base class for sensor nodes"""
    CATEGORY = "utils"
    
    def __init__(self):
        super().__init__()
        self.sensor_type = "generic"
        self.noise_enabled = False
    
    def add_noise(self, data: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to sensor data"""
        if self.noise_enabled and noise_std > 0:
            noise = torch.randn_like(data) * noise_std
            return data + noise
        return data


class ControllerNodeBase(RoboticsNodeBase):
    """Base class for controller nodes"""
    CATEGORY = "utils"
    
    def __init__(self):
        super().__init__()
        self.control_mode = "position"  # position, velocity, torque
        self.safety_limits_enabled = True
    
    def apply_safety_limits(self, commands: torch.Tensor, 
                          limits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply safety limits to control commands"""
        if not self.safety_limits_enabled or limits is None:
            return commands
        
        return torch.clamp(commands, -limits, limits)


class LearningNodeBase(RoboticsNodeBase):
    """Base class for learning/neural network nodes"""
    CATEGORY = "utils"
    
    def __init__(self):
        super().__init__()
        self.training_mode = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def save_checkpoint(self, path: str):
        """Save node state/weights"""
        # Override in subclasses
        pass
    
    def load_checkpoint(self, path: str):
        """Load node state/weights"""
        # Override in subclasses
        pass


class VisualizationNodeBase(RoboticsNodeBase):
    """Base class for visualization/display nodes"""
    CATEGORY = "utils"
    OUTPUT_NODE = True  # These nodes output to UI
    
    def __init__(self):
        super().__init__()
        self.update_rate = 10  # Hz
        self.last_update_time = 0
    
    def should_update(self, current_time: float) -> bool:
        """Check if visualization should update based on rate limit"""
        if current_time - self.last_update_time >= 1.0 / self.update_rate:
            self.last_update_time = current_time
            return True
        return False


# Base classes are for inheritance only - no example nodes or registration here