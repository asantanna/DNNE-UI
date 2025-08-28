"""
Base classes and utilities for all DNNE nodes
Combines base classes from robotics, ML, and other node types
"""

from typing import Dict, Any, Optional, List
import torch
import numpy as np

# Import robotics types from the same directory
from .robotics_types import TensorData, Context


class OutputDictMixin:
    """Mixin to auto-generate RETURN_TYPES and RETURN_NAMES from OUTPUT_DICT"""
    
    OUTPUT_DICT: Dict[int, Dict[str, Any]] = {}
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Auto-generate RETURN_TYPES and RETURN_NAMES from OUTPUT_DICT if it exists
        if hasattr(cls, 'OUTPUT_DICT') and cls.OUTPUT_DICT:
            # Sort by index to ensure consistent ordering
            sorted_outputs = sorted(cls.OUTPUT_DICT.items())
            
            # Generate RETURN_TYPES tuple
            cls.RETURN_TYPES = tuple(out['type'] for idx, out in sorted_outputs)
            
            # Generate RETURN_NAMES tuple
            cls.RETURN_NAMES = tuple(out['name'] for idx, out in sorted_outputs)


class RoboticsNodeBase(OutputDictMixin):
    """Base class for all robotics nodes"""
    
    # Default category for organization in UI
    CATEGORY = "robotics"
    
    # Virtual node flag - set to True for configuration-only nodes
    # Virtual nodes are skipped during export and only provide configuration to non-virtual nodes
    IS_VIRTUAL = False
    
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
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement save_checkpoint() method"
        )
    
    def load_checkpoint(self, path: str):
        """Load node state/weights"""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement load_checkpoint() method"
        )


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


# Global context instance for ML nodes
_context = None

def get_context():
    """Get or create global context"""
    global _context
    if _context is None:
        _context = Context()
    return _context


# Export all base classes
__all__ = [
    'OutputDictMixin',
    'RoboticsNodeBase',
    'SensorNodeBase',
    'ControllerNodeBase',
    'LearningNodeBase',
    'VisualizationNodeBase',
    'get_context'
]