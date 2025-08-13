"""
Balancing Configuration Node (Virtual)
Configuration-only node for setting performance targets on monolithic nodes like PPO_Agent
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class BalancingConfig(RoboticsNodeBase):
    """
    Balancing Configuration Node (Virtual)
    
    This is a virtual configuration node that provides performance targets to
    connected nodes (like PPO_Agent) without generating runtime code itself.
    
    Configuration parameters:
    - Frequency-based targets: min_hz, max_hz, target_hz
    - Throughput-based targets: target_percentage
    - Priority settings: priority, guaranteed
    - Latency requirements: max_latency_ms
    """
    
    # Virtual node - doesn't generate runtime code
    IS_VIRTUAL = True
    
    CATEGORY = "utility"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Enable/disable configuration
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable this configuration"
                }),
                
                # Frequency-based targets (robotics/real-time)
                "min_hz": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Minimum frequency in Hz (-1 = don't care)"
                }),
                "max_hz": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Maximum frequency in Hz (-1 = don't care)"
                }),
                "target_hz": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Target frequency in Hz (-1 = don't care)"
                }),
                
                # Throughput-based targets (batch processing)
                "target_percentage": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Target percentage of total system throughput (-1 = don't care)"
                }),
                
                # Priority settings
                "priority": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Priority level (higher = more important)"
                }),
                "guaranteed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Must meet targets vs best-effort"
                }),
                
                # Latency requirements
                "max_latency_ms": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 10000.0,
                    "step": 0.1,
                    "tooltip": "Maximum processing latency in milliseconds (-1 = don't care)"
                }),
                
            }
        }
    
    RETURN_TYPES = ("BALANCING_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("balancing")["color"]
    BGCOLOR = get_node_colors("balancing")["bgcolor"]


# Node registration
NODE_CLASS_MAPPINGS = {
    "BalancingConfig": BalancingConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BalancingConfig": "Balancing Config"
}