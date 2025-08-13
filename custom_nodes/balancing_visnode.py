"""
Balancing Node
Active passthrough node that measures and enforces performance targets
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class BalancingNode(RoboticsNodeBase):
    """
    Balancing Node
    
    A passthrough node that measures and enforces performance targets while
    forwarding data unchanged. Insert at strategic points in workflows to
    monitor and control execution rates.
    
    Features:
    - Measures throughput, frequency, and latency
    - Enforces min/max frequency limits
    - Reports metrics to adaptive yielding system
    - Minimal overhead (just timestamps and forwards data)
    
    Configuration parameters:
    - Frequency-based targets: min_hz, max_hz, target_hz
    - Throughput-based targets: target_percentage
    - Priority settings: priority, guaranteed
    - Latency requirements: max_latency_ms
    """
    
    # Active node - participates in execution
    IS_VIRTUAL = False
    
    CATEGORY = "utility"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {
                    "tooltip": "Any data to passthrough while monitoring performance"
                }),
            },
            "optional": {
                # Item name for metrics display
                "item_name": ("STRING", {
                    "default": "items",
                    "tooltip": "Unit name for throughput metrics (e.g., 'batches', 'frames', 'steps')"
                }),
                
                # Enable/disable monitoring
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable performance monitoring"
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
                
                # Measurement settings
                "window_size": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "tooltip": "Number of samples for moving average"
                }),
                "log_violations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Log when performance targets are violated"
                }),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("balancing")["color"]
    BGCOLOR = get_node_colors("balancing")["bgcolor"]

# Node registration
NODE_CLASS_MAPPINGS = {
    "BalancingNode": BalancingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BalancingNode": "Balancing Node"
}