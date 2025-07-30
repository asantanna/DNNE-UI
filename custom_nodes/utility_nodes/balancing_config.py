# balancing_config.py
"""
Balancing Configuration Node (Virtual)
Configuration-only node for setting performance targets on monolithic nodes like PPO_Agent
"""

from typing import Dict, Any, Optional
from inspect import cleandoc

# Import base node from robotics nodes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from robotics_nodes.base_node import RoboticsNodeBase
from custom_nodes.node_colors import get_node_colors


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
    FUNCTION = "create_config"
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("balancing")["color"]
    BGCOLOR = get_node_colors("balancing")["bgcolor"]
    
    def create_config(self, enabled=True, min_hz=-1.0, max_hz=-1.0, target_hz=-1.0,
                     target_percentage=-1.0, priority=0, guaranteed=False,
                     max_latency_ms=-1.0) -> tuple:
        """Create balancing configuration dictionary"""
        
        # If disabled, return empty config
        if not enabled:
            return ({"type": "balancing_config", "enabled": False},)
        
        # Build configuration dictionary
        config = {
            "frequency": {
                "min_hz": min_hz if min_hz >= 0 else None,
                "max_hz": max_hz if max_hz >= 0 else None,
                "target_hz": target_hz if target_hz >= 0 else None,
            },
            "throughput": {
                "target_percentage": target_percentage if target_percentage >= 0 else None,
            },
            "scheduling": {
                "priority": priority,
                "guaranteed": guaranteed,
            },
            "latency": {
                "max_latency_ms": max_latency_ms if max_latency_ms >= 0 else None,
            },
            "type": "balancing_config",
            "enabled": True
        }
        
        # Remove empty sub-dictionaries
        config = self._clean_config(config)
        
        # Validate configuration
        validation_msg = self._validate_config(config)
        if validation_msg:
            print(f"⚠️  Balancing Config Warning: {validation_msg}")
        
        # Log configuration
        print(f"✓ Balancing Config created:")
        if config.get("frequency"):
            freq = config["frequency"]
            if freq.get("target_hz"):
                print(f"  - Target: {freq['target_hz']} Hz")
            if freq.get("min_hz") or freq.get("max_hz"):
                print(f"  - Range: {freq.get('min_hz', 'any')} - {freq.get('max_hz', 'any')} Hz")
        if config.get("throughput", {}).get("target_percentage"):
            print(f"  - Throughput: {config['throughput']['target_percentage']}%")
        if config.get("scheduling", {}).get("priority", 0) > 0:
            print(f"  - Priority: {config['scheduling']['priority']}")
        if config.get("scheduling", {}).get("guaranteed"):
            print(f"  - Guaranteed execution")
        if config.get("latency", {}).get("max_latency_ms"):
            print(f"  - Max latency: {config['latency']['max_latency_ms']} ms")
        
        return (config,)
    
    def _clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove empty sub-dictionaries from config"""
        cleaned = {}
        for key, value in config.items():
            if isinstance(value, dict):
                # Remove None values from sub-dict
                sub_dict = {k: v for k, v in value.items() if v is not None}
                if sub_dict:  # Only include if not empty
                    cleaned[key] = sub_dict
            else:
                cleaned[key] = value
        return cleaned
    
    def _validate_config(self, config: Dict[str, Any]) -> Optional[str]:
        """Validate configuration and return warning message if issues found"""
        
        # Check for conflicting frequency settings
        freq = config.get("frequency", {})
        if freq.get("min_hz") and freq.get("max_hz"):
            if freq["min_hz"] > freq["max_hz"]:
                return f"min_hz ({freq['min_hz']}) > max_hz ({freq['max_hz']})"
        
        if freq.get("target_hz"):
            if freq.get("min_hz") and freq["target_hz"] < freq["min_hz"]:
                return f"target_hz ({freq['target_hz']}) < min_hz ({freq['min_hz']})"
            if freq.get("max_hz") and freq["target_hz"] > freq["max_hz"]:
                return f"target_hz ({freq['target_hz']}) > max_hz ({freq['max_hz']})"
        
        # Check if any targets are specified
        has_targets = (
            freq.get("min_hz") or freq.get("max_hz") or freq.get("target_hz") or
            config.get("throughput", {}).get("target_percentage") or
            config.get("latency", {}).get("max_latency_ms")
        )
        
        if not has_targets:
            return "No performance targets specified"
        
        return None
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Config should update when parameters change"""
        return float("nan")  # Always mark as changed