"""
DNNE-UI Nodes
Minimal node system for robotics applications.
"""

import os
import json
import sys
import importlib
import importlib.util
import traceback
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

# Define valid types for the node system
VALID_TYPES = {
    "STRING": "STRING",
    "INT": "INT", 
    "FLOAT": "FLOAT",
    "BOOLEAN": "BOOLEAN",
    "*": "*",  # Wildcard type for "any" connections
}

# Add to ComfyUI's type registry
from custom_nodes.robotics_nodes.robotics_types import ROBOTICS_TYPES
VALID_TYPES.update(ROBOTICS_TYPES)

# Base node class
class DNNENode:
    """Base class for all DNNE nodes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input types - override in subclasses"""
        return {"required": {}}
    
    RETURN_TYPES: Tuple = ()
    RETURN_NAMES: Optional[Tuple] = None
    FUNCTION: str = "compute"
    CATEGORY: str = "dnne"
    
    def compute(self, **kwargs):
        """Main computation function - override in subclasses"""
        raise NotImplementedError("Subclasses must implement compute()")

# Basic utility nodes
class FloatConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0})
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_value"
    CATEGORY = "dnne/constants"
    
    def get_value(self, value):
        return (value,)

class IntConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -10000, "max": 10000})
            }
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_value"
    CATEGORY = "dnne/constants"
    
    def get_value(self, value):
        return (value,)

class StringConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"  
    CATEGORY = "dnne/constants"
    
    def get_value(self, value):
        return (value,)

# Debug/Display nodes
class PrintValue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {}),
                "prefix": ("STRING", {"default": "Value:"})
            }
        }
    
    RETURN_TYPES = ("*",)
    FUNCTION = "print_value"
    CATEGORY = "dnne/debug"
    OUTPUT_NODE = True
    
    def print_value(self, value, prefix):
        print(f"{prefix} {value}")
        return (value,)

# Node mappings for the system
NODE_CLASS_MAPPINGS = {
    "FloatConstant": FloatConstant,
    "IntConstant": IntConstant, 
    "StringConstant": StringConstant,
    "PrintValue": PrintValue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatConstant": "Float Constant",
    "IntConstant": "Integer Constant",
    "StringConstant": "String Constant", 
    "PrintValue": "Print Value",
}

# Global variables to track loaded nodes
custom_nodes_loaded = False
api_nodes_loaded = False
LOADED_MODULE_DIRS = {}  # Track loaded custom node directories
EXTENSION_WEB_DIRS = {}  # Track extension web directories

def init_extra_nodes(init_custom_nodes=True, init_api_nodes=True):
    """
    Initialize extra nodes (custom nodes and API nodes)
    This function is called by main.py during startup
    """
    global custom_nodes_loaded, api_nodes_loaded
    
    print("Initializing DNNE extra nodes...")
    
    if init_custom_nodes and not custom_nodes_loaded:
        load_custom_nodes()
        custom_nodes_loaded = True
    
    if init_api_nodes and not api_nodes_loaded:
        load_api_nodes()
        api_nodes_loaded = True
    
    print(f"DNNE nodes loaded: {len(NODE_CLASS_MAPPINGS)} total")

def load_custom_nodes():
    """Load custom nodes from custom_nodes directory"""
    print("Loading custom nodes...")
    
    custom_nodes_dir = os.path.join(os.path.dirname(__file__), "custom_nodes")
    if not os.path.exists(custom_nodes_dir):
        print(f"Custom nodes directory not found: {custom_nodes_dir}")
        return
    
    # Look for Python files and packages in custom_nodes
    for item in os.listdir(custom_nodes_dir):
        if item.startswith('.') or item.startswith('_'):
            continue
            
        item_path = os.path.join(custom_nodes_dir, item)
        
        try:
            if os.path.isfile(item_path) and item.endswith('.py'):
                # Load Python file
                module_name = item[:-3]  # Remove .py
                load_custom_node_file(item_path, module_name)
                
            elif os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py')):
                # Load Python package
                load_custom_node_package(item_path, item)
                
        except Exception as e:
            print(f"Failed to load custom node {item}: {e}")
            traceback.print_exc()

def load_custom_node_file(file_path, module_name):
    """Load a single custom node Python file"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Track the loaded module
        module_dir = os.path.dirname(file_path)
        LOADED_MODULE_DIRS[module_name] = module_dir
        
        # Check for web directory (less common for single files)
        web_dir = os.path.join(module_dir, "web")
        if os.path.exists(web_dir):
            EXTENSION_WEB_DIRS[module_name] = web_dir
            print(f"  Found web directory: {module_name}")
        
        # Register node classes
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            for name, cls in module.NODE_CLASS_MAPPINGS.items():
                NODE_CLASS_MAPPINGS[name] = cls
                print(f"  Loaded node: {name}")
        
        # Register display names
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

def load_custom_node_package(package_path, package_name):
    """Load a custom node Python package"""
    sys.path.insert(0, os.path.dirname(package_path))
    try:
        module = importlib.import_module(package_name)
        
        # Track the loaded module directory
        LOADED_MODULE_DIRS[package_name] = package_path
        
        # Check for web directory
        web_dir = os.path.join(package_path, "web")
        if os.path.exists(web_dir):
            EXTENSION_WEB_DIRS[package_name] = web_dir
            print(f"  Found web directory: {package_name}")
        
        # Register node classes
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            for name, cls in module.NODE_CLASS_MAPPINGS.items():
                NODE_CLASS_MAPPINGS[name] = cls
                print(f"  Loaded node: {name}")
        
        # Register display names  
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            
    finally:
        if sys.path[0] == os.path.dirname(package_path):
            sys.path.pop(0)

def load_api_nodes():
    """Load API nodes (placeholder for now)"""
    print("Loading API nodes...")
    # For now, just a placeholder
    # TODO: Implement API node loading if needed

def get_object_info():
    """
    Get information about all available nodes
    This is used by the frontend to populate the node palette
    """
    object_info = {}
    
    for name, cls in NODE_CLASS_MAPPINGS.items():
        try:
            info = {
                "input": cls.INPUT_TYPES(),
                "output": getattr(cls, 'RETURN_TYPES', ()),
                "output_name": getattr(cls, 'RETURN_NAMES', ()),
                "name": NODE_DISPLAY_NAME_MAPPINGS.get(name, name),
                "display_name": NODE_DISPLAY_NAME_MAPPINGS.get(name, name),
                "description": getattr(cls, '__doc__', ''),
                "category": getattr(cls, 'CATEGORY', 'dnne'),
                "output_node": getattr(cls, 'OUTPUT_NODE', False),
            }
            object_info[name] = info
        except Exception as e:
            print(f"Error getting info for node {name}: {e}")
    
    return object_info

def load_checkpoint_guess_config(*args, **kwargs):
    """Stub for checkpoint loading - not used in robotics"""
    raise NotImplementedError("Checkpoint loading not supported in DNNE")

def unload_all_models():
    """Stub for model unloading"""
    print("Models unloaded (DNNE stub)")

def soft_empty_cache():
    """Stub for cache clearing"""
    print("Cache cleared (DNNE stub)")

def interrupt_processing(value=True):
    """Stub for interrupt processing"""
    print(f"Processing interrupt: {value}")

# Export for ComfyUI
__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS",
    "LOADED_MODULE_DIRS",
    "EXTENSION_WEB_DIRS",
    "init_extra_nodes",
    "get_object_info",
    "load_checkpoint_guess_config",
    "unload_all_models", 
    "soft_empty_cache",
    "interrupt_processing"
]
