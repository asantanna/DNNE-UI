# This file makes custom_nodes a package so that absolute imports work
# Individual node files are loaded by nodes.py

# Collect all NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all node modules and collect their mappings
import os
import importlib

# First collect all nodes from modules
temp_nodes = []
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith('_visnode.py'):
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f'.{module_name}', package='custom_nodes')
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            for key, cls in module.NODE_CLASS_MAPPINGS.items():
                display_name = module.NODE_DISPLAY_NAME_MAPPINGS.get(key, key) if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS') else key
                temp_nodes.append((key, cls, display_name))

# Sort by display name and build the dictionaries
for key, node_class, display_name in sorted(temp_nodes, key=lambda x: x[2]):
    NODE_CLASS_MAPPINGS[key] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[key] = display_name