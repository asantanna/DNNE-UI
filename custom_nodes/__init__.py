# This file makes custom_nodes a package so that absolute imports work
# Individual node files are loaded by nodes.py

# Collect all NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all node modules and collect their mappings
import os
import importlib

current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith('_visnode.py'):
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f'.{module_name}', package='custom_nodes')
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)