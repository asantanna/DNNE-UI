# This file makes custom_nodes a package so that absolute imports work
# Individual node files are loaded automatically using the decorator system

# Collect all NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all node modules to trigger decorator registration
import os
import importlib
import logging

logger = logging.getLogger(__name__)

# Import decorator to access registry
from custom_nodes.utils.dnne_decorator import get_all_node_classes

# First import all visnode modules to trigger decorators
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith('_visnode.py'):
        module_name = filename[:-3]  # Remove .py extension
        try:
            module = importlib.import_module(f'.{module_name}', package='custom_nodes')
            # Extract mappings from module if they exist (for display names)
            if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        except Exception as e:
            logger.error(f"Failed to import {module_name}: {e}")

# Now get all registered nodes from the decorator
all_nodes = get_all_node_classes()

# Build NODE_CLASS_MAPPINGS from decorator registry
for node_name, node_class in all_nodes.items():
    # Determine the key for NODE_CLASS_MAPPINGS
    # Remove "Node" suffix if present for the key
    if node_name.endswith('Node'):
        key = node_name[:-4]
    else:
        key = node_name
    
    NODE_CLASS_MAPPINGS[key] = node_class
    
    # If no display name was provided, create one
    if key not in NODE_DISPLAY_NAME_MAPPINGS:
        # Convert PascalCase to Title Case
        import re
        display_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', key)
        display_name = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', display_name)
        NODE_DISPLAY_NAME_MAPPINGS[key] = display_name

# Sort by display name for consistent ordering
sorted_items = sorted(NODE_DISPLAY_NAME_MAPPINGS.items(), key=lambda x: x[1])
NODE_CLASS_MAPPINGS = {k: NODE_CLASS_MAPPINGS[k] for k, _ in sorted_items if k in NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = dict(sorted_items)

# For backward compatibility with tests - expose node classes with expected names
# These aliases ensure tests can import nodes directly
for key, node_class in NODE_CLASS_MAPPINGS.items():
    # Make the class available as a module attribute
    globals()[node_class.__name__] = node_class
    
    # Also add common aliases
    if key in NODE_CLASS_MAPPINGS:
        globals()[f"{key}Node"] = NODE_CLASS_MAPPINGS[key]

# Special aliases for tests that expect specific names
if 'IsaacGymEnvs' in NODE_CLASS_MAPPINGS:
    globals()['IsaacGymEnvNode'] = NODE_CLASS_MAPPINGS['IsaacGymEnvs']

# ISAAC_GYM_AVAILABLE flag for tests
try:
    import isaacgym
    ISAAC_GYM_AVAILABLE = True
except ImportError:
    ISAAC_GYM_AVAILABLE = False

globals()['ISAAC_GYM_AVAILABLE'] = ISAAC_GYM_AVAILABLE

logger.info(f"Registered {len(NODE_CLASS_MAPPINGS)} nodes via decorator system")