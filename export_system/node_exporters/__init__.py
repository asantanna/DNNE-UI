"""
Node exporter classes that handle code generation using queue-based templates
Auto-discovery based on naming conventions
"""

import logging
from pathlib import Path
import importlib
import sys

# Add path for custom_nodes imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from custom_nodes.utils.dnne_decorator import get_all_node_classes, is_virtual_node
from custom_nodes.utils.naming_utils import node_class_to_exporter_filename, node_class_to_exporter_class
from ..utils.exporter_discovery import register_all_exporters_auto

logger = logging.getLogger(__name__)

# Main registration function using auto-discovery
def register_all_exporters(exporter):
    """
    Register all node exporters with the graph exporter using auto-discovery.
    
    Args:
        exporter: The GraphExporter instance to register with
        
    Raises:
        RuntimeError: If registration fails
    """
    return register_all_exporters_auto(exporter)

# Dynamically import all exporter classes based on decorator metadata
# This allows direct imports like: from export_system.node_exporters import MNISTDatasetExporter
__all__ = ['register_all_exporters']

# Get all registered nodes from decorator
all_nodes = get_all_node_classes()

# Track errors for non-virtual nodes
import_errors = []

# For each node, try to import its corresponding exporter
for node_name, node_class in all_nodes.items():
    # Check if node is virtual
    is_virtual = is_virtual_node(node_class)
    
    # Generate expected exporter class name
    exporter_class_name = node_class_to_exporter_class(node_name)
    exporter_filename = node_class_to_exporter_filename(node_name)[:-3]  # Remove .py
    
    # Try to import the exporter
    try:
        module = importlib.import_module(f'.{exporter_filename}', package='export_system.node_exporters')
        
        # Get the exporter class - fail fast if missing
        try:
            exporter_class = module.__dict__[exporter_class_name]
            
            # Make it available as a module attribute
            globals()[exporter_class_name] = exporter_class
            __all__.append(exporter_class_name)
            
            logger.debug(f"Imported exporter: {exporter_class_name}")
        except KeyError:
            # Class not found in module
            if not is_virtual:
                error_msg = f"Exporter class {exporter_class_name} not found in module {exporter_filename}"
                logger.error(error_msg)
                import_errors.append(error_msg)
                
    except ImportError as e:
        # ImportError is only expected for virtual nodes
        if is_virtual:
            logger.debug(f"Skipping exporter for virtual node {node_name}")
        else:
            error_msg = f"Failed to import exporter for non-virtual node {node_name}: {e}"
            logger.error(error_msg)
            import_errors.append(error_msg)
            
    except Exception as e:
        # Unexpected errors are always bad
        error_msg = f"Unexpected error importing exporter for {node_name}: {e}"
        logger.error(error_msg)
        import_errors.append(error_msg)

# Fail fast if there were errors for non-virtual nodes
if import_errors:
    raise RuntimeError(
        f"Failed to import required exporters:\n" + 
        '\n'.join(f"  - {err}" for err in import_errors)
    )

logger.info(f"Auto-imported {len(__all__) - 1} exporter classes")