"""
Automatic discovery and registration of node exporters based on naming conventions
"""

import os
import importlib
import logging
from pathlib import Path
from typing import Dict, Type, Optional

# Import shared naming utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from custom_nodes.utils.naming_utils import (
    to_snake_case, 
    to_pascal_case,
    node_class_to_exporter_filename,
    node_class_to_exporter_class
)
from custom_nodes.utils.dnne_decorator import get_all_node_classes, is_virtual_node

logger = logging.getLogger(__name__)


def discover_exporters() -> Dict[str, Type]:
    """
    Automatically discover all exporter classes based on naming convention.
    
    Returns:
        Dictionary mapping node type names to exporter classes
        
    Raises:
        RuntimeError: If any errors occur during discovery
    """
    exporters = {}
    errors = []
    
    # Get the node_exporters directory
    exporters_dir = Path(__file__).parent.parent / 'node_exporters'
    
    if not exporters_dir.exists():
        error_msg = f"Node exporters directory not found: {exporters_dir}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Scan for all exporter files
    for filepath in exporters_dir.glob('*_exporter.py'):
        if filepath.name == '__init__.py':
            continue
            
        # Extract base name (e.g., 'mnist_dataset' from 'mnist_dataset_exporter.py')
        base_name = filepath.stem[:-9]  # Remove '_exporter' suffix
        
        # Convert to expected class name
        exporter_class_name = to_pascal_case(base_name) + 'Exporter'
        
        # Import the module
        module_name = f'export_system.node_exporters.{filepath.stem}'
        try:
            module = importlib.import_module(module_name)
            
            # Get the exporter class
            if hasattr(module, exporter_class_name):
                exporter_class = getattr(module, exporter_class_name)
                
                # Determine the node type this exporter handles
                # Convert snake_case filename to PascalCase node type
                node_type = to_pascal_case(base_name)
                
                exporters[node_type] = exporter_class
                logger.debug(f"Discovered exporter: {node_type} -> {exporter_class_name}")
            else:
                error_msg = f"Expected class {exporter_class_name} not found in {module_name}"
                logger.error(error_msg)
                errors.append(error_msg)
                
        except ImportError as e:
            error_msg = f"Failed to import {module_name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading {module_name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    if errors:
        raise RuntimeError(
            f"Failed to discover exporters:\n" + '\n'.join(f"  - {err}" for err in errors)
        )
    
    return exporters


def get_exporter_for_node(node_class_or_name) -> Optional[Type]:
    """
    Get the exporter class for a given node.
    
    Args:
        node_class_or_name: Either the node class or its string name
        
    Returns:
        The exporter class or None if not found
        
    Raises:
        RuntimeError: If an unexpected error occurs
    """
    try:
        # Get node class name
        if isinstance(node_class_or_name, str):
            node_name = node_class_or_name
        else:
            node_name = node_class_or_name.__name__
        
        # Generate expected exporter filename
        exporter_filename = node_class_to_exporter_filename(node_name)
        exporter_class_name = node_class_to_exporter_class(node_name)
        
        # Build module path
        module_name = f'export_system.node_exporters.{exporter_filename[:-3]}'  # Remove .py
        
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, exporter_class_name):
                return getattr(module, exporter_class_name)
            else:
                # Class not found in module - this is expected for some nodes
                return None
        except ImportError:
            # Module doesn't exist - this is expected for some nodes
            return None
            
    except Exception as e:
        error_msg = f"Unexpected error getting exporter for {node_class_or_name}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def validate_exporter_templates(exporter_class, node_name: str, is_virtual: bool):
    """
    Validate that exporter has correct template configuration.
    
    Args:
        exporter_class: The exporter class to validate
        node_name: Name of the node for error messages
        is_virtual: Whether the node is virtual
        
    Raises:
        ValueError: If validation fails
        RuntimeError: If an unexpected error occurs
    """
    try:
        has_get_template = hasattr(exporter_class, 'get_template_name')
        
        if is_virtual:
            # Virtual nodes MUST NOT have templates
            if has_get_template:
                # Check if it returns None or raises NotImplementedError
                try:
                    template = exporter_class.get_template_name()
                    if template is not None:
                        raise ValueError(
                            f"Virtual node {node_name} has exporter with template '{template}'. "
                            f"Virtual nodes must NOT have templates."
                        )
                except NotImplementedError:
                    # This is acceptable for virtual nodes
                    pass
        else:
            # Non-virtual nodes MUST have templates
            if not has_get_template:
                raise ValueError(
                    f"Non-virtual node {node_name} exporter missing get_template_name() method. "
                    f"Non-virtual nodes MUST have templates."
                )
            
            # Verify it returns a valid template
            try:
                template = exporter_class.get_template_name()
                if not template:
                    raise ValueError(
                        f"Non-virtual node {node_name} exporter get_template_name() returned empty/None. "
                        f"Non-virtual nodes MUST specify a template."
                    )
            except NotImplementedError:
                raise ValueError(
                    f"Non-virtual node {node_name} exporter get_template_name() not implemented. "
                    f"Non-virtual nodes MUST implement this method."
                )
                
    except ValueError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        error_msg = f"Unexpected error validating exporter for {node_name}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def register_all_exporters_auto(graph_exporter):
    """
    Automatically register all node exporters with the graph exporter.
    
    Args:
        graph_exporter: The GraphExporter instance to register with
        
    Raises:
        RuntimeError: If any non-virtual node is missing an exporter, validation fails,
                     or any unexpected error occurs
    """
    try:
        # Get all registered nodes from decorator
        all_nodes = get_all_node_classes()
        
        registered_count = 0
        skipped_virtual = 0
        missing_exporters = []
        validation_errors = []
        
        for node_name, node_class in all_nodes.items():
            try:
                is_virtual = is_virtual_node(node_class)
                
                # Get the exporter for this node
                exporter_class = get_exporter_for_node(node_class)
                
                if exporter_class:
                    # Validate template configuration
                    try:
                        validate_exporter_templates(exporter_class, node_name, is_virtual)
                    except ValueError as e:
                        validation_errors.append(str(e))
                        continue
                    
                    # Register with the expected key (remove 'Node' suffix)
                    if node_name.endswith('Node'):
                        registry_key = node_name[:-4]
                    else:
                        registry_key = node_name
                    
                    graph_exporter.register_node(registry_key, exporter_class)
                    
                    if is_virtual:
                        skipped_virtual += 1
                    else:
                        registered_count += 1
                else:
                    # Missing exporter is only fatal for non-virtual nodes
                    if not is_virtual:
                        missing_exporters.append(node_name)
                    else:
                        skipped_virtual += 1
                        
            except RuntimeError:
                # Re-raise runtime errors from helper functions
                raise
            except Exception as e:
                error_msg = f"Unexpected error processing node {node_name}: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        # Check for fatal errors
        fatal_errors = []
        
        if missing_exporters:
            fatal_errors.append(
                f"Missing exporters for non-virtual nodes: {', '.join(missing_exporters)}\n"
                f"Expected exporter files:\n" + 
                '\n'.join(f"  - {node_class_to_exporter_filename(name)}" for name in missing_exporters)
            )
        
        if validation_errors:
            fatal_errors.append(
                "Template validation errors:\n" + '\n'.join(f"  - {err}" for err in validation_errors)
            )
        
        if fatal_errors:
            error_msg = "Failed to register exporters:\n" + '\n\n'.join(fatal_errors)
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Log summary
        logger.debug(f"Auto-registered {registered_count} exporters for non-virtual nodes")
        logger.debug(f"Registered {skipped_virtual} virtual nodes (no code generation)")
        
        return registered_count
        
    except RuntimeError:
        # Re-raise runtime errors as-is
        raise
    except Exception as e:
        error_msg = f"Unexpected error in register_all_exporters_auto: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)