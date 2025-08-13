#!/usr/bin/env python3
"""
Exporter for Custom Computation node using queue-based template
"""

import os
from pathlib import Path
import dnne_config
from ..graph_exporter import ExportableNode


class CustomComputationExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/custom_computation_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'src_path', 'widget_index': 0}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        src_path = params.get('src_path', '').strip()
        if not src_path:
            raise ValueError(
                f"CustomComputation node {node_id} missing required src_path parameter. "
                f"Must provide path to Python file containing compute() function."
            )
        
        # Check if src_path contains path separators
        if os.sep not in src_path:
            # No path separators - assume it's just a filename in custom_compute_funcs
            dnne_root = dnne_config.get_dnne_root()
            src_path = os.path.join(dnne_root, "user", "default", "custom_compute_funcs", src_path)
        
        # Convert to absolute path if relative
        src_path = os.path.abspath(src_path)
        
        # Validate file exists at export time (fail-fast)
        if not os.path.exists(src_path):
            raise FileNotFoundError(
                f"CustomComputation node {node_id}: Source file not found: {src_path}. "
                f"Please ensure the file exists before exporting."
            )
        
        # Validate it's a Python file
        if not src_path.endswith('.py'):
            raise ValueError(
                f"CustomComputation node {node_id}: Source file must be a Python file (.py), got: {src_path}"
            )
        
        # Generate a safe module name from the file path
        module_name = Path(src_path).stem.replace('-', '_').replace(' ', '_')
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CustomComputationNode",
            "SRC_PATH": src_path,
            "MODULE_NAME": module_name
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import os",
            "import torch",
            "import importlib.util",
            "import inspect",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["output"]
    
    @classmethod
    def get_input_names(cls):
        return ["input"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # Output schema depends on the custom function, so we can't determine it statically
        # The output will match the input schema (passthrough assumption)
        return {
            "outputs": {
                "output": {
                    "type": "tensor",
                    "dynamic": True,  # Indicates schema depends on runtime
                    "dtype": "float32"
                }
            }
        }