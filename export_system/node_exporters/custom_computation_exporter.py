#!/usr/bin/env python3
"""
Exporter for Custom Computation node using queue-based template
"""

import os
from pathlib import Path
import dnne_config
from ..graph_exporter import ExportableNode
from custom_nodes.utils.script_loader import load_custom_script
from ..subsystems import SUBSYSTEM_UTIL


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
        
        # For the exported code, use relative path to the copied file
        # The file will be copied to custom_compute_funcs/ subdirectory
        file_name = os.path.basename(src_path)
        exported_path = f"custom_compute_funcs/{file_name}"
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CustomComputationNode",
            "SRC_PATH": exported_path,  # Use relative path in the export
            "MODULE_NAME": module_name
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import os",
            "import torch",
            "import importlib.util",
            "import inspect",
            "from framework.exceptions import CauseExitException",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["output"]
    
    @classmethod
    def get_input_names(cls):
        return ["input"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """Get initial output schema from the custom script."""
        # Get src_path parameter
        params = cls.get_node_parameters_batch(node_data, [{'name': 'src_path', 'widget_index': 0}])
        src_path = params['src_path'].strip()
        
        if not src_path:
            raise ValueError(f"CustomComputation node missing required src_path")
        
        # Load script and get initial schema
        module = load_custom_script(src_path)
        return module.get_script_output_schema(initial=True)
    
    @classmethod
    def get_output_schema(cls, node_data, connections=None, node_registry=None, 
                         all_nodes=None, all_links=None):
        """Get output schema, potentially resolving from input connections."""
        # Get initial schema
        schema = cls.get_initial_output_schema(node_data)
        
        # Try to resolve with input schema
        if connections and "inputs" in connections and "input" in connections["inputs"]:
            input_info = connections["inputs"]["input"]
            
            # Get input schema from connected node
            if node_registry and all_nodes and all_links:
                source_node_id = input_info["from_node"]
                source_output_slot = input_info["from_slot"]
                
                # Find source node and get its schema
                for node in all_nodes:
                    if str(node["id"]) == source_node_id:
                        source_type = node.get("class_type") or node.get("type")
                        if source_type in node_registry:
                            source_exporter = node_registry[source_type]
                            
                            # Get source node's connections
                            source_connections = {}
                            for link in all_links:
                                if str(link[3]) == source_node_id:  # Target node
                                    target_slot = link[4]
                                    source_connections.setdefault("inputs", {})[target_slot] = {
                                        "from_node": str(link[1]),
                                        "from_slot": link[2]
                                    }
                            
                            # Get source node's output schema
                            source_schema = source_exporter.get_output_schema(
                                node, source_connections, node_registry, all_nodes, all_links
                            )
                            
                            # Get the specific output
                            output_names = source_exporter.get_output_names()
                            if source_output_slot < len(output_names):
                                output_name = output_names[source_output_slot]
                                if "outputs" in source_schema and output_name in source_schema["outputs"]:
                                    input_schema = source_schema["outputs"][output_name]
                                    
                                    # Load script and get resolved schema
                                    params = cls.get_node_parameters_batch(node_data, [{'name': 'src_path', 'widget_index': 0}])
                                    module = load_custom_script(params['src_path'].strip())
                                    
                                    # Get resolved schema from script
                                    schema = module.get_script_output_schema(
                                        initial=False,
                                        input_schema={"input": input_schema}
                                    )
                            break
        
        return schema
    
    @classmethod
    def get_export_files(cls, node_id, node_data):
        """Return list of files to copy during export."""
        # Get the src_path parameter
        param_specs = [
            {'name': 'src_path', 'widget_index': 0},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        src_path = params.get('src_path', '').strip() if params.get('src_path') else ''
        
        # If no source path, nothing to copy
        if not src_path:
            return []
        
        # Resolve the full path using same logic as prepare_template_vars
        import os
        import dnne_config
        
        # If no directory separator, look in standard location
        if os.sep not in src_path:
            dnne_root = dnne_config.get_dnne_root()
            src_path = os.path.join(dnne_root, "user", "default", "custom_compute_funcs", src_path)
        
        # Copy to custom_compute_funcs subdirectory in the export
        return [(src_path, "custom_compute_funcs")]

    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_UTIL