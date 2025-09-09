#!/usr/bin/env python3
"""
DNNE Queue-Based Export System
Converts node graphs to reactive Python scripts using async queues
"""

from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from .utils import export_utils
from .subsystems import ALL_SUBSYSTEMS

class ExportableNode:
    """Base class for nodes that can be exported to code"""
    _schema_cache = {}  # Initialize cache at class level
    
    @classmethod
    def get_template_name(cls) -> str:
        """Return the template file name for this node type"""
        raise NotImplementedError(f"Subclass {cls.__name__} must implement get_template_name() method")
    
    @classmethod
    def prepare_template_vars(cls, node_id: str, node_data: Dict, 
                            connections: Dict, node_registry: Dict = None, 
                            all_nodes: List = None, all_links: List = None) -> Dict[str, Any]:
        """Prepare variables for template substitution"""
        raise NotImplementedError(f"Subclass {cls.__name__} must implement prepare_template_vars() method")
    
    @classmethod
    def get_imports(cls) -> List[str]:
        """Return list of import statements needed by this node"""
        return []
    
    @classmethod
    def get_output_names(cls) -> List[str]:
        """Return list of output names for this node type"""
        raise NotImplementedError(f"Subclass {cls.__name__} must implement get_output_names() method")
    
    @classmethod
    def get_input_names(cls) -> List[str]:
        """Return list of input names for this node type"""
        raise NotImplementedError(f"Subclass {cls.__name__} must implement get_input_names() method")
    
    @classmethod
    def get_subsystem(cls) -> Union[str, List[str]]:
        """Return the subsystem(s) this node belongs to.
        
        Every exporter MUST override this method to declare its subsystem(s).
        Use constants from export_system.subsystems module.
        
        Returns:
            Single subsystem name or list of subsystem names
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"Exporter {cls.__name__} must implement get_subsystem() method. "
            f"Use constants from export_system.subsystems (e.g., SUBSYSTEM_TRAINING). "
            f"Every node must belong to at least one subsystem."
        )
    
    @classmethod
    def validate_subsystem(cls, subsystem: Union[str, List[str]]) -> None:
        """Validate that subsystem(s) are valid constants.
        
        Raises:
            ValueError: If invalid subsystem name is used
        """
        subsystems = [subsystem] if isinstance(subsystem, str) else subsystem
        
        for sub in subsystems:
            if sub not in ALL_SUBSYSTEMS:
                raise ValueError(
                    f"Invalid subsystem '{sub}' in {cls.__name__}. "
                    f"Must use constants from export_system.subsystems. "
                    f"Valid subsystems: {sorted(ALL_SUBSYSTEMS)}"
                )
    
    @classmethod
    def get_input_name_for_slot(cls, slot: int) -> str:
        """Get input name for a specific slot number"""
        input_names = cls.get_input_names()
        if slot < len(input_names):
            return input_names[slot]
        return f"input_{slot}"
    
    @classmethod
    def validate_required_parameters(cls, params: Dict[str, Any], required_params: List[str], 
                                    node_id: str, node_type: str) -> bool:
        """Validate that all required parameters are present.
        
        This centralized method enforces fail-fast principles by checking that all
        required parameters exist and are not None. This prevents silent defaults
        from hiding configuration errors.
        
        Args:
            params: Dictionary of parameters extracted from node
            required_params: List of parameter names that must be present
            node_id: ID of the node being validated
            node_type: Type/class name of the node for error messages
            
        Returns:
            True if all required parameters are present
            
        Raises:
            ValueError: If any required parameters are missing or None
        """
        missing_params = [p for p in required_params if p not in params or params[p] is None]
        if missing_params:
            raise ValueError(
                f"{node_type} node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all required configuration parameters."
            )
        return True
    
    @classmethod
    def get_required_input_names(cls) -> List[str]:
        """Return list of required input names for this node type.
        
        Automatically determines required inputs from the UI node's INPUT_TYPES.
        Only override this for special cases (like Concat with flexible inputs).
        
        Returns:
            List of required input names (connections only, not widgets)
            
        Raises:
            RuntimeError: If UI node class cannot be found or INPUT_TYPES is missing
            ValueError: If INPUT_TYPES structure is invalid
        """
        # Import here to avoid circular imports
        from .utils.export_utils import get_ui_node_class
        
        # Get the UI node class - will raise RuntimeError if not found
        ui_node_class = get_ui_node_class(cls.__name__)
        
        # Verify INPUT_TYPES exists
        if not hasattr(ui_node_class, 'INPUT_TYPES'):
            raise RuntimeError(
                f"UI node class {ui_node_class.__name__} missing INPUT_TYPES classmethod. "
                f"This is a bug - all UI nodes must define INPUT_TYPES."
            )
        
        # Get input types
        input_types = ui_node_class.INPUT_TYPES()
        
        # Validate structure
        if not isinstance(input_types, dict):
            raise ValueError(
                f"UI node {ui_node_class.__name__} INPUT_TYPES returned {type(input_types)} "
                f"instead of dict. This is a bug in the node implementation."
            )
        
        # Get required section (may be empty dict)
        required_section = input_types.get('required', {})
        
        # Get our connection input names (not widgets)
        our_input_names = cls.get_input_names()
        
        # Return only inputs that are both:
        # 1. In the required section of INPUT_TYPES
        # 2. In our list of connection inputs
        required_inputs = []
        for input_name in our_input_names:
            if input_name in required_section:
                required_inputs.append(input_name)
        
        # Note: We DON'T validate that all required UI inputs are in our input list
        # because some UI inputs might be widgets (not connections)
        
        return required_inputs
    
    @classmethod
    def validate_required_connections(cls, node_id: str, connections: Dict) -> None:
        """Validate that all required input connections are present.
        
        This method ensures fail-fast behavior by checking that all required
        inputs have connections at export time, preventing runtime failures.
        
        Args:
            node_id: ID of the node being validated
            connections: Dictionary containing input/output connection info
            
        Raises:
            ValueError: If any required input connections are missing
        """
        required_inputs = cls.get_required_input_names()
        missing_connections = []
        
        # Check each required input
        for input_name in required_inputs:
            if "inputs" not in connections or input_name not in connections.get("inputs", {}):
                missing_connections.append(input_name)
        
        # Raise error with clear message if any connections are missing
        if missing_connections:
            node_type = cls.__name__.replace("Exporter", "")
            raise ValueError(
                f"{node_type} node {node_id} missing required input connections: {missing_connections}. "
                f"Please connect all required inputs before exporting."
            )
    
    @classmethod
    def prepare_template_vars_with_validation(cls, node_id: str, node_data: Dict,
                                             connections: Dict, node_registry: Dict = None,
                                             all_nodes: List = None, all_links: List = None) -> Dict[str, Any]:
        """Wrapper that validates connections before preparing template variables.
        
        This method ensures all required connections are present before attempting
        to generate code, providing fail-fast behavior at export time.
        
        Args:
            node_id: ID of the node
            node_data: Node data from workflow
            connections: Input/output connections for this node
            node_registry: Registry of all node exporters
            all_nodes: List of all nodes in workflow
            all_links: List of all links in workflow
            
        Returns:
            Dictionary of template variables for code generation
            
        Raises:
            ValueError: If required connections are missing
        """
        # Validate connections first
        cls.validate_required_connections(node_id, connections)
        
        # Then prepare template variables
        return cls.prepare_template_vars(node_id, node_data, connections,
                                        node_registry, all_nodes, all_links)
    
    @classmethod
    def get_export_files(cls, node_id: str, node_data: Dict) -> List[Tuple[str, str]]:
        """Return list of files/directories to copy during export.
        
        Returns:
            List of tuples: [(source_path, dest_dir), ...]
            where dest_dir is relative to the export package root.
            
        Override in subclasses that need to export data files.
        """
        return []
    
    @classmethod
    def get_node_parameter(cls, node_data: Dict, param_name: str, default_value=None, widget_index: int = None):
        """
        Universal parameter reader that handles ComfyUI's inconsistent data formats.
        
        ComfyUI sometimes provides parameters in:
        - inputs dict (processed format): {'param_name': value}
        - widgets_values array (raw format): [value1, value2, ...]
        
        Args:
            node_data: Node data from ComfyUI
            param_name: Name of the parameter to retrieve
            default_value: Default value if parameter not found
            widget_index: Index in widgets_values array (for raw format)
            
        Returns:
            Parameter value or default_value if not found
        """
        # Try inputs dict first (processed format)
        inputs = node_data.get("inputs", {})
        if param_name in inputs:
            return inputs[param_name]
        
        # Fall back to widgets_values array (raw format)
        # Note: Fallbacks are allowed in this function
        if widget_index is not None:
            widget_values = node_data.get("widgets_values", [])
            if widget_index < len(widget_values):
                return widget_values[widget_index]
        
        # Return default if not found in either format
        return default_value
    
    @classmethod
    def get_node_parameters_batch(cls, node_data: Dict, param_specs: List[Dict]):
        """
        Get multiple parameters at once using batch specification.
        
        Args:
            node_data: Node data from ComfyUI
            param_specs: List of parameter specifications:
                [{'name': 'param1', 'widget_index': 0, 'default': value}, ...]
                
        Returns:
            Dict mapping parameter names to values
        """
        result = {}
        for spec in param_specs:
            param_name = spec['name']
            widget_index = spec.get('widget_index')
            default_value = spec.get('default')
            
            result[param_name] = cls.get_node_parameter(
                node_data, param_name, default_value, widget_index
            )
        
        return result
    
    @classmethod
    def get_initial_output_schema(cls, node_data: Dict) -> Dict[str, Any]:
        """Return initial schema with unresolved values as None. Override in subclasses."""
        raise NotImplementedError(f"Subclass {cls.__name__} must implement get_initial_output_schema() method")
    
    @classmethod
    def get_output_schema(cls, node_data: Dict, connections: Dict = None, 
                         node_registry: Dict = None, all_nodes: List = None, 
                         all_links: List = None) -> Dict[str, Any]:
        """
        Return schema describing the outputs of this node type.
        Resolves any None values by querying inputs if needed.
        """
        # For backward compatibility, if no connections provided, return initial schema
        if connections is None:
            return cls.get_initial_output_schema(node_data)
        
        # Check if we have cached the output schema
        cache_key = f"output_schema_{id(node_data)}"
        if cache_key in cls._schema_cache:
            return cls._schema_cache[cache_key]
        
        # Get initial schema and resolve None values
        schema = cls.get_initial_output_schema(node_data)
        schema_copy = json.loads(json.dumps(schema))  # Deep copy
        
        # Scan for None values and resolve them
        if cls._resolve_schema_nones(schema_copy, node_data, connections, 
                                     node_registry, all_nodes, all_links):
            # Cache the resolved schema
            cls._schema_cache[cache_key] = schema_copy
        
        return schema_copy
    
    @classmethod
    def get_input_schema(cls, node_data: Dict, connections: Dict,
                        node_registry: Dict, all_nodes: List, 
                        all_links: List) -> Dict[str, Any]:
        """
        Get schema for all inputs by querying connected nodes.
        """
        # Check if we have cached the input schema
        cache_key = f"input_schema_{id(node_data)}"
        if cache_key in cls._schema_cache:
            return cls._schema_cache[cache_key]
        
        # Build input schema by querying each connected input
        input_schema = {}
        input_names = cls.get_input_names()
        
        for input_name in input_names:
            if "inputs" in connections and input_name in connections["inputs"]:
                # Handle both single connection (dict) and multiple connections (list)
                input_connections = connections["inputs"][input_name]
                
                # For backward compatibility and schema resolution, use first connection
                if isinstance(input_connections, list):
                    if len(input_connections) == 0:
                        continue
                    input_info = input_connections[0]
                else:
                    # Legacy single connection format
                    input_info = input_connections
                    
                source_node_id = input_info["from_node"]
                source_output_slot = input_info["from_slot"]
                
                # Find the source node
                source_node_data = None
                source_node_type = None
                for node in all_nodes:
                    if str(node["id"]) == source_node_id:
                        source_node_data = node
                        source_node_type = node.get("class_type") or node.get("type")
                        break
                
                if source_node_data and source_node_type in node_registry:
                    source_node_class = node_registry[source_node_type]
                    
                    # Create a temporary exporter to get connections for the source node
                    temp_exporter = GraphExporter()
                    temp_exporter.node_registry = node_registry
                    source_connections = temp_exporter._get_node_connections(source_node_id, all_links, all_nodes)
                    
                    # Get schema from the source node's specific output
                    source_output_schema = source_node_class.get_output_schema_by_connector(
                        source_output_slot, source_node_data, source_connections,
                        node_registry, all_nodes, all_links
                    )
                    
                    input_schema[input_name] = source_output_schema
            else:
                input_schema[input_name] = None
        
        # Cache the resolved schema
        cls._schema_cache[cache_key] = input_schema
        
        return input_schema
    
    @classmethod
    def get_output_schema_by_connector(cls, connector_slot: int, node_data: Dict,
                                     connections: Dict, node_registry: Dict,
                                     all_nodes: List, all_links: List) -> Dict[str, Any]:
        """
        Get schema for a specific output connector.
        """
        # Get the full output schema
        full_schema = cls.get_output_schema(node_data, connections, 
                                          node_registry, all_nodes, all_links)
        
        # Get the output name for this slot
        output_names = cls.get_output_names()
        if connector_slot >= len(output_names):
            raise ValueError(f"Invalid output slot {connector_slot} for node type {cls.__name__}")
        
        output_name = output_names[connector_slot]
        
        # Return the schema for this specific output
        if "outputs" in full_schema and output_name in full_schema["outputs"]:
            output_schema = full_schema["outputs"][output_name]
            
            # Special handling for schema outputs - return the value directly
            if output_schema.get("type") == "schema" and "value" in output_schema:
                return output_schema["value"]
            
            return output_schema
        else:
            return {"type": "unknown", "shape": None}
    
    @classmethod
    def _resolve_schema_nones(cls, schema: Dict, node_data: Dict, connections: Dict,
                            node_registry: Dict, all_nodes: List, all_links: List) -> bool:
        """
        Recursively resolve None values in schema by querying inputs.
        Returns True if any values were resolved.
        """
        resolved_any = False
        
        if isinstance(schema, dict):
            for key, value in list(schema.items()):
                if value is None:
                    # Try to resolve this None value
                    # Subclasses should override this method to implement resolution logic
                    resolved_value = cls._resolve_schema_value(key, schema, node_data, 
                                                              connections, node_registry, 
                                                              all_nodes, all_links)
                    if resolved_value is not None:
                        schema[key] = resolved_value
                        resolved_any = True
                elif isinstance(value, dict):
                    if cls._resolve_schema_nones(value, node_data, connections,
                                               node_registry, all_nodes, all_links):
                        resolved_any = True
        
        return resolved_any
    
    @classmethod
    def _resolve_schema_value(cls, key: str, parent_schema: Dict, node_data: Dict,
                            connections: Dict, node_registry: Dict,
                            all_nodes: List, all_links: List) -> Any:
        """
        Resolve a specific None value in the schema.
        Subclasses should override this to implement custom resolution logic.
        """
        raise NotImplementedError(f"Subclass {cls.__name__} must implement _resolve_schema_value() method")

    


class GraphExporter:
    """Main export system that converts graphs to queue-based Python scripts"""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"
        self.node_registry = {}  # Maps node types to exportable classes
        self.logger = logging.getLogger(__name__)
        
        # Register all available node exporters
        from .node_exporters import register_all_exporters
        register_all_exporters(self)
        
    def register_node(self, node_type: str, node_class: type):
        """Register an exportable node type"""
        self.node_registry[node_type] = node_class
        self.logger.debug(f"Registered node type: {node_type}")
    
    @staticmethod
    def classname_to_exported_filename(class_name: str) -> str:
        """Convert a class name to its exported filename.
        
        This centralizes the class name → filename transformation logic used
        throughout the export system for consistency and maintainability.
        
        Args:
            class_name: Class name like "NetworkNode_56" or "CrossEntropyLossNode_51"
            
        Returns:
            Filename without extension like "networknode_56" or "crossentropylossnode_51"
            
        Examples:
            NetworkNode_56 -> networknode_56
            CrossEntropyLossNode_51 -> crossentropylossnode_51
            PPOTrainerNode_6 -> ppotrainernode_6
        """
        if not class_name:
            raise ValueError("Class name cannot be empty")
        
        # Standard transformation: replace 'Node_' with 'node_' and lowercase
        filename_base = class_name.replace('Node_', 'node_').lower()
        
        # Validate that the transformation is unambiguous
        if not filename_base or filename_base == 'node_':
            raise ValueError(f"Invalid class name transformation: {class_name} -> {filename_base}")
        
        return filename_base
    
    def _is_virtual_output(self, node_type: str, output_slot: int) -> bool:
        """Check if a specific output slot is virtual"""
        # Get the node class from registry
        if node_type not in self.node_registry:
            return False
            
        node_class = self.node_registry[node_type]
        
        # Check if the node class has the visnode module
        try:
            # Import the visnode module to get the actual node definition
            module_name = f"custom_nodes.{node_type.lower()}_visnode"
            if node_type.endswith("Node"):
                # Remove "Node" suffix for module name
                module_name = f"custom_nodes.{node_type[:-4].lower()}_visnode"
            
            import importlib
            module = importlib.import_module(module_name)
            
            # Get the actual node class
            visnode_class = getattr(module, f"{node_type if node_type.endswith('Node') else node_type + 'Node'}", None)
            if not visnode_class:
                return False
            
            # Check OUTPUT_DICT for virtual flag
            if hasattr(visnode_class, 'OUTPUT_DICT'):
                output_dict = visnode_class.OUTPUT_DICT
                if output_slot in output_dict:
                    return output_dict[output_slot].get('virtual', False)
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Could not check virtual output for {node_type}: {e}")
        
        return False
    
    def _is_virtual_input(self, node_type: str, input_name: str) -> bool:
        """Check if a specific input is virtual"""
        # Get the node class from registry
        if node_type not in self.node_registry:
            return False
            
        node_class = self.node_registry[node_type]
        
        # Check if the node class has the visnode module
        try:
            # Import the visnode module to get the actual node definition
            module_name = f"custom_nodes.{node_type.lower()}_visnode"
            if node_type.endswith("Node"):
                # Remove "Node" suffix for module name
                module_name = f"custom_nodes.{node_type[:-4].lower()}_visnode"
            
            import importlib
            module = importlib.import_module(module_name)
            
            # Get the actual node class
            visnode_class = getattr(module, f"{node_type if node_type.endswith('Node') else node_type + 'Node'}", None)
            if not visnode_class:
                return False
            
            # Check INPUT_TYPES for virtual flag
            if hasattr(visnode_class, 'INPUT_TYPES'):
                input_types = visnode_class.INPUT_TYPES()
                for category in ['required', 'optional']:
                    if category in input_types:
                        for name, spec in input_types[category].items():
                            if name == input_name:
                                # Check if spec has virtual flag
                                if isinstance(spec, tuple) and len(spec) > 1:
                                    config = None
                                    if isinstance(spec[-1], dict):
                                        config = spec[-1]
                                    elif len(spec) > 2 and isinstance(spec[2], dict):
                                        config = spec[2]
                                    
                                    if config and config.get('virtual', False):
                                        return True
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Could not check virtual input for {node_type}: {e}")
        
        return False
    
    def _is_virtual_node(self, node_type: str) -> bool:
        """Check if a node type is virtual (configuration-only)"""
        # External nodes (created by UI/frontend) are always virtual
        if export_utils.is_external_node(node_type):
            return True
            
        # Import decorator utilities
        from custom_nodes.utils.dnne_decorator import is_virtual_node, get_all_node_classes
        
        # Get all registered nodes
        all_nodes = get_all_node_classes()
        
        # Node type must have exact match with "Node" suffix
        # e.g., "MNISTDataset" -> "MNISTDatasetNode"
        expected_class_name = f"{node_type}Node"
        
        if expected_class_name not in all_nodes:
            # This is a fatal error - all nodes must be registered
            error_msg = f"Node type {node_type} (expected class {expected_class_name}) not found in decorator registry"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return is_virtual_node(all_nodes[expected_class_name])
    
    def _validate_runner_args_timestamps(self):
        """Lightweight check that runner_args.json is up-to-date"""
        import os
        import time
        
        template_dir = os.path.join(os.path.dirname(__file__), 'templates/framework')
        arg_parser_path = os.path.join(template_dir, 'arg_parser.tpl')
        runner_args_path = os.path.join(template_dir, 'runner_args.json')
        
        if not os.path.exists(runner_args_path):
            raise Exception(
                "runner_args.json is missing! Cannot export without UI configuration."
            )
        
        arg_parser_mtime = os.path.getmtime(arg_parser_path)
        runner_args_mtime = os.path.getmtime(runner_args_path)
        
        if arg_parser_mtime > runner_args_mtime:
            # Format times for clarity
            arg_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(arg_parser_mtime))
            json_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(runner_args_mtime))
            
            raise Exception(
                f"⚠️ EXPORT BLOCKED: runner_args.json is out of date!\n\n"
                f"arg_parser.tpl modified: {arg_time}\n"
                f"runner_args.json updated: {json_time}\n\n"
                f"The argument parser has been modified more recently than the UI configuration.\n"
                f"Please update runner_args.json to match the arguments in arg_parser.tpl\n"
                f"Run tests to verify: python -m unittest dnne_test_suite.test_runner_args_sync"
            )
    
    def _resolve_labels_from_nodes(self, workflow: Dict) -> List[Tuple]:
        """Resolve label connections from Label node properties (dictionary-free approach).
        
        Returns:
            List of connections as tuples: (from_node_id, from_slot, to_node_id, to_slot)
            
        Raises:
            ValueError: If there are orphaned or invalid label configurations
        """
        nodes = workflow.get("nodes", [])
        
        # Collect Label nodes and their properties
        output_labels = {}  # labelName -> source connection info
        input_labels = []   # List of input labels with target info
        orphaned_labels = []  # Labels with missing connection info
        
        # Debug: log all Label nodes found
        label_count = 0
        for node in nodes:
            if node.get("type") == "Label":
                label_count += 1
                self.logger.debug(f"Found Label node {node.get('id')} with properties: {node.get('properties', {})}")
        
        if label_count == 0:
            self.logger.debug("No Label nodes found in workflow")
        else:
            self.logger.debug(f"Found {label_count} Label nodes")
        
        for node in nodes:
            if node.get("type") == "Label":
                props = node.get("properties", {})
                label_name = props.get("labelName")
                direction = props.get("labelDirection")
                node_id = node.get("id")
                
                if not label_name or not direction:
                    orphaned_labels.append(f"Label node {node_id} missing name or direction")
                    continue
                
                if direction == "output":
                    # Store source connection info
                    source_node_id = props.get("sourceNodeId")
                    source_slot_index = props.get("sourceSlotIndex")
                    
                    if source_node_id is not None and source_slot_index is not None:
                        if label_name in output_labels:
                            raise ValueError(
                                f"Duplicate output label '{label_name}' found.\n"
                                f"Please remove one of the duplicate output labels."
                            )
                        output_labels[label_name] = {
                            "node_id": str(source_node_id),
                            "slot_index": source_slot_index,
                            "label_node_id": node_id
                        }
                        self.logger.debug(f"Found output label '{label_name}': node {source_node_id}, slot {source_slot_index}")
                    else:
                        orphaned_labels.append(
                            f"Output label '{label_name}' (node {node_id}) missing source connection info. "
                            f"This can happen if the source node was deleted. Please delete this orphaned label."
                        )
                
                elif direction == "input":
                    # Store target connection info
                    target_node_id = props.get("targetNodeId")
                    target_slot_index = props.get("targetSlotIndex")
                    connected_to = props.get("connectedToLabel")
                    
                    if target_node_id is not None and target_slot_index is not None and connected_to:
                        input_labels.append({
                            "node_id": str(target_node_id),
                            "slot_index": target_slot_index,
                            "connected_to": connected_to,
                            "label_node_id": node_id
                        })
                        self.logger.debug(f"Found input label connecting to '{connected_to}': node {target_node_id}, slot {target_slot_index}")
                    else:
                        orphaned_labels.append(
                            f"Input label '{label_name}' (node {node_id}) missing target connection info. "
                            f"This can happen if the target node was deleted. Please delete this orphaned label."
                        )
        
        # Check for orphaned labels before continuing
        if orphaned_labels:
            error_msg = "Found orphaned or invalid labels:\n" + "\n".join(f"  - {msg}" for msg in orphaned_labels)
            raise ValueError(error_msg)
        
        # Resolve connections and check for missing output labels
        connections = []
        missing_outputs = []
        
        for input_label in input_labels:
            output_info = output_labels.get(input_label["connected_to"])
            if output_info:
                connection = (
                    output_info["node_id"],
                    output_info["slot_index"],
                    input_label["node_id"],
                    input_label["slot_index"]
                )
                connections.append(connection)
                self.logger.debug(f"Resolved label connection: {output_info['node_id']}[{output_info['slot_index']}] -> {input_label['node_id']}[{input_label['slot_index']}]")
            else:
                missing_outputs.append(
                    f"Input label (node {input_label['label_node_id']}) references missing output label '{input_label['connected_to']}'. "
                    f"Please create the output label or delete this input label."
                )
        
        # Check for missing output labels
        if missing_outputs:
            error_msg = "Found input labels referencing missing output labels:\n" + "\n".join(f"  - {msg}" for msg in missing_outputs)
            raise ValueError(error_msg)
        
        return connections
    
    def generate_label_connections(self, workflow: Dict) -> Tuple[List[Tuple], Dict]:
        """Preprocess labels to generate implied connections using property-based resolution.
        
        Returns:
            - List of new connections as tuples: (from_node, from_slot, to_node, to_slot)
            - Dict with bidirectional lookups:
              - 'by_input': Maps "{to_node}_{to_slot}" to source info
              - 'by_output': Maps "{from_node}_{from_slot}" to destination info
        """
        label_connections_dict = {
            "by_input": {},   # For looking up what connects TO a node's input
            "by_output": {}   # For looking up what connects FROM a node's output
        }
        
        # Use dictionary-free approach (properties stored in Label nodes)
        connections = self._resolve_labels_from_nodes(workflow)
        
        if connections:
            self.logger.debug(f"Resolved {len(connections)} label connections from node properties")
            # Build the label_connections_dict from resolved connections
            for from_node_id, from_slot, to_node_id, to_slot in connections:
                # Store in both directions for bidirectional lookups
                label_connections_dict["by_input"][f"{to_node_id}_{to_slot}"] = {
                    "from_node": from_node_id,
                    "from_slot": from_slot,
                    "type": "*"  # Type is stored in node properties if needed
                }
                label_connections_dict["by_output"][f"{from_node_id}_{from_slot}"] = {
                    "to_node": to_node_id,
                    "to_slot": to_slot,
                    "type": "*"
                }
        else:
            self.logger.debug("No label connections found in workflow")
        
        return connections, label_connections_dict
    
    def _validate_workflow_integrity(self, nodes: List, links: List) -> None:
        """Validate that all links reference existing nodes. Fail fast on broken references."""
        # Build set of valid node IDs
        valid_node_ids = {str(node.get("id")) for node in nodes}
        
        # Check all links for broken references
        broken_links = []
        for link in links:
            if len(link) >= 5:
                link_id = link[0]
                from_node = str(link[1])
                to_node = str(link[3])
                
                missing_nodes = []
                if from_node not in valid_node_ids:
                    missing_nodes.append(f"source node {from_node}")
                if to_node not in valid_node_ids:
                    missing_nodes.append(f"target node {to_node}")
                
                if missing_nodes:
                    broken_links.append(f"Link {link_id}: references non-existent {' and '.join(missing_nodes)}")
        
        if broken_links:
            error_msg = "Export failed due to broken connections:\n"
            for broken_link in broken_links:
                error_msg += f"  • {broken_link}\n"
            error_msg += "\nPlease fix the workflow before exporting. You can use:\n"
            error_msg += "  python claude_scripts/analyze_workflow.py <workflow_name> --repair-workflow\n"
            error_msg += "to automatically remove broken connections."
            raise ValueError(error_msg)
    
    def export_workflow(self, workflow: Dict, output_path: Optional[Path] = None) -> str:
        """Convert workflow JSON to modular Python package"""
        # Clear the schema cache to prevent stale data from previous exports
        # The cache uses memory addresses as keys, which can be reused between exports
        ExportableNode._schema_cache.clear()
        
        # Validate that runner_args.json is up-to-date
        self._validate_runner_args_timestamps()
        
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])
        metadata = workflow.get("metadata", {})
        
        # Validate workflow integrity BEFORE any processing
        self._validate_workflow_integrity(nodes, links)
        
        # Generate label connections but DON'T add them to links yet
        # We need to fix corrupted slots first, then add label connections
        label_connections, _ = self.generate_label_connections(workflow)
        
        # WORKAROUND: Fix corrupted to_slot values by reading original JSON
        # ComfyUI pipeline corrupts all to_slot values to 0, so we restore them
        # Skip if metadata indicates this is a programmatically created workflow
        if not metadata.get("skip-slot-correction", False):
            links = self._fix_corrupted_slots(links, metadata)
        
        # NOW add the label connections after slot correction
        if label_connections:
            self.logger.debug(f"Adding {len(label_connections)} label-based connections to workflow")
            # Convert to link format and add to links
            max_link_id = max([link[0] for link in links] + [0])
            for i, (from_node, from_slot, to_node, to_slot) in enumerate(label_connections):
                # Create a new link entry
                new_link = [
                    max_link_id + i + 1,  # link_id
                    int(from_node),       # from_node
                    from_slot,            # from_slot  
                    int(to_node),         # to_node
                    to_slot,              # to_slot
                    "*"                   # type (will be resolved later)
                ]
                links.append(new_link)
                self.logger.debug(f"Added resolved label link: from node {from_node} slot {from_slot} to node {to_node} slot {to_slot}")
            
            # Remove all links to/from Label nodes now that we have resolved connections
            # This prevents follow_node_connection from finding Label nodes instead of the resolved targets
            label_node_ids = {str(node.get("id")) for node in nodes if node.get("type") == "Label"}
            if label_node_ids:
                original_link_count = len(links)
                links[:] = [link for link in links if len(link) >= 5 and 
                           str(link[1]) not in label_node_ids and  # from_node not a Label
                           str(link[3]) not in label_node_ids]     # to_node not a Label
                removed_count = original_link_count - len(links)
                if removed_count > 0:
                    self.logger.debug(f"Removed {removed_count} links to/from Label nodes after resolution")
        
        # Set export context for utility functions
        export_utils.set_export_context({
            'nodes': nodes,
            'links': links,
            'node_registry': self.node_registry
        })
        
        if output_path:
            output_path = Path(output_path)
            # For modular export, output_path should be a directory, not a file
            if output_path.suffix == '.py':
                output_path = output_path.parent
                
            # Validate export path - must be within export_system/exports/
            export_base = Path(__file__).parent / "exports"
            try:
                # Resolve both paths to handle relative paths correctly
                output_resolved = output_path.resolve()
                export_base_resolved = export_base.resolve()
                
                # Check if output path is within the exports directory
                output_resolved.relative_to(export_base_resolved)
            except ValueError:
                # Path is not relative to export base - raise clear error
                raise ValueError(
                    f"Export path must be within 'export_system/exports/' directory.\n"
                    f"Attempted path: {output_path}\n"
                    f"Required base path: {export_base}\n"
                    f"Example correct usage: export_workflow(workflow, Path('export_system/exports/MyWorkflow'))"
                )
        else:
            # Default to exports directory with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workflow_name = metadata.get("workflow_name", "unnamed")
            output_path = Path(__file__).parent / "exports" / f"{workflow_name}_{timestamp}"
        
        # Wrap entire export process in try-except to clean up on failure
        export_successful = False
        try:
            # Create package structure
            framework_dir, nodes_dir = self._create_package_structure(output_path)
            
            # Create metadata.json with workflow information
            import hashlib
            from datetime import datetime
            
            # Generate workflow ID from content hash
            workflow_json = json.dumps(workflow, sort_keys=True)
            content_hash = hashlib.sha256(workflow_json.encode()).hexdigest()[:12]
            workflow_id = f"wf_{content_hash}"
            
            # Extract workflow name (from path or metadata)
            workflow_name = output_path.name if output_path.name else metadata.get("workflow_name", "unnamed")
            
            metadata_content = {
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "export_timestamp": datetime.now().isoformat(),
                "node_count": len(nodes),
                "link_count": len(links),
                "framework_version": "1.0.0",
                "exported_by": "DNNE Export System"
            }
            
            metadata_path = output_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_content, f, indent=2)
            
            self.logger.debug(f"Created metadata.json with workflow_id: {workflow_id}, name: {workflow_name}")
            
            # Export framework
            self._export_framework(framework_dir)
            
            # Track node information for __init__.py generation
            node_classes = []
            node_instances = []
            virtual_nodes = {}  # Track virtual nodes for connection handling
            
            # First pass: identify virtual nodes
            for node in nodes:
                node_id = str(node["id"])
                node_type = node.get("class_type") or node.get("type")
                
                if self._is_virtual_node(node_type):
                    virtual_nodes[node_id] = node
            
            for node in nodes:
                node_id = str(node["id"])
                node_type = node.get("class_type") or node.get("type")
                
                # Skip virtual nodes (configuration-only nodes)
                if self._is_virtual_node(node_type):
                    self.logger.debug(f"Skipping virtual node {node_id} ({node_type}) - configuration only")
                    continue
                
                if node_type in self.node_registry:
                    node_class = self.node_registry[node_type]
                    
                    # Get template and prepare variables
                    template_name = node_class.get_template_name()
                    template_vars = node_class.prepare_template_vars_with_validation(
                        node_id, node, self._get_node_connections(node_id, links, nodes), 
                        self.node_registry, nodes, links
                    )
                    
                    # Load and process template
                    template_content = self._load_template(template_name)
                    node_code = self._process_template(template_content, template_vars)
                    
                    # Get node-specific imports
                    node_imports = list(node_class.get_imports())
                    
                    # Export node to file and get class name
                    class_name = self._export_node_to_file(nodes_dir, node_id, node_type, node_code, node_imports)
                    node_classes.append((node_id, node_type, class_name))
                    
                    # Create instance
                    instance_name = f"node_{node_id}"
                    
                    # Check if node has custom instance code
                    if hasattr(node_class, 'get_instance_code'):
                        node_connections = self._get_node_connections(node_id, links, nodes)
                        instance_code = node_class.get_instance_code(node_id, node, node_connections)
                        node_instances.append(instance_code)
                    else:
                        node_instances.append(f'{instance_name} = {class_name}("{node_id}")')
                    
                else:
                    self.logger.warning(f"Unknown node type: {node_type}")
                    # Generate placeholder
                    placeholder_code = self._generate_placeholder_node(node_id, node_type)
                    class_name = f"PlaceholderNode_{node_id}"
                    
                    # Export placeholder to file
                    self._export_node_to_file(nodes_dir, node_id, node_type, placeholder_code, [])
                    node_classes.append((node_id, node_type, class_name))
                    node_instances.append(f'node_{node_id} = {class_name}("{node_id}")')
            
            # Collect and process file copy requests from all nodes
            self._process_file_copy_requests(nodes, output_path)
        
            # Generate nodes/__init__.py
            self._generate_node_init(nodes_dir, node_classes)
            
            # Generate connections
            connections = self._generate_connections(links, nodes)
            
            # Generate minimal runner.py
            self._generate_minimal_runner(output_path, node_instances, connections, nodes, metadata)
            
            # Mark export as successful
            export_successful = True
            
            self.logger.info(f"Exported modular package to: {output_path}")
            
        except Exception as e:
            # Clean up partial export on failure
            if output_path.exists():
                import shutil
                self.logger.error(f"Export failed, cleaning up partial export at: {output_path}")
                shutil.rmtree(output_path)
            
            # Clear export context even on failure
            export_utils.clear_export_context()
            
            # Re-raise the exception with additional context if needed
            if "non-existent" in str(e):
                # Already has a good error message
                raise
            else:
                # Add context to other errors
                raise RuntimeError(f"Export failed: {e}") from e
        
        finally:
            # Always clear export context
            export_utils.clear_export_context()
        
        # Return the path to the runner for backward compatibility
        return str(output_path / "runner.py")
    
    def _fix_corrupted_slots(self, links: List, workflow_metadata: Dict = None) -> List:
        """WORKAROUND: Fix to_slot values corrupted by ComfyUI pipeline"""
        try:
            # Try to find the correct workflow JSON file
            from pathlib import Path
            import json
            
            # Try to get workflow name from metadata
            # Check if slot correction should be skipped first
            if workflow_metadata and workflow_metadata.get("skip-slot-correction"):
                return links
            
            workflow_name = None
            if workflow_metadata:
                workflow_name = workflow_metadata.get("workflow_name")
            
            # Require workflow name to be specified
            if not workflow_name:
                raise ValueError("Workflow name is required for slot correction")
            
            # Only try the specific workflow file - no fallbacks
            workflow_dir = Path("user/default/workflows")
            workflow_path = workflow_dir / f"{workflow_name}.json"
            
            if not workflow_path.exists():
                raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
            
            self.logger.debug(f"Reading original workflow from: {workflow_path}")
            with open(workflow_path, 'r') as f:
                original_workflow = json.load(f)
            original_links = original_workflow.get("links", [])
            
            # Create a mapping from connection pattern to correct to_slot value
            # Use (from_node, from_slot, to_node) as key since link IDs might not match
            slot_corrections = {}
            for link in original_links:
                if len(link) >= 5:
                    from_node, from_slot, to_node, to_slot = str(link[1]), link[2], str(link[3]), link[4]
                    connection_key = (from_node, from_slot, to_node)
                    slot_corrections[connection_key] = to_slot
            
            # Apply corrections to the corrupted links
            fixed_links = []
            fixes_applied = 0
            for link in links:
                if len(link) >= 5:
                    link_id = link[0]
                    from_node, from_slot, to_node, corrupted_to_slot = str(link[1]), link[2], str(link[3]), link[4]
                    connection_key = (from_node, from_slot, to_node)
                    
                    if connection_key in slot_corrections:
                        correct_to_slot = slot_corrections[connection_key]
                        if corrupted_to_slot != correct_to_slot:
                            # Fix the to_slot value
                            fixed_link = list(link)
                            fixed_link[4] = correct_to_slot
                            fixed_links.append(fixed_link)
                            fixes_applied += 1
                            self.logger.debug(f"Fixed connection {from_node}.{from_slot}→{to_node}: to_slot {corrupted_to_slot} → {correct_to_slot}")
                        else:
                            fixed_links.append(link)
                    else:
                        # Connection not found in original - this is an error
                        raise ValueError(
                            f"Connection {from_node}.{from_slot}→{to_node} not found in original workflow. "
                            f"The workflow may have been modified or corrupted."
                        )
                else:
                    fixed_links.append(link)
            
            self.logger.debug(f"Applied {fixes_applied} slot corrections from {workflow_path}")
            return fixed_links
                
        except Exception as e:
            self.logger.error(f"Failed to fix corrupted slots: {e}")
            raise RuntimeError(f"Cannot export workflow without fixing corrupted slots: {e}")
    
    def _load_template(self, template_name: str) -> str:
        """Load template file content"""
        template_path = self.templates_dir / template_name
        
        # If file doesn't exist and ends with .py, try .tpl
        if not template_path.exists() and template_name.endswith('.py'):
            tpl_name = template_name[:-3] + '.tpl'
            template_path = self.templates_dir / tpl_name
            
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return template_path.read_text(encoding='utf-8')
    
    
    def _process_file_copy_requests(self, nodes: List[Dict], output_path: Path):
        """Collect and process file copy requests from all nodes with collision detection."""
        import shutil
        import os
        from collections import defaultdict
        
        # Accumulate all file copy requests
        file_requests = []  # List of (node_id, src_path, dest_dir)
        
        for node in nodes:
            node_id = str(node["id"])
            node_type = node.get("class_type") or node.get("type")
            
            # Skip virtual nodes
            if self._is_virtual_node(node_type):
                continue
            
            if node_type in self.node_registry:
                node_class = self.node_registry[node_type]
                
                # Check if node has files to export
                if hasattr(node_class, 'get_export_files'):
                    files_to_copy = node_class.get_export_files(node_id, node)
                    for src_path, dest_dir in files_to_copy:
                        file_requests.append((node_id, src_path, dest_dir))
        
        # If no files to copy, we're done
        if not file_requests:
            return
        
        # Build collision detection map: destination file -> (node_id, source_path)
        dest_map = {}  # Maps final destination paths to (node_id, src_path)
        
        for node_id, src_path, dest_dir in file_requests:
            # Resolve source path
            src_path = Path(src_path)
            
            # Validate source exists
            if not src_path.exists():
                raise FileNotFoundError(
                    f"DataStreamer node {node_id}: Source path does not exist: {src_path}"
                )
            
            # Validate dest_dir is relative
            if os.path.isabs(dest_dir):
                raise ValueError(
                    f"DataStreamer node {node_id}: dest_dir must be a relative path, got: {dest_dir}"
                )
            
            # Calculate destination path
            dest_base = output_path / dest_dir if dest_dir != "." else output_path
            
            # Determine what files will be created
            if src_path.is_file():
                # Single file will be copied
                final_dest = dest_base / src_path.name
                
                # Check for collision
                if final_dest in dest_map:
                    other_node_id, other_src = dest_map[final_dest]
                    if other_src != src_path:  # Different sources to same destination
                        raise ValueError(
                            f"File collision detected: Multiple nodes trying to write to '{final_dest.relative_to(output_path)}':\n"
                            f"  - Node {other_node_id}: copying from {other_src}\n"
                            f"  - Node {node_id}: copying from {src_path}"
                        )
                else:
                    dest_map[final_dest] = (node_id, src_path)
            
            elif src_path.is_dir():
                # Directory tree will be copied
                # We need to check all files that would be created
                for root, dirs, files in os.walk(src_path):
                    rel_root = Path(root).relative_to(src_path.parent)
                    for file in files:
                        src_file = Path(root) / file
                        final_dest = dest_base / rel_root / file
                        
                        # Check for collision
                        if final_dest in dest_map:
                            other_node_id, other_src = dest_map[final_dest]
                            if other_src != src_file:  # Different sources to same destination
                                raise ValueError(
                                    f"File collision detected: Multiple nodes trying to write to '{final_dest.relative_to(output_path)}':\n"
                                    f"  - Node {other_node_id}: copying from {other_src}\n"
                                    f"  - Node {node_id}: copying from {src_file}"
                                )
                        else:
                            dest_map[final_dest] = (node_id, src_file)
        
        # All validation passed, now perform the copies
        for node_id, src_path, dest_dir in file_requests:
            src_path = Path(src_path)
            dest_base = output_path / dest_dir if dest_dir != "." else output_path
            
            # Create destination directory if needed
            dest_base.mkdir(parents=True, exist_ok=True)
            
            if src_path.is_file():
                # Copy single file
                dest_file = dest_base / src_path.name
                shutil.copy2(src_path, dest_file)
                self.logger.debug(f"Copied file: {src_path} -> {dest_file.relative_to(output_path)}")
            
            elif src_path.is_dir():
                # Copy directory tree
                dest_subdir = dest_base / src_path.name
                if dest_subdir.exists():
                    shutil.rmtree(dest_subdir)  # Remove existing to ensure clean copy
                shutil.copytree(src_path, dest_subdir)
                self.logger.debug(f"Copied directory: {src_path} -> {dest_subdir.relative_to(output_path)}")
    
    def _process_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Process template by replacing variables"""
        # First, handle the double-brace problem in f-strings
        # Replace {{ and }} with temporary placeholders
        template = template.replace('{{', '__DOUBLE_OPEN_BRACE__')
        template = template.replace('}}', '__DOUBLE_CLOSE_BRACE__')
        
        # Replace template variables
        for key, value in variables.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        # Restore the f-string braces
        template = template.replace('__DOUBLE_OPEN_BRACE__', '{')
        template = template.replace('__DOUBLE_CLOSE_BRACE__', '}')
        
        # Remove template_vars declaration section
        lines = template.split('\n')
        processed_lines = []
        skip_template_vars = False
        brace_count = 0
        
        for line in lines:
            if line.strip().startswith('template_vars = {'):
                skip_template_vars = True
                brace_count = 1
                continue
            elif skip_template_vars:
                # Count braces to handle multi-line dicts
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    skip_template_vars = False
                continue
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _get_node_connections(self, node_id: str, links: List, nodes: List) -> Dict:
        """Get incoming and outgoing connections for a node"""
        connections = {
            "inputs": {},
            "outputs": {}
        }
        
        # Get the node type and class to map slot numbers to names
        node_data = None
        node_type = None
        for node in nodes:
            if str(node["id"]) == node_id:
                node_data = node
                node_type = node.get("class_type") or node.get("type")
                break
        
        node_class = self.node_registry.get(node_type) if node_type else None
        
        # Debug logging for Network node
        if node_type == "NetworkNode":
            self.logger.debug(f"Getting connections for Network node {node_id}")
            self.logger.debug(f"Total links to process: {len(links)}")
        
        for link in links:
            # Link format: [link_id, from_node, from_slot, to_node, to_slot]
            if len(link) >= 5:
                from_node = str(link[1])
                to_node = str(link[3])
                
                # Skip connections from/to external nodes (UI-created) but keep other virtual nodes
                # Network nodes need to see LinearLayer connections to build the network
                from_node_data = next((n for n in nodes if str(n["id"]) == from_node), None)
                to_node_data = next((n for n in nodes if str(n["id"]) == to_node), None)
                
                from_node_type = (from_node_data.get("type") or from_node_data.get("class_type")) if from_node_data else None
                to_node_type = (to_node_data.get("type") or to_node_data.get("class_type")) if to_node_data else None
                
                # Debug logging for Network node
                if node_type == "NetworkNode" and (to_node == node_id or from_node == node_id):
                    self.logger.debug(f"  Link: from {from_node}({from_node_type}) to {to_node}({to_node_type})")
                
                if from_node_type and export_utils.is_external_node(from_node_type):
                    if node_type == "NetworkNode" and to_node == node_id:
                        self.logger.debug(f"    Skipping link from external node {from_node_type}")
                    continue
                if to_node_type and export_utils.is_external_node(to_node_type):
                    if node_type == "NetworkNode" and from_node == node_id:
                        self.logger.debug(f"    Skipping link to external node {to_node_type}")
                    continue
                
                if to_node == node_id:
                    # Incoming connection - map slot number to input name
                    to_slot = link[4]
                    input_name = None
                    
                    if node_class and hasattr(node_class, 'get_input_name_for_slot'):
                        input_name = node_class.get_input_name_for_slot(to_slot)
                    elif node_class and hasattr(node_class, 'get_input_names'):
                        input_names = node_class.get_input_names()
                        if to_slot < len(input_names):
                            input_name = input_names[to_slot]
                    
                    # Error if name mapping fails
                    if input_name is None:
                        raise ValueError(f"Cannot map input slot {to_slot} to input name for node {node_id} of type {node_type}")
                    
                    # Support multiple connections per input
                    # Store as a list of connections
                    if input_name not in connections["inputs"]:
                        connections["inputs"][input_name] = []
                    
                    connections["inputs"][input_name].append({
                        "from_node": from_node,
                        "from_slot": link[2]
                    })
                    
                    # Debug logging for Network node
                    if node_type == "NetworkNode":
                        self.logger.debug(f"    Added input connection: {input_name} from node {from_node}")
                elif from_node == node_id:
                    # Outgoing connection
                    if link[2] not in connections["outputs"]:
                        connections["outputs"][link[2]] = []
                    connections["outputs"][link[2]].append({
                        "to_node": to_node,
                        "to_slot": link[4]
                    })
        
        # Debug logging for Network node
        if node_type == "NetworkNode":
            self.logger.debug(f"Final connections for Network node {node_id}:")
            self.logger.debug(f"  Inputs: {list(connections['inputs'].keys())}")
            self.logger.debug(f"  Outputs: {list(connections['outputs'].keys())}")
        
        return connections
    
    def _generate_connections(self, links: List, nodes: List) -> List[str]:
        """Generate connection tuples for wire_nodes, including label-based connections"""
        connections = []
        
        # Identify virtual nodes that will be skipped
        virtual_nodes = set()
        for node in nodes:
            node_id = str(node["id"])
            node_type = node.get("class_type") or node.get("type")
            if self._is_virtual_node(node_type):
                virtual_nodes.add(node_id)
        
        # Build a map of node_id to node_type and exporter class
        node_info = {}
        for node in nodes:
            node_id = str(node["id"])
            node_type = node.get("class_type") or node.get("type")
            node_class = self.node_registry.get(node_type)
            node_info[node_id] = {
                "type": node_type,
                "class": node_class,
                "outputs": node_class.get_output_names() if node_class else [f"output_{i}" for i in range(3)],
                "inputs": node_class.get_input_names() if node_class else []
            }
        
        for link in links:
            if len(link) >= 5:
                from_node = str(link[1])
                from_slot = link[2]
                to_node = str(link[3])
                to_slot = link[4]
                
                # Skip connections to/from virtual nodes
                if from_node in virtual_nodes or to_node in virtual_nodes:
                    self.logger.debug(f"Skipping connection from {from_node} to {to_node} - involves virtual node")
                    continue
                
                # Get actual output name
                if from_node not in node_info:
                    # This should never happen after validation, but fail fast if it does
                    raise ValueError(
                        f"Export failed: Link {link[0] if len(link) > 0 else 'unknown'} references "
                        f"non-existent source node {from_node}. This indicates a workflow integrity issue.\n"
                        f"Please run: python claude_scripts/analyze_workflow.py <workflow_name> --repair-workflow"
                    )
                    
                outputs = node_info[from_node]["outputs"]
                output_name = outputs[from_slot] if from_slot < len(outputs) else f"output_{from_slot}"
                
                # Get input name
                if to_node not in node_info:
                    # This should never happen after validation, but fail fast if it does
                    raise ValueError(
                        f"Export failed: Link {link[0] if len(link) > 0 else 'unknown'} references "
                        f"non-existent target node {to_node}. This indicates a workflow integrity issue.\n"
                        f"Please run: python claude_scripts/analyze_workflow.py <workflow_name> --repair-workflow"
                    )
                    
                to_node_class = node_info[to_node]["class"]
                if to_node_class and hasattr(to_node_class, 'get_input_names'):
                    input_names = to_node_class.get_input_names()
                    if to_slot < len(input_names):
                        input_name = input_names[to_slot]
                    else:
                        raise ValueError(f"Input slot {to_slot} out of range for node {to_node} of type {node_info[to_node]['type']}. Available inputs: {input_names}")
                else:
                    raise ValueError(f"Cannot determine input name for slot {to_slot} on node {to_node} of type {node_info[to_node]['type']}")
                
                # Check if this is a virtual connection
                from_node_type = node_info[from_node]["type"]
                to_node_type = node_info[to_node]["type"]
                
                # Check if output is virtual
                if self._is_virtual_output(from_node_type, from_slot):
                    self.logger.debug(f"Skipping virtual output connection: {from_node_type}[{from_slot}] -> {to_node_type}")
                    continue
                
                # Check if input is virtual
                if self._is_virtual_input(to_node_type, input_name):
                    self.logger.debug(f"Skipping virtual input connection: {from_node_type} -> {to_node_type}[{input_name}]")
                    continue
                
                connections.append(
                    f'("{from_node}", "{output_name}", "{to_node}", "{input_name}")'
                )
        
        return connections
    
    
    def _generate_placeholder_node(self, node_id: str, node_type: str) -> str:
        """Generate placeholder for unknown node types"""
        return f'''
class PlaceholderNode_{node_id}(QueueNode):
    """Placeholder for {node_type} node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input_0"])
        self.setup_outputs(["output_0"])
        self.logger.warning(f"Using placeholder for unknown node type: {node_type}")
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        self.logger.debug(f"Placeholder compute for {node_type}")
        return {{"output_0": inputs.get("input_0", None)}}
'''
    
    
    def _assemble_script(self, imports: List[str], base_framework: str,
                        node_implementations: List[str], node_instances: List[str],
                        connections: List[str], metadata: Dict) -> str:
        """Assemble the complete script"""
        script_parts = []
        
        # Header with shebang first
        script_parts.extend([
            "#!/usr/bin/env python3",
            '"""',
            "Generated by DNNE Queue-Based Export System",
            f"Metadata: {json.dumps(metadata, indent=2) if metadata else 'None'}",
            '"""',
            "",
            "# Imports",
        ])
        
        # Add imports
        script_parts.extend(imports)
        
        # Add framework
        script_parts.extend([
            "",
            "# " + "=" * 78,
            "# Queue Framework",
            "# " + "=" * 78,
            base_framework,
            "",
            "# " + "=" * 78,
            "# Node Implementations",
            "# " + "=" * 78,
        ])
        
        # Add node implementations
        for impl in node_implementations:
            script_parts.append(impl)
            script_parts.append("")
        
        # Add main function
        script_parts.extend([
            "# " + "=" * 78,
            "# Main Execution",
            "# " + "=" * 78,
            "",
            "async def main():",
            '    """Main execution function"""',
            '    print("🚀 Starting DNNE Queue-Based Execution")',
            '    print("=" * 60)',
            "",
            "    # Create nodes",
        ])
        
        # Add node instances
        for instance in node_instances:
            script_parts.append(f"    {instance}")
        
        script_parts.extend([
            "",
            "    # Create runner",
            "    runner = GraphRunner()",
            "",
            "    # Add nodes to runner",
        ])
        
        # Add nodes to runner
        for instance in node_instances:
            node_var = instance.split(" = ")[0].strip()
            script_parts.append(f"    runner.add_node({node_var})")
        
        script_parts.extend([
            "",
            "    # Wire connections",
            "    connections = [",
        ])
        
        # Add connections
        for conn in connections:
            script_parts.append(f"        {conn},")
        
        script_parts.extend([
            "    ]",
            "    runner.wire_nodes(connections)",
            "",
            "    # Run the graph",
            "    try:",
            "        # Run indefinitely (Ctrl+C to stop)",
            "        await runner.run()",
            "        # Or run for specific duration:",
            "        # await runner.run(duration=10.0)  # Run for 10 seconds",
            "    except KeyboardInterrupt:",
            "        print('\\n🛑 Stopped by user')",
            "",
            "    # Show final statistics",
            "    print('\\n📊 Final Statistics:')",
            "    stats = runner.get_stats()",
            "    for node_id, node_stats in stats.items():",
            "        print(f'  {node_id}: {node_stats[\"compute_count\"]} computations, '",
            "              f'avg time: {node_stats[\"last_compute_time\"]:.3f}s')",
            "",
            "    # Show concurrency report if PPO nodes detected",
            "    if any('PPOAgent' in str(type(node).__name__) for node in runner.nodes.values()):",
            "        g.print_concurrency_report()",
            "",
            "",
            "if __name__ == '__main__':",
            "    asyncio.run(main())",
        ])
        
        return "\n".join(script_parts)
    
    def _create_package_structure(self, output_path: Path):
        """Create the package directory structure"""
        # Clean existing directory if it exists
        if output_path.exists():
            import shutil
            
            # Safety check - only delete if it looks like an export directory
            # Check for runner.py or nodes/ directory as indicators
            is_export_dir = (
                (output_path / "runner.py").exists() or
                (output_path / "nodes").exists() or
                (output_path / "framework").exists()
            )
            
            if is_export_dir:
                self.logger.debug(f"Cleaning existing export directory: {output_path}")
                shutil.rmtree(output_path)
            else:
                # Directory exists but doesn't look like an export - be cautious
                self.logger.warning(f"Directory {output_path} exists but doesn't appear to be an export directory")
                raise ValueError(
                    f"Target directory exists but doesn't appear to be a DNNE export: {output_path}\n"
                    f"Please choose a different directory or manually remove the existing one."
                )
        
        # Create main directories
        output_path.mkdir(parents=True, exist_ok=True)
        framework_dir = output_path / "framework"
        framework_dir.mkdir(exist_ok=True)
        nodes_dir = output_path / "nodes"
        nodes_dir.mkdir(exist_ok=True)
        
        # Create __init__.py files
        (output_path / "__init__.py").write_text("# DNNE Generated Package\n", encoding='utf-8')
        
        return framework_dir, nodes_dir
    
    
    def _export_framework(self, framework_dir: Path):
        """Export the queue framework components to framework/"""
        
        # Export framework __init__.py
        framework_init = self._load_template("framework/__init__.py")
        (framework_dir / "__init__.py").write_text(framework_init, encoding='utf-8')
        
        # Define framework files to export
        # Format: (template_path, output_name, description, required)
        framework_files = [
            ("framework/exceptions.py", "exceptions.py", None, True),
            ("framework/base_nodes.py", "base_nodes.py", None, True),
            ("framework/graph_runner.py", "graph_runner.py", None, True),
            ("framework/checkpoint.py", "checkpoint.py", None, True),
            ("framework/globals.py", "globals.py", None, True),
            ("framework/globals_threadsafe.py", "globals_threadsafe.py", "thread-safe yielding support", True),
            ("framework/dnne_exceptions.py", "dnne_exceptions.py", None, True),
            ("framework/multi_waiter.py", "multi_waiter.py", "efficient async input handling", True),
            ("framework/override_parser.py", "override_parser.py", "runtime parameter overrides", True),
            ("framework/arg_parser.tpl", "arg_parser.py", "command-line argument parsing", True),
            ("framework/telemetry.py", "telemetry.py", "telemetry support", True),
            ("framework/deadlock_utils.py", "deadlock_utils.py", "deadlock debugging", True),
            ("framework/logging_utils.py", "logging_utils.py", "relative time logging", True),
            ("framework/time_utils.py", "time_utils.py", "time duration parsing", True),
        ]
        
        # Export all framework files
        for template_path, output_name, description, required in framework_files:
            try:
                content = self._load_template(template_path)
                (framework_dir / output_name).write_text(content, encoding='utf-8')
                if description:
                    self.logger.debug(f"Exported {output_name} for {description}")
            except FileNotFoundError:
                if required:
                    self.logger.error(f"{template_path} not found in templates")
                    raise FileNotFoundError(f"{template_path} not found in templates")
                else:
                    self.logger.warning(f"Optional file {template_path} not found, skipping")
        
        # Copy dnne_config.py and dnne_config.json from root
        import shutil
        dnne_root = Path(__file__).parent.parent
        
        # Create a custom dnne_config.py that loads from exported_config.json
        dnne_config_content = '''#!/usr/bin/env python3
"""
DNNE Configuration Module (Exported Version)
Loads configuration from exported_config.json
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class DNNEConfig:
    """Manages DNNE configuration for exported workflows"""
    
    def __init__(self):
        self._config = {}
        self._config_file = None
        self.load_config()
    
    def load_config(self):
        """Load configuration from exported_config.json"""
        # Try environment variable first
        env_config_path = os.environ.get('DNNE_CONFIG_PATH')
        if env_config_path and os.path.exists(env_config_path):
            self._config_file = env_config_path
            self._load_from_file(env_config_path)
            return
        
        # Try project root (where this file is located)
        project_config = Path(__file__).parent / 'exported_config.json'
        if project_config.exists():
            self._config_file = str(project_config)
            self._load_from_file(project_config)
            return
        
        # If no config found, raise error
        raise FileNotFoundError(
            "No exported_config.json found. This file should be created during export."
        )
    
    def _load_from_file(self, config_path: Path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                self._config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    
    def _convert_path_for_os(self, path: str) -> str:
        """Convert path based on current OS"""
        # For exported code, just expand ~
        if path.startswith('~'):
            path = os.path.expanduser(path)
        return path
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_path(self, path_key: str) -> str:
        """Get a path from the paths section"""
        raw_path = self.get(f'paths.{path_key}', '')
        return self._convert_path_for_os(raw_path)
    
    def get_conda_activate_command(self) -> str:
        """Get the conda activation command"""
        conda_path = self.get('conda.conda_path', '')
        conda_env = self.get('conda.conda_env', '')
        
        if conda_path and conda_env:
            conda_path = self._convert_path_for_os(conda_path)
            return f"source {conda_path}/bin/activate {conda_env}"
        return ""
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory"""
        return Path(self.get('paths.temp_directory', '/tmp'))
    
    def get_all_paths(self) -> Dict[str, str]:
        """Get all configured paths"""
        return self.get('paths', {})


# Global configuration instance
config = DNNEConfig()


# Convenience functions
def get_linux_support_path() -> Path:
    """Get Linux support directory"""
    return Path(config.get_path('linux_support'))


def get_isaac_gym_path() -> Path:
    """Get Isaac Gym directory"""
    linux_support = get_linux_support_path()
    subdir = config.get('linux_support_subdirs.isaac_gym', 'isaacgym')
    return linux_support / subdir


def get_isaac_gym_envs_path() -> Path:
    """Get IsaacGymEnvs directory"""
    linux_support = get_linux_support_path()
    subdir = config.get('linux_support_subdirs.isaac_gym_envs', 'IsaacGymEnvs')
    return linux_support / subdir


def get_rl_games_path() -> Path:
    """Get rl_games_dnne directory"""
    linux_support = get_linux_support_path()
    subdir = config.get('linux_support_subdirs.rl_games_dnne', 'rl_games_dnne')
    return linux_support / subdir
'''
        
        # Write the custom dnne_config.py
        with open(framework_dir / "dnne_config.py", 'w') as f:
            f.write(dnne_config_content)
        
        self.logger.debug("Created custom dnne_config.py for exported workflows")
        
        # Create exported_config.json with only exported and shared sections
        dnne_config_json_src = dnne_root / "dnne_config.json"
        if not dnne_config_json_src.exists():
            raise FileNotFoundError(f"dnne_config.json not found at {dnne_config_json_src}")
        
        # Load full config and extract exported/shared sections
        import json
        with open(dnne_config_json_src, 'r') as f:
            full_config = json.load(f)
        
        # Create exported config with only safe sections
        exported_config = {}
        
        # Add exported section (flattened)
        if 'exported' in full_config:
            for key, value in full_config['exported'].items():
                exported_config[key] = value
        
        # Add shared section
        if 'shared' in full_config:
            for key, value in full_config['shared'].items():
                exported_config[key] = value
        
        # Write exported config
        exported_config_path = framework_dir / "exported_config.json"
        with open(exported_config_path, 'w') as f:
            json.dump(exported_config, f, indent=2)
        
        self.logger.debug("Created exported_config.json with client-safe configuration")
    
    def _export_node_to_file(self, nodes_dir: Path, node_id: str, node_type: str, 
                            node_code: str, node_imports: List[str]) -> str:
        """Export a single node to its own file and return the class name"""
        # Extract class name from the node code first
        import re
        class_match = re.search(r'class ([a-zA-Z_][a-zA-Z0-9_-]*)\(', node_code)
        if not class_match:
            raise ValueError(f"Could not extract class name from node {node_id}")
        class_name = class_match.group(1)
        
        # Generate filename using centralized utility function
        filename_base = self.classname_to_exported_filename(class_name)
        filename = f"{filename_base}.py"
        
        # Prepare the file content
        file_content = []
        
        # Check if node code uses Dict, Any, asyncio, or time and add necessary imports
        code_needs_dict_any = 'Dict[' in node_code or 'Any]' in node_code or '-> Dict' in node_code
        code_needs_asyncio = 'asyncio.' in node_code or 'await asyncio' in node_code
        code_needs_time = 'time.time()' in node_code or 'time.sleep' in node_code
        
        # Add standard imports first
        if code_needs_asyncio:
            file_content.append("import asyncio")
        if code_needs_time:
            file_content.append("import time")
        if code_needs_dict_any:
            file_content.append("from typing import Dict, Any")
        
        # Add node-specific imports
        file_content.extend(node_imports)
        file_content.append("from framework import QueueNode, SensorNode")
        file_content.append("")
        
        # Add the node implementation (without template_vars section)
        lines = node_code.split('\n')
        skip_template_vars = False
        brace_count = 0
        
        for line in lines:
            if line.strip().startswith('template_vars = {'):
                skip_template_vars = True
                brace_count = 1
                continue
            elif skip_template_vars:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    skip_template_vars = False
                continue
            else:
                file_content.append(line)
        
        # Write the file
        (nodes_dir / filename).write_text("\n".join(file_content).strip() + "\n", encoding='utf-8')
        
        return class_name
    
    def _generate_node_init(self, nodes_dir: Path, node_classes: List[tuple]):
        """Generate nodes/__init__.py with all node imports"""
        init_content = ['"""DNNE Generated Nodes"""', ""]
        
        for node_id, node_type, class_name in node_classes:
            # Use centralized utility for consistent filename generation
            filename = self.classname_to_exported_filename(class_name)
            init_content.append(f"from .{filename} import {class_name}")
        
        init_content.extend([
            "",
            "__all__ = [",
        ])
        
        for _, _, class_name in node_classes:
            init_content.append(f'    "{class_name}",')
        
        init_content.append("]")
        
        (nodes_dir / "__init__.py").write_text("\n".join(init_content) + "\n", encoding='utf-8')
    
    def _generate_node_imports_section(self, node_imports: List[str]) -> str:
        """Generate the node imports section for the template"""
        lines = []
        
        if node_imports:
            lines.append("# Import required nodes")
            for node_class in node_imports:
                # Convert class name to module name using centralized utility
                module_name = self.classname_to_exported_filename(node_class)
                lines.append(f"from nodes.{module_name} import {node_class}")
        
        return "\n".join(lines)
    
    def _generate_node_instances_section(self, node_instances: List[str]) -> str:
        """Generate the node instances section for the template"""
        lines = []
        for instance in node_instances:
            lines.append(f"    {instance}")
        return "\n".join(lines)
    
    def _generate_node_dictionary_section(self, node_instances: List[str]) -> str:
        """Generate the node dictionary section for the template"""
        lines = []
        for instance in node_instances:
            node_var = instance.split(" = ")[0].strip()
            node_id = instance.split('("')[1].split('")')[0]  # Extract node ID from instantiation
            lines.append(f'        "{node_id}": {node_var},')
        return "\n".join(lines)
    
    def _generate_add_nodes_to_runner_section(self, node_instances: List[str]) -> str:
        """Generate the add nodes to runner section for the template"""
        lines = []
        for instance in node_instances:
            node_var = instance.split(" = ")[0].strip()
            lines.append(f"    runner.add_node({node_var})")
        return "\n".join(lines)
    
    def _generate_connections_section(self, connections: List[str]) -> str:
        """Generate the connections section for the template"""
        lines = []
        for conn in connections:
            lines.append(f"        {conn},")
        return "\n".join(lines)
    
    def _generate_workflow_nodes_info(self, nodes: List[Dict]) -> str:
        """Generate the workflow nodes info section for the template including subsystem info"""
        lines = []
        for node in nodes:
            node_id = str(node["id"])
            node_type = node.get("class_type") or node.get("type")
            
            # Skip virtual nodes
            if self._is_virtual_node(node_type):
                continue
                
            # Extract just the base node type (remove Node suffix if present)
            if node_type.endswith("Node"):
                base_type = node_type
            else:
                base_type = node_type + "Node"
            
            # Get subsystem for this node if available
            subsystem = "unknown"
            if node_type in self.node_registry:
                exporter_class = self.node_registry[node_type]
                try:
                    node_subsystems = exporter_class.get_subsystem()
                    # Handle both single subsystem and list of subsystems
                    if isinstance(node_subsystems, list):
                        subsystem = node_subsystems[0] if node_subsystems else "unknown"
                    else:
                        subsystem = node_subsystems
                except (AttributeError, NotImplementedError):
                    subsystem = "unknown"
                
            lines.append(f'        "{node_id}": {{"type": "{base_type}", "subsystem": "{subsystem}"}},')
        
        # Remove trailing comma from last line
        if lines:
            lines[-1] = lines[-1].rstrip(',')
            
        return "\n".join(lines)
    
    def _generate_subsystem_mapping(self, nodes: List[Dict]) -> str:
        """Generate a mapping of subsystems to node IDs for the runner"""
        from collections import defaultdict
        subsystem_to_nodes = defaultdict(list)
        
        for node in nodes:
            node_id = str(node["id"])
            node_type = node.get("class_type") or node.get("type")
            
            # Skip virtual nodes
            if self._is_virtual_node(node_type):
                continue
                
            # Get subsystem for this node if available
            if node_type in self.node_registry:
                exporter_class = self.node_registry[node_type]
                try:
                    node_subsystems = exporter_class.get_subsystem()
                    # Handle both single subsystem and list of subsystems
                    if isinstance(node_subsystems, list):
                        for subsystem in node_subsystems:
                            subsystem_to_nodes[subsystem].append(node_id)
                    else:
                        subsystem_to_nodes[node_subsystems].append(node_id)
                except (AttributeError, NotImplementedError):
                    raise RuntimeError(f"Node {node_id} of type {node_type} is missing a subsystem definition.")

        # Generate the Python dict literal
        lines = []
        
        # Add special "all" subsystem that includes all nodes
        all_node_ids = set()
        for node_list in subsystem_to_nodes.values():
            all_node_ids.update(node_list)
        if all_node_ids:
            all_node_ids_str = ', '.join(f'"{nid}"' for nid in sorted(all_node_ids))
            lines.append(f'        "all": [{all_node_ids_str}],')
        
        # Add regular subsystems
        for subsystem, node_ids in sorted(subsystem_to_nodes.items()):
            node_ids_str = ', '.join(f'"{nid}"' for nid in sorted(node_ids))
            lines.append(f'        "{subsystem}": [{node_ids_str}],')
        
        # Remove trailing comma from last line
        if lines:
            lines[-1] = lines[-1].rstrip(',')
            
        return "\n".join(lines)
    
    def _generate_minimal_runner(self, output_path: Path, node_instances: List[str], 
                               connections: List[str], nodes: List[Dict], metadata: Dict):
        """Generate a minimal runner.py that imports and wires nodes"""
        
        # Read the runner template
        template_path = Path(__file__).parent / "templates" / "framework" / "runner.tpl"
        if not template_path.exists():
            raise FileNotFoundError(f"Runner template not found at {template_path}")
        
        template_content = template_path.read_text(encoding='utf-8')
        
        # Extract all node classes from node_instances
        node_imports = []
        for instance in node_instances:
            # Parse "node_1 = IsaacGymEnvNode_1("1")" to get class name
            parts = instance.split(' = ')
            if len(parts) == 2:
                class_instantiation = parts[1].split('(')[0]
                node_imports.append(class_instantiation)
        
        # Generate all the dynamic sections
        node_imports_section = self._generate_node_imports_section(node_imports)
        node_instances_section = self._generate_node_instances_section(node_instances)
        node_dictionary_section = self._generate_node_dictionary_section(node_instances)
        add_nodes_to_runner_section = self._generate_add_nodes_to_runner_section(node_instances)
        connections_section = self._generate_connections_section(connections)
        workflow_nodes_info = self._generate_workflow_nodes_info(nodes)
        subsystem_mapping = self._generate_subsystem_mapping(nodes)
        
        # Prepare template variables for substitution
        template_vars = {
            "METADATA": json.dumps(metadata, indent=2) if metadata else 'None',
            "NODE_IMPORTS_SECTION": node_imports_section,
            "NODE_INSTANCES_SECTION": node_instances_section,
            "NODE_DICTIONARY_SECTION": node_dictionary_section,
            "ADD_NODES_TO_RUNNER_SECTION": add_nodes_to_runner_section,
            "CONNECTIONS_SECTION": connections_section,
            "WORKFLOW_NODES_INFO": workflow_nodes_info,
            "SUBSYSTEM_MAPPING": subsystem_mapping
        }
        
        # Substitute placeholders in template
        runner_content = template_content.format(**template_vars)
        
        # Write the generated runner.py
        (output_path / "runner.py").write_text(runner_content, encoding='utf-8')
