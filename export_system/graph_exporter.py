#!/usr/bin/env python3
"""
DNNE Queue-Based Export System
Converts node graphs to reactive Python scripts using async queues
"""

from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import logging

class ExportableNode:
    """Base class for nodes that can be exported to code"""
    
    @classmethod
    def get_template_name(cls) -> str:
        """Return the template file name for this node type"""
        raise NotImplementedError
    
    @classmethod
    def prepare_template_vars(cls, node_id: str, node_data: Dict, 
                            connections: Dict, node_registry: Dict = None, 
                            all_nodes: List = None, all_links: List = None) -> Dict[str, Any]:
        """Prepare variables for template substitution"""
        raise NotImplementedError
    
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
    def get_input_name_for_slot(cls, slot: int) -> str:
        """Get input name for a specific slot number"""
        input_names = cls.get_input_names()
        if slot < len(input_names):
            return input_names[slot]
        return f"input_{slot}"
    
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
        if hasattr(cls, '_schema_cache') and cache_key in cls._schema_cache:
            return cls._schema_cache[cache_key]
        
        # Get initial schema and resolve None values
        schema = cls.get_initial_output_schema(node_data)
        schema_copy = json.loads(json.dumps(schema))  # Deep copy
        
        # Scan for None values and resolve them
        if cls._resolve_schema_nones(schema_copy, node_data, connections, 
                                     node_registry, all_nodes, all_links):
            # Cache the resolved schema
            if not hasattr(cls, '_schema_cache'):
                cls._schema_cache = {}
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
        if hasattr(cls, '_schema_cache') and cache_key in cls._schema_cache:
            return cls._schema_cache[cache_key]
        
        # Build input schema by querying each connected input
        input_schema = {}
        input_names = cls.get_input_names()
        
        for input_name in input_names:
            if "inputs" in connections and input_name in connections["inputs"]:
                input_info = connections["inputs"][input_name]
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
        if not hasattr(cls, '_schema_cache'):
            cls._schema_cache = {}
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
        self.logger.info(f"Registered node type: {node_type}")
    
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
    
    def _is_virtual_node(self, node_type: str) -> bool:
        """Check if a node type is virtual (configuration-only)"""
        if node_type not in self.node_registry:
            return False
        
        node_exporter_class = self.node_registry[node_type]
        
        # Check if the exporter has is_virtual method
        if hasattr(node_exporter_class, 'is_virtual'):
            return node_exporter_class.is_virtual()
        
        return False
    
    def export_workflow(self, workflow: Dict, output_path: Optional[Path] = None) -> str:
        """Convert workflow JSON to modular Python package"""
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])
        metadata = workflow.get("metadata", {})
        
        # WORKAROUND: Fix corrupted to_slot values by reading original JSON
        # ComfyUI pipeline corrupts all to_slot values to 0, so we restore them
        links = self._fix_corrupted_slots(links, metadata)
        
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
        
        # Create package structure
        framework_dir, nodes_dir = self._create_package_structure(output_path)
        
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
                self.logger.info(f"Skipping virtual node {node_id} ({node_type}) - configuration only")
                continue
            
            if node_type in self.node_registry:
                node_class = self.node_registry[node_type]
                
                # Get template and prepare variables
                template_name = node_class.get_template_name()
                template_vars = node_class.prepare_template_vars(
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
                
                # Handle node dependencies (e.g., rlgames_ppo_components.py)
                if hasattr(node_class, 'get_dependencies'):
                    dependencies = node_class.get_dependencies()
                    for dep_file in dependencies:
                        self._copy_dependency(nodes_dir, dep_file)
                
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
        
        # Generate nodes/__init__.py
        self._generate_node_init(nodes_dir, node_classes)
        
        # Generate connections
        connections = self._generate_connections(links, nodes)
        
        # Generate minimal runner.py
        self._generate_minimal_runner(output_path, node_instances, connections, metadata)
        
        self.logger.info(f"Exported modular package to: {output_path}")
        
        # Return the path to the runner for backward compatibility
        return str(output_path / "runner.py")
    
    def _fix_corrupted_slots(self, links: List, workflow_metadata: Dict = None) -> List:
        """WORKAROUND: Fix to_slot values corrupted by ComfyUI pipeline"""
        try:
            # Try to find the correct workflow JSON file
            from pathlib import Path
            import json
            
            # Try to get workflow name from metadata
            workflow_name = None
            if workflow_metadata:
                workflow_name = workflow_metadata.get("workflow_name")
            
            # Require workflow name to be specified
            if not workflow_name:
                # Check if this is a test workflow
                if workflow_metadata and workflow_metadata.get("dnne-test"):
                    return links
                raise ValueError("Workflow name is required for slot correction")
            
            # Only try the specific workflow file - no fallbacks
            workflow_dir = Path("user/default/workflows")
            workflow_path = workflow_dir / f"{workflow_name}.json"
            
            if not workflow_path.exists():
                raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
            
            self.logger.info(f"Reading original workflow from: {workflow_path}")
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
                            self.logger.info(f"Fixed connection {from_node}.{from_slot}→{to_node}: to_slot {corrupted_to_slot} → {correct_to_slot}")
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
            
            self.logger.info(f"Applied {fixes_applied} slot corrections from {workflow_path}")
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
    
    def _copy_dependency(self, target_dir: Path, dep_filename: str):
        """Copy a dependency file from templates to the target export directory"""
        import shutil
        
        # Handle paths that may include subdirectories
        if '/' in dep_filename:
            # Split into directory and filename
            dep_parts = dep_filename.split('/')
            dep_subdir = '/'.join(dep_parts[:-1])
            dep_file = dep_parts[-1]
            
            # Source path in templates/nodes
            source_path = self.templates_dir / "nodes" / dep_subdir / dep_file
            
            # Target path maintains the same structure in nodes directory
            target_subdir = target_dir / dep_subdir
            target_subdir.mkdir(parents=True, exist_ok=True)
            target_path = target_subdir / dep_file
        else:
            # Single file without subdirectory - in nodes/
            source_path = self.templates_dir / "nodes" / dep_filename
            target_path = target_dir / dep_filename
        
        if source_path.exists():
            # Copy the dependency file
            shutil.copy2(source_path, target_path)
            self.logger.info(f"Copied dependency: {dep_filename} -> {target_path}")
        else:
            raise FileNotFoundError(f"Required dependency file not found: {source_path}. "
                                    f"The node requires '{dep_filename}' but it does not exist in templates/")
    
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
        
        for link in links:
            # Link format: [link_id, from_node, from_slot, to_node, to_slot]
            if len(link) >= 5:
                from_node = str(link[1])
                to_node = str(link[3])
                
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
                    
                    connections["inputs"][input_name] = {
                        "from_node": from_node,
                        "from_slot": link[2]
                    }
                elif from_node == node_id:
                    # Outgoing connection
                    if link[2] not in connections["outputs"]:
                        connections["outputs"][link[2]] = []
                    connections["outputs"][link[2]].append({
                        "to_node": to_node,
                        "to_slot": link[4]
                    })
        
        return connections
    
    def _generate_connections(self, links: List, nodes: List) -> List[str]:
        """Generate connection tuples for wire_nodes"""
        connections = []
        
        # First, identify which nodes are being skipped (consumed by networks)
        network_consumed_nodes = set()
        for node in nodes:
            if (node.get("class_type") or node.get("type")) == "Network":
                network_id = str(node["id"])
                network_class = self.node_registry.get("Network")
                if network_class:
                    consumed_layers = network_class._detect_network_layers(network_id, nodes, links)
                    for layer_info in consumed_layers:
                        network_consumed_nodes.add(layer_info["node_id"])
        
        # Also identify virtual nodes that will be skipped
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
                
                # Skip connections to/from consumed nodes
                if from_node in network_consumed_nodes or to_node in network_consumed_nodes:
                    self.logger.info(f"Skipping connection from {from_node} to {to_node} - involves consumed node")
                    continue
                
                # Skip connections to/from virtual nodes
                if from_node in virtual_nodes or to_node in virtual_nodes:
                    self.logger.info(f"Skipping connection from {from_node} to {to_node} - involves virtual node")
                    continue
                
                # Get actual output name
                if from_node not in node_info:
                    self.logger.warning(f"From node {from_node} not in node_info, skipping connection")
                    continue
                    
                outputs = node_info[from_node]["outputs"]
                output_name = outputs[from_slot] if from_slot < len(outputs) else f"output_{from_slot}"
                
                # Get input name
                if to_node not in node_info:
                    self.logger.warning(f"To node {to_node} not in node_info, skipping connection")
                    continue
                    
                to_node_class = node_info[to_node]["class"]
                if to_node_class and hasattr(to_node_class, 'get_input_names'):
                    input_names = to_node_class.get_input_names()
                    if to_slot < len(input_names):
                        input_name = input_names[to_slot]
                    else:
                        raise ValueError(f"Input slot {to_slot} out of range for node {to_node} of type {node_info[to_node]['type']}. Available inputs: {input_names}")
                else:
                    raise ValueError(f"Cannot determine input name for slot {to_slot} on node {to_node} of type {node_info[to_node]['type']}")
                
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
        self.logger.info(f"Placeholder compute for {node_type}")
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
                self.logger.info(f"Cleaning existing export directory: {output_path}")
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
        
        # Export exceptions.py
        exceptions_content = self._load_template("framework/exceptions.py")
        (framework_dir / "exceptions.py").write_text(exceptions_content, encoding='utf-8')
        
        # Export base_nodes.py
        base_nodes_content = self._load_template("framework/base_nodes.py")
        (framework_dir / "base_nodes.py").write_text(base_nodes_content, encoding='utf-8')
        
        # Export graph_runner.py
        graph_runner_content = self._load_template("framework/graph_runner.py")
        (framework_dir / "graph_runner.py").write_text(graph_runner_content, encoding='utf-8')
        
        # Export checkpoint.py (formerly run_utils.py)
        checkpoint_content = self._load_template("framework/checkpoint.py")
        (framework_dir / "checkpoint.py").write_text(checkpoint_content, encoding='utf-8')
        
        # Export globals.py
        globals_content = self._load_template("framework/globals.py")
        (framework_dir / "globals.py").write_text(globals_content, encoding='utf-8')
        
        # Export globals_threadsafe.py if it exists
        try:
            globals_threadsafe_content = self._load_template("framework/globals_threadsafe.py")
            (framework_dir / "globals_threadsafe.py").write_text(globals_threadsafe_content, encoding='utf-8')
            self.logger.info("Exported globals_threadsafe.py for thread-safe yielding support")
        except FileNotFoundError:
            self.logger.error("globals_threadsafe.py not found in templates")
            raise FileNotFoundError("globals_threadsafe.py not found in templates")
        
        # Export dnne_exceptions.py
        dnne_exceptions_content = self._load_template("framework/dnne_exceptions.py")
        (framework_dir / "dnne_exceptions.py").write_text(dnne_exceptions_content, encoding='utf-8')
        
        # Export metrics_logger.py (required by BalancingNode)
        metrics_logger_content = self._load_template("framework/metrics_logger.py")
        (framework_dir / "metrics_logger.py").write_text(metrics_logger_content, encoding='utf-8')
        self.logger.info("Exported metrics_logger.py for balancing node support")
    
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
    
    def _generate_minimal_runner(self, output_path: Path, node_instances: List[str], 
                               connections: List[str], metadata: Dict):
        """Generate a minimal runner.py that imports and wires nodes"""
        
        # Read the runner template
        template_path = Path(__file__).parent / "templates" / "framework" / "runner.py"
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
        
        # Prepare template variables for substitution
        template_vars = {
            "METADATA": json.dumps(metadata, indent=2) if metadata else 'None',
            "NODE_IMPORTS_SECTION": node_imports_section,
            "NODE_INSTANCES_SECTION": node_instances_section,
            "NODE_DICTIONARY_SECTION": node_dictionary_section,
            "ADD_NODES_TO_RUNNER_SECTION": add_nodes_to_runner_section,
            "CONNECTIONS_SECTION": connections_section
        }
        
        # Substitute placeholders in template
        runner_content = template_content.format(**template_vars)
        
        # Write the generated runner.py
        (output_path / "runner.py").write_text(runner_content, encoding='utf-8')
