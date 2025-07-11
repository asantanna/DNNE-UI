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
    def get_output_schema(cls, node_data: Dict) -> Dict[str, Any]:
        """Return schema describing the outputs of this node type"""
        return {
            "outputs": {
                "output": {
                    "type": "unknown",
                    "shape": None
                }
            },
            "num_samples": 1
        }
    
    @classmethod
    def get_output_tensor_size(cls, node_data: Dict, output_name: str, connections: Dict) -> int:
        """Get the tensor size for a specific output, potentially querying connected nodes"""
        schema = cls.get_output_schema(node_data)
        if output_name in schema.get("outputs", {}):
            output_info = schema["outputs"][output_name]
            if "flattened_size" in output_info:
                return output_info["flattened_size"]
            elif "shape" in output_info and output_info["shape"]:
                # Calculate flattened size from shape
                shape = output_info["shape"]
                if isinstance(shape, (list, tuple)) and len(shape) > 0:
                    size = 1
                    for dim in shape:
                        size *= dim
                    return size
        
        raise ValueError(f"Cannot determine tensor size for output '{output_name}' of node type {cls.__name__}")
    
    @classmethod
    def query_input_tensor_size(cls, input_name: str, connections: Dict, node_registry: Dict, all_nodes: List, all_links: List = None) -> int:
        """Query the tensor size from a connected input source"""
        if not connections or "inputs" not in connections or input_name not in connections["inputs"]:
            raise ValueError(f"No input connection found for '{input_name}' in {cls.__name__}")
            
        input_info = connections["inputs"][input_name]
        source_node_id = input_info["from_node"]
        source_output_slot = input_info["from_slot"]
        
        # Find the source node data
        source_node_data = None
        source_node_type = None
        for node in all_nodes:
            if str(node["id"]) == source_node_id:
                source_node_data = node
                source_node_type = node.get("class_type") or node.get("type")
                break
                
        if not source_node_data:
            raise ValueError(f"Source node {source_node_id} not found for input '{input_name}'")
            
        if source_node_type not in node_registry:
            raise ValueError(f"Unknown source node type '{source_node_type}' for input '{input_name}'")
            
        source_node_class = node_registry[source_node_type]
        
        # Get the source node's output names
        source_outputs = source_node_class.get_output_names()
        if source_output_slot >= len(source_outputs):
            raise ValueError(f"Invalid output slot {source_output_slot} for node {source_node_id} (has {len(source_outputs)} outputs)")
            
        source_output_name = source_outputs[source_output_slot]
        
        # Query the source node's schema for this output
        schema = source_node_class.get_output_schema(source_node_data)
        
        # Handle pass-through nodes (like Network)
        if schema == "pass_through":
            # For pass-through nodes, query their input connection instead
            if not all_links:
                raise ValueError(f"Cannot query pass-through node {source_node_id} without links data")
                
            # Create a temporary exporter to use _get_node_connections
            temp_exporter = GraphExporter()
            temp_exporter.node_registry = node_registry
            source_connections = temp_exporter._get_node_connections(source_node_id, all_links, all_nodes)
            
            # Try to find a connected input - try common input names
            possible_input_names = ["input", "input_a", "input_b", "input_c"]
            input_info = None
            
            for input_name in possible_input_names:
                if "inputs" in source_connections and input_name in source_connections["inputs"]:
                    input_info = source_connections["inputs"][input_name]
                    break
            
            if input_info is None:
                raise ValueError(f"Pass-through node {source_node_id} has no connected inputs to query (tried: {possible_input_names})")
            
            # Recursively query the node connected to this pass-through node's input
            upstream_node_id = input_info["from_node"]
            upstream_output_slot = input_info["from_slot"]
            
            # Find the upstream node data
            upstream_node_data = None
            for node in all_nodes:
                if str(node["id"]) == upstream_node_id:
                    upstream_node_data = node
                    break
            
            if not upstream_node_data:
                raise ValueError(f"Cannot find upstream node {upstream_node_id} for pass-through query")
            
            upstream_node_type = upstream_node_data.get("class_type") or upstream_node_data.get("type")
            upstream_node_class = node_registry.get(upstream_node_type)
            
            if not upstream_node_class:
                raise ValueError(f"No exporter found for upstream node type {upstream_node_type}")
            
            # Query the upstream node's schema
            upstream_schema = upstream_node_class.get_output_schema(upstream_node_data)
            upstream_outputs = upstream_node_class.get_output_names()
            
            if upstream_output_slot >= len(upstream_outputs):
                raise ValueError(f"Invalid upstream output slot {upstream_output_slot}")
            
            upstream_output_name = upstream_outputs[upstream_output_slot]
            
            if "outputs" not in upstream_schema or upstream_output_name not in upstream_schema["outputs"]:
                raise ValueError(f"No schema found for upstream output '{upstream_output_name}' of node {upstream_node_id}")
            
            upstream_output_info = upstream_schema["outputs"][upstream_output_name]
            
            if "flattened_size" in upstream_output_info:
                return upstream_output_info["flattened_size"]
            elif "contains" in upstream_output_info and "images" in upstream_output_info["contains"]:
                images_info = upstream_output_info["contains"]["images"]
                if "flattened_size" in images_info:
                    return images_info["flattened_size"]
            
            raise ValueError(f"Cannot determine tensor size from upstream node {upstream_node_id} output '{upstream_output_name}'")
        
        
        if "outputs" not in schema or source_output_name not in schema["outputs"]:
            raise ValueError(f"No schema found for output '{source_output_name}' of node {source_node_id}")
            
        output_info = schema["outputs"][source_output_name]
        
        if "flattened_size" in output_info:
            return output_info["flattened_size"]
        elif "contains" in output_info and "images" in output_info["contains"]:
            # For datasets, look inside the contained data
            images_info = output_info["contains"]["images"]
            if "flattened_size" in images_info:
                return images_info["flattened_size"]
                
        raise ValueError(f"Cannot determine tensor size for output '{source_output_name}' of node {source_node_id}")


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
        
        # First pass: identify nodes that are part of networks to skip individual layer processing
        network_consumed_nodes = set()
        for node in nodes:
            # Handle both ComfyUI formats: "type" and "class_type"
            node_type = node.get("class_type") or node.get("type")
            if node_type == "Network":
                network_id = str(node["id"])
                # Find which layer nodes this network consumes
                network_class = self.node_registry.get("Network")
                if network_class:
                    consumed_layers = network_class._detect_network_layers(network_id, nodes, links)
                    for layer_info in consumed_layers:
                        network_consumed_nodes.add(layer_info["node_id"])
        
        # Track node information for __init__.py generation
        node_classes = []
        node_instances = []
        
        for node in nodes:
            node_id = str(node["id"])
            node_type = node.get("class_type") or node.get("type")
            
            # Skip individual layer nodes that are consumed by Network nodes
            if node_id in network_consumed_nodes and node_type == "LinearLayer":
                self.logger.info(f"Skipping LinearLayer node {node_id} - consumed by Network node")
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
            
            # List of possible workflow files to try
            workflow_dir = Path("user/default/workflows")
            possible_files = []
            
            if workflow_name:
                possible_files.append(workflow_dir / f"{workflow_name}.json")
            
            # Also try common files
            possible_files.extend([
                workflow_dir / "Minimal.json",
                workflow_dir / "MNIST Test.json"
            ])
            
            # Try each possible file
            for workflow_path in possible_files:
                if workflow_path.exists():
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
                                # Connection not found in original, keep as-is
                                fixed_links.append(link)
                        else:
                            fixed_links.append(link)
                    
                    self.logger.info(f"Applied {fixes_applied} slot corrections from {workflow_path}")
                    return fixed_links
            
            self.logger.warning("Could not find any workflow JSON files for slot correction")
            return links
                
        except Exception as e:
            self.logger.warning(f"Failed to fix corrupted slots: {e}")
            return links
    
    def _load_template(self, template_name: str) -> str:
        """Load template file content"""
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return template_path.read_text(encoding='utf-8')
    
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
            "",
            "if __name__ == '__main__':",
            "    asyncio.run(main())",
        ])
        
        return "\n".join(script_parts)
    
    def _create_package_structure(self, output_path: Path):
        """Create the package directory structure"""
        # Create main directories
        output_path.mkdir(parents=True, exist_ok=True)
        framework_dir = output_path / "framework"
        framework_dir.mkdir(exist_ok=True)
        nodes_dir = output_path / "nodes"
        nodes_dir.mkdir(exist_ok=True)
        
        # Create __init__.py files
        (output_path / "__init__.py").write_text("# DNNE Generated Package\n", encoding='utf-8')
        (framework_dir / "__init__.py").write_text("from .base import QueueNode, SensorNode, GraphRunner\n", encoding='utf-8')
        
        return framework_dir, nodes_dir
    
    def _export_framework(self, framework_dir: Path):
        """Export the queue framework to framework/base.py"""
        base_framework = self._load_template("base/queue_framework.py")
        
        # Add proper module header
        framework_content = [
            '"""Queue-Based Node Framework"""',
            "import asyncio",
            "import time",
            "import logging",
            "from typing import Dict, Any, List, Optional",
            "from abc import ABC, abstractmethod",
            "from asyncio import Queue",
            "",
            base_framework
        ]
        
        (framework_dir / "base.py").write_text("\n".join(framework_content), encoding='utf-8')
        
        # Also export run_utils.py to the root directory (not framework dir)
        # This matches the import path: from run_utils import ...
        try:
            run_utils_content = self._load_template("base/run_utils.py")
            output_root = framework_dir.parent  # Go up one level to the export root
            (output_root / "run_utils.py").write_text(run_utils_content, encoding='utf-8')
        except Exception as e:
            self.logger.warning(f"Could not export run_utils.py: {e}")
    
    def _export_node_to_file(self, nodes_dir: Path, node_id: str, node_type: str, 
                            node_code: str, node_imports: List[str]) -> str:
        """Export a single node to its own file and return the class name"""
        # Generate filename based on node type and ID
        node_type_snake = node_type.lower().replace(" ", "_")
        filename = f"{node_type_snake}_{node_id}.py"
        
        # Extract class name from the node code
        import re
        class_match = re.search(r'class ([a-zA-Z_][a-zA-Z0-9_-]*)\(', node_code)
        if not class_match:
            raise ValueError(f"Could not extract class name from node {node_id}")
        class_name = class_match.group(1)
        
        # Prepare the file content
        file_content = [
            f'"""Node implementation for {node_type} (ID: {node_id})"""'
        ]
        
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
        file_content.append("from framework.base import QueueNode, SensorNode")
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
        
        # CRITICAL: Sort nodes to ensure Isaac Gym nodes are imported first
        # This prevents "PyTorch was imported before isaacgym" error
        isaac_gym_nodes = []
        other_nodes = []
        
        for node_id, node_type, class_name in node_classes:
            if "IsaacGym" in node_type or "isaac" in node_type.lower():
                isaac_gym_nodes.append((node_id, node_type, class_name))
            else:
                other_nodes.append((node_id, node_type, class_name))
        
        # Import Isaac Gym nodes first, then others
        sorted_nodes = isaac_gym_nodes + other_nodes
        
        for node_id, node_type, class_name in sorted_nodes:
            node_type_snake = node_type.lower().replace(" ", "_")
            filename = f"{node_type_snake}_{node_id}"
            init_content.append(f"from .{filename} import {class_name}")
        
        init_content.extend([
            "",
            "__all__ = [",
        ])
        
        for _, _, class_name in sorted_nodes:
            init_content.append(f'    "{class_name}",')
        
        init_content.append("]")
        
        (nodes_dir / "__init__.py").write_text("\n".join(init_content) + "\n", encoding='utf-8')
    
    def _generate_minimal_runner(self, output_path: Path, node_instances: List[str], 
                               connections: List[str], metadata: Dict):
        """Generate a minimal runner.py that imports and wires nodes"""
        
        # Extract Isaac Gym node classes from node_instances
        isaac_gym_imports = []
        for instance in node_instances:
            # Parse "node_1 = IsaacGymEnvNode_1("1")" to get class name
            parts = instance.split(' = ')
            if len(parts) == 2:
                class_instantiation = parts[1].split('(')[0]
                if 'IsaacGym' in class_instantiation:
                    isaac_gym_imports.append(class_instantiation)
        
        runner_content = [
            "#!/usr/bin/env python3",
            '"""',
            "Generated by DNNE - Main Entry Point",
            f"Metadata: {json.dumps(metadata, indent=2) if metadata else 'None'}",
            '"""',
            "",
        ]
        
        # CRITICAL: Import isaacgym FIRST if any Isaac Gym nodes exist
        if isaac_gym_imports:
            runner_content.extend([
                "# CRITICAL: Import isaacgym before ANY other imports to prevent torch import order issues",
                "# This MUST be the first import after the header comments",
                "import isaacgym",
                ""
            ])
        
        runner_content.extend([
            "import sys",
            "import argparse",
            "from pathlib import Path",
            "",
            "# Add current directory to Python path for imports",
            "sys.path.insert(0, str(Path(__file__).parent))",
            "",
            "import asyncio",
            "import logging",
            "",
        ])
        
        # Add Isaac Gym node imports if any exist
        if isaac_gym_imports:
            runner_content.extend([
                "# CRITICAL: Import Isaac Gym nodes directly (not via __init__.py) to prevent torch import order issues",
                "# This prevents 'PyTorch was imported before isaacgym' error"
            ])
            for isaac_class in isaac_gym_imports:
                # Convert class name to module name
                # IsaacGymEnvNode_1 -> isaacgymenvnode_1
                module_name = isaac_class.replace('Node_', 'node_').lower()
                runner_content.append(f"from nodes.{module_name} import {isaac_class}")
            runner_content.append("")
        
        runner_content.extend([
            "from framework.base import GraphRunner",
            "from nodes import *",
            "",
            "def configure_logging(verbose=False):",
            '    """Configure logging based on verbose flag"""',
            "    if verbose:",
            "        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')",
            "    else:",
            "        # Only show WARNING and above for quiet mode",
            "        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(message)s')",
            "",
            "async def main():",
            '    """Main execution function"""',
            "    # Parse command line arguments",
            "    parser = argparse.ArgumentParser(description='DNNE Generated Training')",
            "    parser.add_argument('--verbose', '-v', action='store_true',",
            "                       help='Enable verbose batch-level logging')",
            "    parser.add_argument('--test-mode', action='store_true',",
            "                       help='Run in test mode with limited duration and performance tracking')",
            "    parser.add_argument('--save-checkpoint-dir', type=str,",
            "                       help='Directory to save checkpoints (will create node_<id> subdirectories)')",
            "    parser.add_argument('--load-checkpoint-dir', type=str,",
            "                       help='Directory to load checkpoints from (expects node_<id> subdirectories)')",
            "    parser.add_argument('--timeout', type=str,",
            "                       help='Training duration (e.g., \"5m\", \"30s\", \"1h30m\")')",
            "    parser.add_argument('--visual', action='store_true',",
            "                       help='Enable visual mode (overrides headless setting)')",
            "    parser.add_argument('--headless', action='store_true',",
            "                       help='Force headless mode (default)')",
            "    parser.add_argument('--inference', action='store_true',",
            "                       help='Run in inference mode (no training, no gradients)')",
            "    args = parser.parse_args()",
            "",
            "    # Parse timeout if provided",
            "    duration_seconds = None",
            "    if args.timeout:",
            "        try:",
            "            from run_utils import CheckpointManager",
            "            manager = CheckpointManager(\"temp\")",
            "            duration_seconds = manager.parse_time_format(args.timeout)",
            "            print(f\"⏱️  Running for {args.timeout} ({duration_seconds:.0f} seconds)\")",
            "        except Exception as e:",
            "            print(f\"⚠️  Invalid timeout format: {e}\")",
            "            print(\"   Use formats like: 5m, 30s, 1h30m\")",
            "            import sys",
            "            sys.exit(1)",
            "",
            "    # Set global flags for nodes to access",
            "    import builtins",
            "    builtins.VERBOSE = args.verbose",
            "    builtins.SAVE_CHECKPOINT_DIR = args.save_checkpoint_dir",
            "    builtins.LOAD_CHECKPOINT_DIR = args.load_checkpoint_dir",
            "    builtins.VISUAL_MODE = args.visual",
            "    builtins.HEADLESS_MODE = args.headless",
            "    builtins.INFERENCE_MODE = args.inference",
            "    configure_logging(args.verbose)",
            "",
            '    print("🚀 Starting DNNE Queue-Based Execution")',
            "    if args.inference:",
            '        print("🔍 Inference mode enabled - no training or gradients")',
            "    if args.verbose:",
            '        print("📝 Verbose mode enabled - showing all batch details")',
            "    else:",
            '        print("📊 Quiet mode - showing epoch summaries only")',
            "    if args.save_checkpoint_dir:",
            '        print(f"💾 Checkpoint saving to: {args.save_checkpoint_dir}")',
            "    if args.load_checkpoint_dir:",
            '        print(f"📁 Loading checkpoints from: {args.load_checkpoint_dir}")',
            '    print("=" * 60)',
            "",
            "    # Create nodes",
        ])
        
        # Add node instances
        for instance in node_instances:
            runner_content.append(f"    {instance}")
        
        runner_content.extend([
            "",
            "    # Create runner",
            "    runner = GraphRunner()",
            "",
            "    # Add nodes to runner",
        ])
        
        # Add nodes to runner
        for instance in node_instances:
            node_var = instance.split(" = ")[0].strip()
            runner_content.append(f"    runner.add_node({node_var})")
        
        runner_content.extend([
            "",
            "    # Wire connections",
            "    connections = [",
        ])
        
        # Add connections
        for conn in connections:
            runner_content.append(f"        {conn},")
        
        runner_content.extend([
            "    ]",
            "    runner.wire_nodes(connections)",
            "",
            "    # Run the graph",
            "    try:",
            "        if args.test_mode:",
            "            print('🧪 Test mode: Running for 30 seconds with performance tracking')",
            "            import time",
            "            start_time = time.time()",
            "            await runner.run(duration=30.0)  # Run for 30 seconds in test mode",
            "            end_time = time.time()",
            "            print(f'✅ Test mode completed in {end_time - start_time:.1f} seconds')",
            "        elif duration_seconds is not None:",
            "            # Run for specified timeout duration",
            "            import time",
            "            start_time = time.time()",
            "            await runner.run(duration=duration_seconds)",
            "            end_time = time.time()",
            "            print(f'✅ Completed {args.timeout} run in {end_time - start_time:.1f} seconds')",
            "        else:",
            "            # Run indefinitely (Ctrl+C to stop)",
            "            await runner.run()",
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
            "",
            "if __name__ == '__main__':",
            "    asyncio.run(main())",
        ])
        
        (output_path / "runner.py").write_text("\n".join(runner_content), encoding='utf-8')