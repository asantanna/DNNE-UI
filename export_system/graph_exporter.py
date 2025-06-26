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
        return ["output"]
    
    @classmethod
    def get_input_names(cls) -> List[str]:
        """Return list of input names for this node type"""
        return ["input"]
    
    @classmethod
    def get_input_name_for_slot(cls, slot: int) -> str:
        """Get input name for a specific slot number"""
        input_names = cls.get_input_names()
        if slot < len(input_names):
            return input_names[slot]
        return f"input_{slot}"
    
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
                source_node_type = node["class_type"]
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
            
            if "inputs" not in source_connections or "input" not in source_connections["inputs"]:
                raise ValueError(f"Pass-through node {source_node_id} has no 'input' connection to query")
            
            # Recursively query the node connected to this pass-through node's input
            input_info = source_connections["inputs"]["input"]
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
            
            upstream_node_type = upstream_node_data["class_type"]
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
        
    def register_node(self, node_type: str, node_class: type):
        """Register an exportable node type"""
        self.node_registry[node_type] = node_class
        self.logger.info(f"Registered node type: {node_type}")
    
    def export_workflow(self, workflow: Dict, output_path: Optional[Path] = None) -> str:
        """Convert workflow JSON to queue-based Python script"""
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])
        metadata = workflow.get("metadata", {})
        
        # Collect all imports
        imports = {
            "import asyncio",
            "import time",
            "import logging",
            "from typing import Dict, Any, List, Optional",
            "from abc import ABC, abstractmethod",
            "from asyncio import Queue",
            "",
            "# Configure logging",
            "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')",
        }
        
        # Generate node implementations
        node_implementations = []
        node_instances = []
        
        for node in nodes:
            node_id = str(node["id"])
            node_type = node["class_type"]
            
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
                node_implementations.append(node_code)
                
                # Create instance with valid Python variable name
                class_name = template_vars.get("CLASS_NAME", node_type + "Node")
                instance_name = f"node_{node_id}"
                
                # Check if node has custom instance code
                if hasattr(node_class, 'get_instance_code'):
                    node_connections = self._get_node_connections(node_id, links, nodes)
                    instance_code = node_class.get_instance_code(node_id, node, node_connections)
                    node_instances.append(instance_code)
                else:
                    node_instances.append(f'{instance_name} = {class_name}_{node_id}("{node_id}")')
                
                # Add imports
                imports.update(node_class.get_imports())
            else:
                self.logger.warning(f"Unknown node type: {node_type}")
                # Generate placeholder
                placeholder_code = self._generate_placeholder_node(node_id, node_type)
                node_implementations.append(placeholder_code)
                node_instances.append(f'node_{node_id} = PlaceholderNode_{node_id}("{node_id}")')
        
        # Load base framework template
        base_framework = self._load_template("base/queue_framework.py")
        
        # Generate connections
        connections = self._generate_connections(links, nodes)
        
        # Assemble final script
        script = self._assemble_script(
            sorted(list(imports)),
            base_framework,
            node_implementations,
            node_instances,
            connections,
            metadata
        )
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(script, encoding='utf-8')
            self.logger.info(f"Exported script to: {output_path}")
        
        return script
    
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
                node_type = node["class_type"]
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
        
        # Build a map of node_id to node_type and exporter class
        node_info = {}
        for node in nodes:
            node_id = str(node["id"])
            node_type = node["class_type"]
            node_class = self.node_registry.get(node_type)
            node_info[node_id] = {
                "type": node_type,
                "class": node_class,
                "outputs": node_class.get_output_names() if node_class else [f"output_{i}" for i in range(3)]
            }
        
        for link in links:
            if len(link) >= 5:
                from_node = str(link[1])
                from_slot = link[2]
                to_node = str(link[3])
                to_slot = link[4]
                
                # Get actual output name
                outputs = node_info[from_node]["outputs"]
                output_name = outputs[from_slot] if from_slot < len(outputs) else f"output_{from_slot}"
                
                # Get input name
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