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
                            connections: Dict) -> Dict[str, Any]:
        """Prepare variables for template substitution"""
        raise NotImplementedError
    
    @classmethod
    def get_imports(cls) -> List[str]:
        """Return list of import statements needed by this node"""
        return []


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
                    node_id, node, self._get_node_connections(node_id, links, nodes)
                )
                
                # Load and process template
                template_content = self._load_template(template_name)
                node_code = self._process_template(template_content, template_vars)
                node_implementations.append(node_code)
                
                # Create instance with valid Python variable name
                class_name = template_vars.get("CLASS_NAME", node_type + "Node")
                instance_name = f"node_{node_id}"
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
        
        for link in links:
            # Link format: [link_id, from_node, from_slot, to_node, to_slot]
            if len(link) >= 5:
                from_node = str(link[1])
                to_node = str(link[3])
                
                if to_node == node_id:
                    # Incoming connection
                    connections["inputs"][link[4]] = {
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
        
        # Map node types to their output names
        node_outputs = {}
        for node in nodes:
            node_id = str(node["id"])
            node_type = node["class_type"]
            
            # Default output names based on node type
            if node_type == "MNISTDataset":
                node_outputs[node_id] = ["batch_data", "batch_labels"]
            elif node_type == "LinearLayer":
                node_outputs[node_id] = ["output_tensor"]
            elif node_type == "CameraSensor":
                node_outputs[node_id] = ["image", "timestamp"]
            elif node_type == "AudioSensor":
                node_outputs[node_id] = ["audio_data", "timestamp"]
            elif node_type == "IMUSensor":
                node_outputs[node_id] = ["acceleration", "angular_velocity", "orientation"]
            elif node_type == "VisionNetwork":
                node_outputs[node_id] = ["vision_features"]
            elif node_type == "SoundNetwork":
                node_outputs[node_id] = ["sound_features"]
            elif node_type == "DecisionNetwork":
                node_outputs[node_id] = ["action", "confidence"]
            elif node_type == "Loss":
                node_outputs[node_id] = ["loss", "accuracy"]
            elif node_type == "Display":
                node_outputs[node_id] = []  # No outputs
            elif node_type == "RobotController":
                node_outputs[node_id] = ["joint_commands", "status"]
            elif node_type == "Optimizer":
                node_outputs[node_id] = ["step_complete"]
            # Add more as needed
            else:
                node_outputs[node_id] = [f"output_{i}" for i in range(3)]
        
        for link in links:
            if len(link) >= 5:
                from_node = str(link[1])
                from_slot = link[2]
                to_node = str(link[3])
                to_slot = link[4]
                
                # Get actual output name
                outputs = node_outputs.get(from_node, [])
                output_name = outputs[from_slot] if from_slot < len(outputs) else f"output_{from_slot}"
                
                # Input names are more standardized
                input_name = self._get_input_name_for_slot(nodes, to_node, to_slot)
                
                connections.append(
                    f'("{from_node}", "{output_name}", "{to_node}", "{input_name}")'
                )
        
        return connections
    
    def _get_input_name_for_slot(self, nodes: List, node_id: str, slot: int) -> str:
        """Get input name for a given slot"""
        # Find node type
        for node in nodes:
            if str(node["id"]) == node_id:
                node_type = node["class_type"]
                
                # Map based on node type
                if node_type == "LinearLayer":
                    return "input_tensor"
                elif node_type == "VisionNetwork":
                    return "camera_data"
                elif node_type == "SoundNetwork":
                    return "audio_data"
                elif node_type == "DecisionNetwork":
                    return ["vision_features", "sound_features"][slot] if slot < 2 else f"input_{slot}"
                elif node_type == "Loss":
                    return ["predictions", "labels"][slot] if slot < 2 else f"input_{slot}"
                elif node_type == "Display":
                    return "input_0"
                elif node_type == "Optimizer":
                    return "loss"
                elif node_type == "RobotController":
                    return "action"
                # Add more mappings as needed
                
        return f"input_{slot}"
    
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