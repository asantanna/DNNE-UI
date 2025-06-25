"""
Main export system that converts node graphs to Python scripts
"""

from pathlib import Path
import json
from typing import Dict, List, Any, Optional

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
    """Main export system that converts graphs to Python scripts"""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"
        self.node_registry = {}  # Maps node types to exportable classes
        
    def register_node(self, node_type: str, node_class: type):
        """Register an exportable node type"""
        self.node_registry[node_type] = node_class
        print(f"Registered node type: {node_type}")
    
    def export_workflow(self, workflow: Dict) -> str:
        """Convert workflow JSON to Python script"""
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])
        
        # Build connection map
        connections = self._build_connection_map(links, nodes)
        
        # Collect all imports
        imports = set([
            "import torch",
            "import torch.nn as nn", 
            "import torch.nn.functional as F"
        ])
        
        # Generate code for each node
        node_code_sections = []
        
        for node in nodes:
            node_id = str(node["id"])
            node_type = node["class_type"]
            
            print(f"Processing node {node_id} of type {node_type}")
            
            if node_type in self.node_registry:
                exporter_class = self.node_registry[node_type]
                
                # Add node-specific imports
                imports.update(exporter_class.get_imports())
                
                # Generate node code
                try:
                    template_vars = exporter_class.prepare_template_vars(
                        node_id, node, connections
                    )
                    code = self._render_template(
                        exporter_class.get_template_name(),
                        template_vars
                    )
                    
                    # Add section header
                    section = f"\n# Node {node_id}: {node_type}\n{code}\n"
                    node_code_sections.append(section)
                    
                except Exception as e:
                    print(f"Error processing node {node_id}: {e}")
                    node_code_sections.append(
                        f"\n# ERROR: Could not process node {node_id} ({node_type}): {e}\n"
                    )
            else:
                print(f"Warning: No exporter registered for node type: {node_type}")
                node_code_sections.append(
                    f"\n# WARNING: No exporter for node type: {node_type}\n"
                )
        
        # Load context template
        context_template = (self.templates_dir / "base" / "context.py").read_text()
        
        # Assemble final script
        script = self._assemble_script(
            list(imports),
            context_template,
            node_code_sections,
            workflow.get("metadata", {})
        )
        
        return script
    
    def _build_connection_map(self, links: List, nodes: List) -> Dict:
        """Build a map of node connections from links"""
        connections = {}
        
        # Create node ID to node mapping for easier lookup
        node_map = {str(n["id"]): n for n in nodes}
        
        for link in links:
            # Link format: [link_id, source_node_id, source_slot, target_node_id, target_slot]
            if len(link) >= 5:
                source_node_id = str(link[1])
                source_slot = link[2]
                target_node_id = str(link[3])
                target_slot = link[4]
                
                if target_node_id not in connections:
                    connections[target_node_id] = {}
                
                # Map target input name to source
                # For now, use slot indices - in real implementation, 
                # would map to actual input/output names
                connections[target_node_id][f"input_{target_slot}"] = {
                    "source_id": source_node_id,
                    "source_slot": source_slot,
                    "source_var": f"node_{source_node_id}_output"
                }
        
        return connections
    
    def _render_template(self, template_name: str, template_vars: Dict) -> str:
        """Render a template with given variables"""
        template_path = self.templates_dir / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        template_code = template_path.read_text()
        
        # Replace template_vars dictionary
        # Find the template_vars line
        lines = template_code.split('\n')
        in_template_vars = False
        new_lines = []
        brace_count = 0
        
        for line in lines:
            if line.strip().startswith('template_vars = {'):
                # Start of template_vars - replace with new dict
                new_lines.append(f'template_vars = {repr(template_vars)}')
                in_template_vars = True
                brace_count = line.count('{') - line.count('}')
            elif in_template_vars:
                # Track braces to handle multi-line dicts
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    in_template_vars = False
                # Skip original template_vars lines
            else:
                # Keep all other lines as-is
                new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    def _assemble_script(self, imports: List[str], context_code: str,
                        node_sections: List[str], metadata: Dict) -> str:
        """Assemble final Python script"""
        script_parts = [
            "#!/usr/bin/env python3",
            '"""',
            "Generated by DNNE Export System",
            f"Metadata: {json.dumps(metadata, indent=2) if metadata else 'None'}",
            '"""',
            "",
            "# Imports",
        ]
        
        # Add sorted imports
        script_parts.extend(sorted(set(imports)))
        
        script_parts.extend([
            "",
            "# Context definition",
            context_code,
            "",
            "# Initialize context",
            "context = Context()",
            "",
            "# Node implementations",
        ])
        
        # Add node sections
        script_parts.extend(node_sections)
        
        # Add basic execution  
        main_check = 'if __name__ == "__main__":'
        script_parts.extend([
            "",
            "# Execution",
            main_check,
            '    print("DNNE exported script")',
            '    print(f"Loaded {len(context.memory)} components")',
        ])
        
        return '\n'.join(script_parts)