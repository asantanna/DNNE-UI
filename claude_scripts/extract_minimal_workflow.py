#!/usr/bin/env python3
"""
Extract minimal workflow structure from DNNE workflow JSON files.

This script loads a workflow and outputs only the keys that graph_exporter.py
actually uses, making it suitable for test fixtures or debugging.

Usage:
    python extract_minimal_workflow.py WORKFLOW_NAME
    
Example:
    python extract_minimal_workflow.py MNIST_Test
"""

import sys
import json
from pathlib import Path


def extract_minimal_workflow(workflow_data, add_test_metadata=False):
    """Extract only the essential keys that graph_exporter uses."""
    minimal = {}
    
    # Extract top-level keys
    if "metadata" in workflow_data:
        minimal["metadata"] = workflow_data["metadata"]
    elif add_test_metadata:
        # Add test metadata if requested
        minimal["metadata"] = {"skip-slot-correction": True}
    
    # Extract nodes with only essential fields
    if "nodes" in workflow_data:
        minimal["nodes"] = []
        for node in workflow_data["nodes"]:
            minimal_node = {}
            
            # Required fields
            if "id" in node:
                minimal_node["id"] = str(node["id"])  # Ensure string for consistency
            
            # Node type (check both possible keys)
            if "type" in node:
                minimal_node["type"] = node["type"]
            elif "class_type" in node:
                minimal_node["type"] = node["class_type"]
            
            # Always use empty inputs dict for test fixtures
            # (The actual input connections come from the links)
            minimal_node["inputs"] = {}
                
            if "widgets_values" in node:
                minimal_node["widgets_values"] = node["widgets_values"]
            
            minimal["nodes"].append(minimal_node)
    
    # Links are already minimal, just copy them
    if "links" in workflow_data:
        minimal["links"] = workflow_data["links"]
    
    return minimal


def json_to_python_literal(obj):
    """Convert JSON object to Python literal string with compact formatting."""
    import re
    
    # Custom JSON encoder for compact arrays
    def compact_json_dumps(obj, indent=0):
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            lines = ["{"]
            items = list(obj.items())
            for i, (key, value) in enumerate(items):
                # Format the value
                if isinstance(value, list) and len(value) <= 8 and all(not isinstance(item, (dict, list)) for item in value):
                    # Small simple lists on one line
                    value_str = json.dumps(value)
                else:
                    value_str = compact_json_dumps(value, indent + 4)
                
                # Add the key-value pair
                comma = "," if i < len(items) - 1 else ""
                lines.append(f'{" " * (indent + 4)}"{key}": {value_str}{comma}')
            lines.append(" " * indent + "}")
            return "\n".join(lines)
        
        elif isinstance(obj, list):
            if not obj:
                return "[]"
            
            # Check if it's a simple list that can go on one line
            if len(obj) <= 8 and all(not isinstance(item, (dict, list)) for item in obj):
                return json.dumps(obj)
            
            # For links array, put each link on one line
            if all(isinstance(item, list) and len(item) >= 5 for item in obj):
                lines = ["["]
                for i, link in enumerate(obj):
                    comma = "," if i < len(obj) - 1 else ""
                    lines.append(f'{" " * (indent + 4)}{json.dumps(link)}{comma}')
                lines.append(" " * indent + "]")
                return "\n".join(lines)
            
            # Otherwise format as multi-line
            lines = ["["]
            for i, item in enumerate(obj):
                comma = "," if i < len(obj) - 1 else ""
                item_str = compact_json_dumps(item, indent + 4)
                if isinstance(item, dict):
                    lines.append(" " * (indent + 4) + item_str.lstrip() + comma)
                else:
                    lines.append(f'{" " * (indent + 4)}{item_str}{comma}')
            lines.append(" " * indent + "]")
            return "\n".join(lines)
        
        else:
            return json.dumps(obj)
    
    # Convert to compact JSON string
    json_str = compact_json_dumps(obj)
    
    # Replace JSON literals with Python literals
    python_str = json_str.replace('true', 'True')
    python_str = python_str.replace('false', 'False')
    python_str = python_str.replace('null', 'None')
    
    return python_str


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_minimal_workflow.py WORKFLOW_NAME [--add_metadata]", file=sys.stderr)
        print("Example: python extract_minimal_workflow.py MNIST_Test", file=sys.stderr)
        print("         python extract_minimal_workflow.py MNIST_Test --add_metadata", file=sys.stderr)
        sys.exit(1)
    
    workflow_name = sys.argv[1]
    add_test_metadata = "--add_metadata" in sys.argv
    
    # Construct path to workflow file
    workflow_path = Path("user/default/workflows") / f"{workflow_name}.json"
    
    if not workflow_path.exists():
        print(f"Error: Workflow file not found: {workflow_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load workflow
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        
        # Extract minimal version
        minimal = extract_minimal_workflow(workflow_data, add_test_metadata)
        
        # Output as Python code
        print(f"# Minimal version of {workflow_name} workflow")
        print(f"{workflow_name.upper().replace(' ', '_')}_MINIMAL = {json_to_python_literal(minimal)}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in workflow file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()