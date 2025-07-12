#!/usr/bin/env python3
"""
Test script for the new modular export system
"""

import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def main():
    print("Testing DNNE Modular Export System...")
    print("=" * 60)
    
    # Create exporter and register nodes
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f"✅ Registered {len(exporter.node_registry)} node types")
    
    # Load MNIST Test workflow
    workflow_path = Path("user/default/workflows/MNIST Test.json")
    if not workflow_path.exists():
        print(f"❌ Cannot find workflow file: {workflow_path}")
        return
    
    print(f"📄 Loading workflow: {workflow_path}")
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Convert workflow format if needed (handle 'type' vs 'class_type')
    for node in workflow.get("nodes", []):
        if "type" in node and "class_type" not in node:
            node["class_type"] = node["type"]
        
        # Also convert widgets_values to inputs dict format
        if "widgets_values" in node and isinstance(node.get("inputs", []), list):
            # Create inputs dict from widgets_values
            inputs_dict = {}
            widget_names = []
            for inp in node.get("inputs", []):
                if "widget" in inp and inp["widget"]:
                    widget_names.append(inp["name"])
            
            # Map widget values to names
            for i, value in enumerate(node.get("widgets_values", [])):
                if i < len(widget_names):
                    inputs_dict[widget_names[i]] = value
            
            node["inputs"] = inputs_dict
    
    # Export to a new directory
    export_dir = Path("export_system/exports/MNIST-Test")
    
    print(f"📦 Exporting to: {export_dir}")
    print("=" * 60)
    
    try:
        # Export the workflow
        runner_path = exporter.export_workflow(workflow, export_dir)
        print(f"✅ Export successful!")
        print(f"   Runner path: {runner_path}")
        
        # List the generated files
        print("\n📁 Generated package structure:")
        for root, dirs, files in os.walk(export_dir):
            level = root.replace(str(export_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Show runner.py content
        print("\n📄 runner.py content (first 50 lines):")
        print("-" * 60)
        runner_content = Path(runner_path).read_text()
        lines = runner_content.splitlines()
        for i, line in enumerate(lines[:50]):
            print(f"{i+1:3d}: {line}")
        print("-" * 60)
        
        print(f"\n✅ Test complete! You can now run:")
        print(f"   cd {export_dir}")
        print(f"   python runner.py")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()