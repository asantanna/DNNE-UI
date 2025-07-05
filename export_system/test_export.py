#!/usr/bin/env python3
"""
Test export of MNIST Test workflow
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def main():
    print("Testing MNIST Test Workflow Export...")
    print("=" * 60)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f" Loaded export system with {len(exporter.node_registry)} node types")
    
    # Load MNIST Test workflow
    workflow_path = Path("user/default/workflows/MNIST Test.json")
    if not workflow_path.exists():
        print(f"L Workflow not found: {workflow_path}")
        return
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print(f" Loaded workflow with {len(workflow.get('nodes', []))} nodes")
    
    # Export the workflow
    try:
        output_path = Path("export_system/exports/MNIST-Test")
        result = exporter.export_workflow(workflow, output_path)
        print(" Export successful!")
        print(f" Generated package at: {output_path}")
        
    except Exception as e:
        print(f"L Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()